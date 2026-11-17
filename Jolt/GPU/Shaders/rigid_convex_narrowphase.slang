// Same numerical writer/substep as ordinary narrowphase. Host must hold the
// CollidePairs producer barrier until narrowphase_job_progress reports Success.
#include "rigid_convex_manifold.slang"
#include "component_union.slang"
ConstantBuffer<StepParams> step_params;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<BodyState> body_states;
RWStructuredBuffer<ContactPatch> contact_patches;
RWStructuredBuffer<Contact> contacts;
StructuredBuffer<uint4> convex_geometry;
RWStructuredBuffer<uint4> convex_scratch;
StructuredBuffer<BodyModel> body_models;
RWStructuredBuffer<SweepBody> sweep_bodies;
RWStructuredBuffer<Atomic<uint>> sweep_choice;

uint cvx_step_slots()
{
    uint words,stride; convex_scratch.GetDimensions(words,stride);
    return words>=4u?min(32u,(words-4u)/CvxSlotWords):0u;
}
void cvx_initialize_jobs(uint index)
{
    uint words,stride; convex_scratch.GetDimensions(words,stride);
    if(index>=words) return;
    uint4 value=uint4(0); uint slots=cvx_step_slots();
    if(index==0u) value=uint4(slots==0u?QueryNumericalFailure:QueryPending,0,0,0);
    if(index==1u) value=uint4(stage_work.range.x,stage_work.range.x<step_params.solve.w?min(stage_work.range.y,step_params.solve.w-stage_work.range.x):0u,slots,step_params.selection.x);
    if(index>=4u && (index-4u)%CvxSlotWords==48u) value=uint4(stage_work.range.x+(index-4u)/CvxSlotWords,0,0,0);
    convex_scratch[index]=value;
}

[shader("compute")]
[numthreads(64,1,1)]
void narrowphase_init_jobs(uint3 tid:SV_DispatchThreadID) { cvx_initialize_jobs(tid.x); }
void cvx_step_next(uint b,uint slots)
{
    uint4 owner=convex_scratch[b+48u]; owner.x+=slots; owner.yz=uint2(0); convex_scratch[b+48u]=owner;
}
void cvx_step_start(uint b,uint slots)
{
    uint4 owner=convex_scratch[b+48u],job=convex_scratch[1];
    if(owner.x-job.x>=job.y) { convex_scratch[b+48u].w=1u; return; }
    uint patchIndex=contact_patches[0].leaves.x+owner.x;
    ContactPatch patch=contact_patches[patchIndex];
    if(patch.leaves.x>=step_params.counts.z || patch.leaves.y>=step_params.counts.z) { cvx_fail(convex_scratch,b,40); convex_scratch[b+48u].w=2; return; }
    ShapeModel a=shape_models[patch.leaves.x],p=shape_models[patch.leaves.y];
    if(a.meta.x<=ShapeCapsule && p.meta.x<=ShapeCapsule) { cvx_step_next(b,slots); return; }
    CvxOperand oa=cvx_leaf_operand(a,body_states[a.meta.y]),op=cvx_leaf_operand(p,body_states[p.meta.y]);
    if(cvx_triangle_backfacing(oa,op.center.xyz) || cvx_triangle_backfacing(op,oa.center.xyz)) {
        cvx_step_next(b,slots); return;
    }
    float scale;
    if(!cvx_pair_scale(convex_geometry,oa,op,scale)) { cvx_fail(convex_scratch,b,7u); convex_scratch[b+48u].w=2u; return; }
    cvx_begin(convex_scratch,b,oa,op,oa.center.xyz,scale,1e-4f,false);
    convex_scratch[b+48u].y=1u;
}
void cvx_step_metric_complete(uint b,uint slots)
{
    uint ordinal=convex_scratch[b+48u].x;
    ContactPatch patch=contact_patches[contact_patches[0].leaves.x+ordinal];
    ShapeModel a=shape_models[patch.leaves.x],p=shape_models[patch.leaves.y];
    bool sensor=jph_sensor(a)||jph_sensor(p);
    float band=sensor?0.0f:step_params.contact.x;
    if(cvx_f(convex_scratch,b,7).w>band) { cvx_step_next(b,slots); return; }
    if(sensor) { cvx_manifold_primary(convex_scratch,b); convex_scratch[b+49u]=uint4(0); convex_scratch[b+48u].y=3u; return; }
    BodyState sa=body_states[a.meta.y],sp=body_states[p.meta.y];
    CvxOperand oa=cvx_leaf_operand(a,sa),op=cvx_leaf_operand(p,sp);
    float3 normal=cvx_f(convex_scratch,b,7).xyz;
    if(cvx_is_triangle(oa))
        normal=-cvx_triangle_fix_normal(convex_geometry,oa,cvx_f(convex_scratch,b,8).xyz,-normal,
                                         sp.linearVelocity.xyz-sa.linearVelocity.xyz);
    else if(cvx_is_triangle(op))
        normal=cvx_triangle_fix_normal(convex_geometry,op,cvx_f(convex_scratch,b,9).xyz,normal,
                                        sa.linearVelocity.xyz-sp.linearVelocity.xyz);
    float4 metric=cvx_f(convex_scratch,b,7); metric.xyz=normal; cvx_put(convex_scratch,b,7,metric);
    cvx_manifold_begin(convex_scratch,b,max(band,1e-4f),body_states[a.meta.y].positionCOM.xyz);
    convex_scratch[b+48u].y=2u;
}
void cvx_step_contact(uint b,uint slots)
{
    uint ordinal=convex_scratch[b+48u].x,point=convex_scratch[b+49u].x;
    uint patchIndex=contact_patches[0].leaves.x+ordinal;
    ContactPatch patch=contact_patches[patchIndex];
    ShapeModel a=shape_models[patch.leaves.x],p=shape_models[patch.leaves.y];
    BodyState sa=body_states[a.meta.y],sb=body_states[p.meta.y];
    bool sensor=jph_sensor(a)||jph_sensor(p); uint count=convex_scratch[b+36u].w;
    Contact result; result.meta=uint4(a.meta.y,p.meta.y,0,0);
    result.normalDepth=result.localAnchorA=result.localAnchorB=result.impulse=float4(0);
    if(point<count) {
        float3 pa=cvx_f(convex_scratch,b,CvxManifoldOut+3u*point).xyz,pb=cvx_f(convex_scratch,b,CvxManifoldOut+3u*point+1u).xyz;
        float3 normal=cvx_f(convex_scratch,b,7).xyz;
        uint feature=convex_scratch[b+CvxManifoldOut+3u*point+2u].x;
        result.meta.zw=uint2(sensor?0u:1u,(a.meta.x<<28u)|(p.meta.x<<24u)|(feature&0x00ffffffu));
        result.normalDepth=float4(normal,dot(pa-pb,normal));
        result.localAnchorA=float4(jph_inverse_rotate(sa.rotation,pa-sa.positionCOM.xyz),0);
        result.localAnchorB=float4(jph_inverse_rotate(sb.rotation,pb-sb.positionCOM.xyz),0);
        if(!all(isfinite(result.normalDepth)) || !all(isfinite(result.localAnchorA)) || !all(isfinite(result.localAnchorB))) {
            cvx_fail(convex_scratch,b,41); convex_scratch[b+48u].w=2; return;
        }
    }
    contacts[ordinal*MaxContactPointsPerPair+point]=result;
    ++convex_scratch[b+49u].x;
    if(point+1u==MaxContactPointsPerPair) {
        patch.leaves.z=(sensor?PatchSensor:PatchActive)|(cvx_f(convex_scratch,b,7).w<=0?PatchOverlapping:0u);
        patch.leaves.w=count; contact_patches[patchIndex]=patch; cvx_step_next(b,slots);
    }
}
[shader("compute")]
[numthreads(64,1,1)]
void narrowphase_advance_jobs(uint3 tid:SV_DispatchThreadID)
{
    uint slots=cvx_step_slots(); if(tid.x>=slots) return;
    uint b=4u+tid.x*CvxSlotWords;
    for(uint unit=0u;unit<32u && convex_scratch[b+48u].w==0u;++unit) {
        uint phase=convex_scratch[b+48u].y;
        if(phase==0u) cvx_step_start(b,slots);
        else if(phase==1u) {
            cvx_advance(convex_geometry,convex_scratch,b,1u);
            if(convex_scratch[b].x==CvxSuccess) cvx_step_metric_complete(b,slots);
        } else if(phase==2u) {
            cvx_manifold_advance(convex_geometry,convex_scratch,b,1u);
            if(convex_scratch[b].x==CvxSuccess) { convex_scratch[b+49u]=uint4(0); convex_scratch[b+48u].y=3u; }
        } else cvx_step_contact(b,slots);
        if(phase!=0u && convex_scratch[b].x==CvxFailure) convex_scratch[b+48u].w=2u;
    }
}
void cvx_step_progress()
{
    uint slots=cvx_step_slots(),complete=0; uint4 summary=uint4(QuerySuccess,0,0,0);
    for(uint i=0u;i<slots;++i) {
        uint b=4u+i*CvxSlotWords; uint4 owner=convex_scratch[b+48u];
        if(owner.w==2u) { summary=uint4(QueryNumericalFailure,complete,owner.x,convex_scratch[b].w); break; }
        if(owner.w==1u) ++complete;
        else summary.x=QueryPending;
    }
    if(slots==0u) summary=uint4(QueryNumericalFailure,0,0,42);
    summary.y=complete; convex_scratch[0]=summary;
}

[shader("compute")]
[numthreads(64,1,1)]
void narrowphase_job_progress(uint3 tid:SV_DispatchThreadID)
{
    if (tid.x == 0u) cvx_step_progress();
}

void cvx_sweep_start(uint base, uint slots)
{
    uint ordinal = convex_scratch[base + 48u].x;
    uint4 job = convex_scratch[1];
    if (ordinal - job.x >= job.y)
    {
        convex_scratch[base + 48u].w = 1u;
        return;
    }
    uint2 leaves = contact_patches[contact_patches[0].leaves.x + ordinal].leaves.xy;
    ShapeModel a = shape_models[leaves.x], b = shape_models[leaves.y];
    SweepBody ma = sweep_bodies[a.meta.y - step_params.selection.y];
    SweepBody mb = sweep_bodies[b.meta.y - step_params.selection.y];
    bool twoDynamic = ma.displacement.w != 0.0f && mb.displacement.w != 0.0f;
    float radiusA = body_models[a.meta.y].invInertiaDiagonal.w;
    float radiusB = body_models[b.meta.y].invInertiaDiagonal.w;
    // Dynamic pairs are driven by the opted-in endpoint. With two such
    // endpoints the smaller radius gives the conservative travel threshold.
    bool swap = twoDynamic ? (radiusB < 0.0f && (radiusA >= 0.0f || radiusB > radiusA))
                           : ma.displacement.w <= 0.0f;
    ShapeModel driver = swap ? b : a, target = swap ? a : b;
    SweepBody from = swap ? mb : ma, to = swap ? ma : mb;
    BodyState sa = jph_sweep_state(from), sb = jph_sweep_state(to);
    sb.positionCOM.xyz -= sa.positionCOM.xyz;
    sa.positionCOM.xyz = float3(0.0f);
    CvxOperand oa = cvx_leaf_operand(driver, sa), ob = cvx_leaf_operand(target, sb);
    float3 relative = from.displacement.xyz - to.displacement.xyz;
    float distance = length(relative);
    float radius = abs(swap ? radiusB : radiusA);
    if (distance <= radius * step_params.restitution.y || cvx_triangle_backfacing(ob, oa.center.xyz))
    {
        cvx_step_next(base, slots);
        return;
    }
    convex_scratch[base + 49u] = uint4(swap ? leaves.yx : leaves, driver.meta.y, target.meta.y);
    cvx_cast_begin(convex_geometry, convex_scratch, base, oa, ob, relative / distance, distance);
    convex_scratch[base + 48u].y = 1u;
}

bool cvx_sweep_hit(uint base)
{
    uint4 identity = convex_scratch[base + 49u];
    SweepBody a = sweep_bodies[identity.z - step_params.selection.y];
    SweepBody b = sweep_bodies[identity.w - step_params.selection.y];
    float4 sweep = cvx_f(convex_scratch, base, CvxCastMotion);
    float4 metric = cvx_f(convex_scratch, base, 7u);
    float fraction = cvx_cast_distance(convex_scratch, base) / sweep.w;
    float3 origin = cvx_cast_origin(convex_scratch, base);
    float3 pa = cvx_f(convex_scratch, base, 8u).xyz;
    float3 pb = cvx_f(convex_scratch, base, 9u).xyz;
    CvxOperand target = cvx_operand(convex_scratch, base, 1u);
    float3 normal = metric.xyz;
    if (cvx_is_triangle(target))
        normal = cvx_triangle_fix_normal(convex_geometry, target, pb, normal, sweep.xyz);
    float normalTravel = dot(normal, sweep.xyz) * sweep.w;
    if (normalTravel <= 0.0f) return false;
    float3 ra = pa + origin + (b.displacement.xyz - a.displacement.xyz) * fraction;
    float3 rb = pb + origin - (b.positionCOM.xyz - a.positionCOM.xyz);
    float3 relativeVelocity = a.linearVelocity.xyz + cross(a.angularVelocity.xyz, ra)
                            - b.linearVelocity.xyz - cross(b.angularVelocity.xyz, rb);
    if (dot(normal, relativeVelocity) <= 0.0f) return false;
    // Apply Jolt's penetration allowance along the relative normal travel.
    float radius = abs(body_models[identity.z].invInertiaDiagonal.w);
    float slop = min(step_params.contact.y, radius * step_params.restitution.z);
    float cutoff = min(1.0f, fraction + slop / normalTravel);
    Contact hit;
    hit.meta = uint4(identity.zw, 1u, 0u);
    hit.normalDepth = float4(normal, fraction);
    hit.localAnchorA = float4(ra, 0.0f);
    hit.localAnchorB = float4(rb, 0.0f);
    hit.impulse = float4(0.0f, 0.0f, 0.0f, cutoff);
    contacts[convex_scratch[base + 48u].x] = hit;
    uint count = step_params.selectedCounts.x;
    uint2 edge = identity.zw - step_params.selection.y;
    sweep_choice[count + edge.x].min(asuint(cutoff));
    if (b.displacement.w == 0.0f) return false;
    sweep_choice[3u * count].store(1u);
    convex_scratch[base + 50u] = uint4(edge, 0u, 0u);
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_advance_jobs(uint3 tid : SV_DispatchThreadID)
{
    uint slots = cvx_step_slots();
    if (tid.x >= slots) return;
    uint base = 4u + tid.x * CvxSlotWords;
    for (uint unit = 0u; unit < 32u && convex_scratch[base + 48u].w == 0u; ++unit)
    {
        if (convex_scratch[base + 48u].y == 0u)
        {
            cvx_sweep_start(base, slots);
            continue;
        }
        if (convex_scratch[base + 48u].y == 2u)
        {
            uint2 edge = jph_join_components(sweep_choice, convex_scratch[base + 50u].xy, 1u);
            if (edge.x == edge.y) cvx_step_next(base, slots);
            else convex_scratch[base + 50u].xy = edge;
            continue;
        }
        uint status = cvx_cast_unit(convex_geometry, convex_scratch, base);
        if (status == CvxCastPending) continue;
        if (status == CvxCastError)
            convex_scratch[base + 48u].w = 2u;
        else
        {
            if (status == CvxCastHit && cvx_sweep_hit(base))
                convex_scratch[base + 48u].y = 2u;
            else
                cvx_step_next(base, slots);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_initialize_jobs(uint3 tid : SV_DispatchThreadID)
{
    cvx_initialize_jobs(tid.x);
    uint count = step_params.selectedCounts.x;
    // Contiguous parent, cutoff and ordinal arrays, then the dynamic-hit flag.
    if (tid.x < count)
    {
        sweep_choice[tid.x].store(tid.x);
        sweep_choice[count + tid.x].store(0xffffffffu);
        sweep_choice[2u * count + tid.x].store(0xffffffffu);
    }
    if (tid.x == 0u) sweep_choice[3u * count].store(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_job_progress(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    cvx_step_progress();
    if (convex_scratch[0].x == QuerySuccess)
        convex_scratch[0].w = sweep_choice[3u * step_params.selectedCounts.x].load();
}

// Every pass shortens a chain by at least a factor of 32. Host derives the
// pass count from the body range, and emits complete passes in queue order.
[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_flatten_parents(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint local = stage_work.range.x + tid.x;
    uint root = sweep_choice[local].load();
    if (root == local) return;
    for (uint hop = 0u; hop < 32u; ++hop)
    {
        uint parent = sweep_choice[root].load();
        if (parent == root) break;
        root = parent;
    }
    sweep_choice[local].store(root);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_reduce_times(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint local = stage_work.range.x + tid.x;
    uint root = sweep_choice[local].load();
    uint count = step_params.selectedCounts.x;
    if (root != local)
        sweep_choice[count + root].min(sweep_choice[count + local].load());
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_store_cutoffs(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint local = stage_work.range.x + tid.x;
    uint root = sweep_choice[local].load();
    uint cutoff = sweep_choice[step_params.selectedCounts.x + root].load();
    // Clip this connected set to its first trustworthy linear sweep. A later
    // hit predicted on a trajectory changed by the first impulse is stale.
    if (cutoff != 0xffffffffu)
        sweep_bodies[local].linearResponse.w = asfloat(cutoff);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_select_pair(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint ordinal = stage_work.range.x + tid.x;
    Contact hit = contacts[ordinal];
    if (hit.meta.z == 0u) return;
    uint root = sweep_choice[hit.meta.x - step_params.selection.y].load();
    uint count = step_params.selectedCounts.x;
    if (sweep_choice[count + root].load() == asuint(hit.impulse.w))
        sweep_choice[2u * count + root].min(ordinal);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_apply_impact(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint ordinal = stage_work.range.x + tid.x;
    uint index = contact_patches[0].leaves.x + ordinal;
    ContactPatch patch = contact_patches[index];
    patch.leaves.zw = uint2(0u);
    Contact hit = contacts[ordinal];
    if (hit.meta.z == 0u)
    {
        contact_patches[index] = patch;
        return;
    }
    uint i = hit.meta.x - step_params.selection.y;
    if (sweep_choice[2u * step_params.selectedCounts.x + sweep_choice[i].load()].load() != ordinal)
    {
        contact_patches[index] = patch;
        return;
    }
    uint j = hit.meta.y - step_params.selection.y;
    SweepBody a = sweep_bodies[i], b = sweep_bodies[j];
    BodyState sa = jph_sweep_state(a), sb = jph_sweep_state(b);
    BodyModel ma = body_models[hit.meta.x], mb = body_models[hit.meta.y];
    // The target may be a temporarily driven authored Static. Its captured
    // point velocity still participates, with zero inverse mass.
    bool dynamicTarget = jph_dynamic(mb);
    if (!dynamicTarget) mb.meta.z = MotionKinematic;
    float3 n = hit.normalDepth.xyz, ra = hit.localAnchorA.xyz, rb = hit.localAnchorB.xyz;
    float3 incoming = jph_point_velocity(ma, sa, ra) - jph_point_velocity(mb, sb, rb);
    float speed = dot(incoming, n);
    if (shape_models[patch.leaves.x].meta.y != hit.meta.x) patch.leaves.xy = patch.leaves.yx;
    ShapeModel ashape = shape_models[patch.leaves.x], bshape = shape_models[patch.leaves.y];
    float restitution = speed > step_params.restitution.x
        ? jph_combine_material(ashape.material.y, ashape.surface.w, bshape.material.y, bshape.surface.w) : 0.0f;
    float normalImpulse = (1.0f + restitution) * speed * jph_effective_mass(ma, sa, ra, mb, sb, rb, n);
    jph_apply_impulse(ma, sa, ra, mb, sb, rb, n, normalImpulse);
    float3 relative = jph_point_velocity(ma, sa, ra) - jph_point_velocity(mb, sb, rb);
    float3 tangent = relative - n * dot(n, relative);
    float friction = jph_combine_material(ashape.material.x, ashape.surface.z, bshape.material.x, bshape.surface.z);
    float tangentSpeed = length(tangent), tangentImpulse = 0.0f;
    if (tangentSpeed > 0.0f && friction > 0.0f)
    {
        tangent /= tangentSpeed;
        tangentImpulse = min(friction * normalImpulse, tangentSpeed * jph_effective_mass(ma, sa, ra, mb, sb, rb, tangent));
        jph_apply_impulse(ma, sa, ra, mb, sb, rb, tangent, tangentImpulse);
    }
    float fraction = hit.normalDepth.w;
    float3 linear = jph_clamp_speed(sa.linearVelocity.xyz, ma.damping.z);
    float3 angular = jph_clamp_speed(sa.angularVelocity.xyz, ma.damping.w);
    sweep_bodies[i].linearResponse = float4(linear - a.linearVelocity.xyz, hit.impulse.w);
    sweep_bodies[i].angularResponse = float4(angular - a.angularVelocity.xyz, 1.0f);
    if (dynamicTarget)
    {
        float3 linearB = jph_clamp_speed(sb.linearVelocity.xyz, mb.damping.z);
        float3 angularB = jph_clamp_speed(sb.angularVelocity.xyz, mb.damping.w);
        sweep_bodies[j].linearResponse = float4(linearB - b.linearVelocity.xyz, hit.impulse.w);
        sweep_bodies[j].angularResponse = float4(angularB - b.angularVelocity.xyz, 1.0f);
    }
    // Publish the witness and incoming velocity at the impact, before the
    // common integrator changes COM pose. The report reads this one-point row.
    hit.normalDepth.w = 0.0f;
    hit.localAnchorA = float4(a.positionCOM.xyz + fraction * a.displacement.xyz + ra, 0.0f);
    hit.localAnchorB = float4(-incoming, 0.0f);
    hit.impulse = float4(n * normalImpulse + tangent * tangentImpulse, 0.0f);
    contacts[ordinal] = hit;
    patch.leaves.zw = uint2(PatchActive, 1u);
    contact_patches[index] = patch;
}
