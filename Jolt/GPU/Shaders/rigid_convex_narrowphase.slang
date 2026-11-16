// Same numerical writer/substep as ordinary narrowphase. Host must hold the
// CollidePairs producer barrier until narrowphase_job_progress reports Success.
#include "rigid_convex_manifold.slang"
ConstantBuffer<StepParams> step_params;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<BodyState> body_states;
RWStructuredBuffer<ContactPatch> contact_patches;
RWStructuredBuffer<Contact> contacts;
StructuredBuffer<uint4> convex_geometry;
RWStructuredBuffer<uint4> convex_scratch;

uint cvx_step_slots()
{
    uint words,stride; convex_scratch.GetDimensions(words,stride);
    return words>=4u?min(32u,(words-4u)/CvxSlotWords):0u;
}
CvxOperand cvx_step_operand(ShapeModel shape,BodyState state)
{
    CvxOperand o; o.meta=uint4(shape.meta.x,shape.meta.w,0,0);
    o.center=float4(state.positionCOM.xyz+jph_rotate(state.rotation,shape.localPositionCOM.xyz),0);
    o.rotation=jph_quat_mul(state.rotation,shape.localRotation); o.geometry=shape.geometry;
    o.geometryReference=shape.geometryReference; return o;
}
bool cvx_operand_extent(StructuredBuffer<uint4> g,CvxOperand o,out float extent)
{
    extent=0;
    if(o.meta.x==CvxHull) {
        uint count,first;
        if(!cvx_hull_header(g,o,count,first,true)) return false;
        float3 low=asfloat(g[o.geometryReference.x+3u]).xyz,high=asfloat(g[o.geometryReference.x+4u]).xyz;
        float3 v=max(abs(low),abs(high)); extent=max(v.x,max(v.y,v.z));
    } else if(o.meta.x==CvxCylinder) extent=max(o.geometry.x,o.geometry.y);
    else if(o.meta.x==CvxCapsule) extent=o.geometry.y+o.geometry.w;
    else if(o.meta.x==CvxBox) extent=max(o.geometry.x,max(o.geometry.y,o.geometry.z));
    else extent=o.geometry.w;
    return true;
}
bool cvx_pair_scale(StructuredBuffer<uint4> g,CvxOperand a,CvxOperand p,out float scale)
{
    scale=0;
    float extentA,extentP;
    if(!cvx_operand_extent(g,a,extentA) || !cvx_operand_extent(g,p,extentP)) return false;
    float3 d=abs(p.center.xyz-a.center.xyz);
    scale=max(max(d.x,max(d.y,d.z)),max(extentA,extentP));
    return true;
}

[shader("compute")]
[numthreads(64,1,1)]
void narrowphase_init_jobs(uint3 tid:SV_DispatchThreadID)
{
    uint words,stride; convex_scratch.GetDimensions(words,stride);
    if(tid.x>=words) return;
    uint4 value=uint4(0); uint slots=cvx_step_slots();
    if(tid.x==0u) value=uint4(slots==0u?QueryNumericalFailure:QueryPending,0,0,0);
    if(tid.x==1u) value=uint4(stage_work.range.x,stage_work.range.x<step_params.solve.w?min(stage_work.range.y,step_params.solve.w-stage_work.range.x):0u,slots,step_params.selection.x);
    if(tid.x>=4u && (tid.x-4u)%CvxSlotWords==48u) value=uint4(stage_work.range.x+(tid.x-4u)/CvxSlotWords,0,0,0);
    convex_scratch[tid.x]=value;
}
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
    CvxOperand oa=cvx_step_operand(a,body_states[a.meta.y]),op=cvx_step_operand(p,body_states[p.meta.y]);
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
    CvxOperand oa=cvx_step_operand(a,sa),op=cvx_step_operand(p,sp);
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
[shader("compute")]
[numthreads(64,1,1)]
void narrowphase_job_progress(uint3 tid:SV_DispatchThreadID)
{
    if(tid.x!=0u) return;
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
