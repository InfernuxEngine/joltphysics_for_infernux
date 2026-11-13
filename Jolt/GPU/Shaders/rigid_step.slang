// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021-2026 Jorrit Rouwe
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_joints.slang"

ConstantBuffer<StepParams> step_params;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<WorldRange> worlds;
RWStructuredBuffer<BodyForces> body_forces;
RWStructuredBuffer<BodyState> body_states;
RWStructuredBuffer<Aabb> bounds;
RWStructuredBuffer<Contact> contacts;
RWStructuredBuffer<KinematicTarget> kinematic_targets;
RWStructuredBuffer<ContactPatch> contact_patches;

void jph_joint_bodies(JointAuthor author, out BodyModel m1, out BodyState s1,
    out BodyModel m2, out BodyState s2)
{
    m1 = jph_joint_world_model();
    s1 = jph_joint_world_state();
    if (author.bodies.x != InvalidQueryBodySlot)
    {
        m1 = jph_effective_model(body_models[author.bodies.x], kinematic_targets[author.bodies.x]);
        s1 = body_states[author.bodies.x];
    }
    m2 = jph_effective_model(body_models[author.bodies.y], kinematic_targets[author.bodies.y]);
    s2 = body_states[author.bodies.y];
}

// A world lane alone owns both endpoints. Each call consumes one joint and
// each dispatch visits only the explicitly selected local joint range.
void jph_step_joint(uint index, uint phase)
{
    JointAuthor author = jph_joint_load_author(index);
    JointState runtime = jph_joint_load_state(index);
    BodyModel m1, m2;
    BodyState s1, s2;
    jph_joint_bodies(author, m1, s1, m2, s2);
    if (!jph_resident(m1) || !jph_resident(m2))
    {
        jph_joint_clear_impulses(runtime);
        jph_joint_store_state(index, runtime);
        return;
    }
    BodyState old1 = s1, old2 = s2;
    if (phase == 1u)
        jph_joint_solve_velocity(author, runtime, m1, s1, m2, s2, step_params.gravityDt.w);
    else if (phase == 2u)
        jph_joint_solve_position(author, runtime, m1, s1, m2, s2, step_params.contact.w);
    else
        jph_joint_warm_start(author, runtime, m1, s1, m2, s2);
    if (runtime.flags.y != JointStatusSuccess || !jph_joint_finite(s1) || !jph_joint_finite(s2)
        || !jph_joint_state_finite(runtime))
    {
        if (runtime.flags.y == JointStatusSuccess)
            runtime.flags.y = JointStatusNumericalFailure;
        s1 = old1;
        s2 = old2;
    }
    if (jph_dynamic(m1)) body_states[author.bodies.x] = s1;
    if (jph_dynamic(m2)) body_states[author.bodies.y] = s2;
    jph_joint_store_state(index, runtime);
}

void jph_step_world_joints(uint world_lane, uint phase)
{
    if (world_lane >= stage_work.range.w) return;
    uint world_index = stage_work.range.z + world_lane;
    if (world_index >= step_params.counts.y) return;
    WorldRange world = worlds[world_index];
    uint first = stage_work.range.x;
    if (first >= world.leaves.w) return;
    uint end = first + min(stage_work.range.y, world.leaves.w - first);
    for (uint local = first; local < end; ++local)
        jph_step_joint(world.leaves.z + local, phase);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void setup_joints(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (index >= joint_stream[0].value.x) return;
    JointAuthor author = jph_joint_load_author(index);
    JointState runtime = jph_joint_load_state(index);
    BodyModel m1, m2;
    BodyState s1, s2;
    jph_joint_bodies(author, m1, s1, m2, s2);
    if (!jph_resident(m1) || !jph_resident(m2))
        jph_joint_clear_impulses(runtime);
    else
    {
        // This stage stores runtime only. All global setup units precede any
        // warm unit, so shared endpoints are read before warm impulses change w.
        jph_joint_setup(author, runtime, m1, s1, m2, s2,
            step_params.gravityDt.w, step_params.counts.w != 0u);
        if (runtime.flags.y == JointStatusSuccess
            && (!jph_joint_finite(s1) || !jph_joint_finite(s2) || !jph_joint_state_finite(runtime)))
            runtime.flags.y = JointStatusNumericalFailure;
    }
    jph_joint_store_state(index, runtime);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void warm_joints(uint3 tid : SV_DispatchThreadID)
{
    jph_step_world_joints(tid.x, 3u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_joint_velocity(uint3 tid : SV_DispatchThreadID)
{
    jph_step_world_joints(tid.x, 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_joint_position(uint3 tid : SV_DispatchThreadID)
{
    jph_step_world_joints(tid.x, 2u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_joints(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (index >= joint_stream[0].value.x) return;
    JointState runtime = jph_joint_load_state(index);
    runtime.rotor.w = step_params.gravityDt.w;
    jph_joint_store_state(index, runtime);
}

// Only the primitive geometry uses this local pose. Body state, principal
// inertia and contact anchors remain in the original body COM frame.
float3 jph_leaf_position(ShapeModel shape, BodyState state)
{
    return state.positionCOM.xyz + jph_rotate(state.rotation, shape.localPositionCOM.xyz);
}

float4 jph_leaf_rotation(ShapeModel shape, BodyState state)
{
    return jph_quat_mul(state.rotation, shape.localRotation);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prepare_kinematics(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (index >= step_params.counts.x)
        return;
    BodyModel model = body_models[index];
    if (!jph_resident(model))
        return;
    KinematicTarget target = kinematic_targets[index];
    uint motion = model.meta.z & ~MotionNotResident;
    if (motion != MotionKinematic &&
        !(motion == MotionStatic && (target.state.x & TargetTemporaryStatic) != 0u))
        return;
    BodyState state = body_states[index];
    if ((target.state.x & TargetPending) != 0u)
    {
        // Capture this logical tick's start once. Requested time belongs to
        // the command; only later Integrate dispatches use substep time h.
        float requested_dt = target.rootCOMDt.w;
        float3 root_com = target.rootCOMDt.xyz;
        float3 origin = state.positionCOM.xyz - jph_rotate(state.rotation, root_com);
        float3 delta = target.positionSpeed.xyz - origin;
        float maximum_distance = target.positionSpeed.w * requested_dt;
        float distance = length(delta);
        if (target.positionSpeed.w > 0.0f && distance > maximum_distance)
        {
            // Preserve the old origin-based transform-drive cap exactly:
            // excess translation AND rotation teleport before the last stretch.
            origin = target.positionSpeed.xyz - delta * (maximum_distance / distance);
            state.rotation = target.rotation;
            state.positionCOM.xyz = origin + jph_rotate(state.rotation, root_com);
        }
        float3 target_com = target.positionSpeed.xyz + jph_rotate(target.rotation, root_com);
        state.linearVelocity = float4((target_com - state.positionCOM.xyz) / requested_dt, 0.0f);
        state.angularVelocity = float4(jph_kinematic_angular_velocity(target.rotation, state.rotation, requested_dt), 0.0f);
        target.state.x = (target.state.x & TargetTemporaryStatic) | TargetActive;
    }
    else if ((target.state.x & TargetDrivenLastTick) != 0u)
    {
        state.linearVelocity = float4(0.0f);
        state.angularVelocity = float4(0.0f);
        target.state = uint4(0u);
    }
    body_states[index] = state;
    kinematic_targets[index] = target;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_forces(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (index >= step_params.counts.x)
        return;
    BodyModel model = body_models[index];
    if (!jph_dynamic(model) || !jph_resident(model))
        return;
    BodyState state = body_states[index];
    BodyForces forces = body_forces[index];
    float dt = step_params.gravityDt.w;
    // PhysicsSystem applies gyro before external force/torque and drag.
    jph_apply_gyroscopic_force(model, state, dt);
    state.linearVelocity.xyz += dt * jph_acceleration(model, forces, step_params.gravityDt.xyz);
    state.angularVelocity.xyz += dt * jph_inverse_inertia(model, state, forces.torque.xyz);
    state.linearVelocity.xyz *= max(0.0f, 1.0f - model.damping.x * dt);
    state.angularVelocity.xyz *= max(0.0f, 1.0f - model.damping.y * dt);
    state.linearVelocity.xyz = jph_clamp_speed(state.linearVelocity.xyz, model.damping.z);
    state.angularVelocity.xyz = jph_clamp_speed(state.angularVelocity.xyz, model.damping.w);
    body_states[index] = state;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_tick(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (index >= step_params.counts.x)
        return;
    if (!jph_resident(body_models[index]))
        return;
    BodyForces cleared;
    cleared.force = float4(0.0f);
    cleared.torque = float4(0.0f);
    body_forces[index] = cleared;
    KinematicTarget target = kinematic_targets[index];
    if ((target.state.x & TargetActive) != 0u)
        target.state.x = (target.state.x & TargetTemporaryStatic) | TargetDrivenLastTick;
    kinematic_targets[index] = target;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_bounds(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (index >= step_params.counts.z)
        return;
    ShapeModel shape = shape_models[index];
    BodyState state = body_states[shape.meta.y];
    float4 geometry = shape.geometry;
    float3 center = jph_leaf_position(shape, state);
    float4 rotation = jph_leaf_rotation(shape, state);
    float3 extent;
    if (shape.meta.x == ShapeSphere)
        extent = float3(geometry.w);
    else if (shape.meta.x == ShapeCapsule)
        extent = abs(jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f))) * geometry.y + geometry.w;
    else
        extent = abs(jph_rotate(rotation, float3(1.0f, 0.0f, 0.0f))) * geometry.x
               + abs(jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f))) * geometry.y
               + abs(jph_rotate(rotation, float3(0.0f, 0.0f, 1.0f))) * geometry.z;
    // Expand each leaf by half the contact band, so the pair band is exactly
    // the configured speculative distance, not twice that distance.
    extent += 0.5f * step_params.contact.x;
    Aabb result;
    result.minimum = float4(center - extent, 0.0f);
    result.maximum = float4(center + extent, 0.0f);
    bounds[index] = result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void narrowphase_candidates(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y || stage_work.range.x >= step_params.solve.w) return;
    if (tid.x >= step_params.solve.w - stage_work.range.x) return;
    uint candidate_ordinal = stage_work.range.x + tid.x;
    uint candidate_base = contact_patches[0].leaves.x;
    uint patch_index = candidate_base + candidate_ordinal; // WORD address.
    uint slot = candidate_ordinal * MaxContactPointsPerPair; // CONTACT ordinal.
    ContactPatch patch = contact_patches[patch_index];
    ShapeModel shape_a = shape_models[patch.leaves.x];
    ShapeModel shape_b = shape_models[patch.leaves.y];
    uint a = shape_a.meta.y;
    uint b = shape_b.meta.y;
    bool sensor = jph_sensor(shape_a) || jph_sensor(shape_b);
    // Eligibility was established in count/fill against the same frozen model
    // and bounds. There is no third filter that could change that contract.
    patch.leaves.z = sensor ? PatchSensor : 0u;
    patch.leaves.w = 0u;
    contact_patches[patch_index] = patch;
    Contact result;
    result.meta = uint4(a, b, 0u, 0u);
    result.normalDepth = float4(0.0f);
    result.localAnchorA = float4(0.0f);
    result.localAnchorB = float4(0.0f);
    result.impulse = float4(0.0f);
    // Every accepted candidate is a complete four-Contact writer, including
    // no-manifold and sensor candidates. Other capacity is not read/written.
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        contacts[slot + point] = result;

    BodyState sa = body_states[a];
    BodyState sb = body_states[b];
    float4 ga = shape_a.geometry;
    float4 gb = shape_b.geometry;
    float3 ca = jph_leaf_position(shape_a, sa);
    float3 cb = jph_leaf_position(shape_b, sb);
    float4 qa = jph_leaf_rotation(shape_a, sa);
    float4 qb = jph_leaf_rotation(shape_b, sb);
    float speculative_distance = sensor ? 0.0f : step_params.contact.x;
    if ((shape_a.meta.x == ShapeBox && shape_b.meta.x == ShapeBox) ||
        shape_a.meta.x == ShapeCapsule || shape_b.meta.x == ShapeCapsule)
    {
        JphBoxManifold manifold;
        bool flip = false;
        if (shape_a.meta.x == ShapeBox && shape_b.meta.x == ShapeBox)
            manifold = jph_box_box(ca, qa, ga, cb, qb, gb, speculative_distance);
        else if (shape_a.meta.x == ShapeCapsule && shape_b.meta.x == ShapeCapsule)
            manifold = jph_capsule_capsule(ca, qa, ga.y, ga.w, cb, qb, gb.y, gb.w, speculative_distance);
        else if (shape_a.meta.x == ShapeCapsule && shape_b.meta.x == ShapeSphere)
            manifold = jph_capsule_sphere(ca, qa, ga.y, ga.w, cb, gb.w, speculative_distance);
        else if (shape_a.meta.x == ShapeSphere && shape_b.meta.x == ShapeCapsule)
        {
            manifold = jph_capsule_sphere(cb, qb, gb.y, gb.w, ca, ga.w, speculative_distance);
            flip = true;
        }
        else if (shape_a.meta.x == ShapeCapsule && shape_b.meta.x == ShapeBox)
            manifold = jph_capsule_box(ca, qa, ga.y, ga.w, cb, qb, gb, speculative_distance);
        else if (shape_a.meta.x == ShapeBox && shape_b.meta.x == ShapeCapsule)
        {
            manifold = jph_capsule_box(cb, qb, gb.y, gb.w, ca, qa, ga, speculative_distance);
            flip = true;
        }
        else
            return; // Unsupported leaf kinds are rejected at author admission.
        if (manifold.count == 0u)
            return;
        if (flip)
            manifold.normal = -manifold.normal;
        patch.leaves.z |= (sensor ? 0u : PatchActive) |
                          (manifold.penetration >= 0.0f ? PatchOverlapping : 0u);
        contact_patches[patch_index] = patch;
        for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
            if (point < manifold.count)
            {
                JphBoxPoint contact = manifold.points[point];
                if (flip)
                {
                    float3 point_a = contact.pointA;
                    contact.pointA = contact.pointB;
                    contact.pointB = point_a;
                }
                result.meta = uint4(a, b, sensor ? 0u : 1u, contact.feature);
                result.normalDepth = float4(manifold.normal, contact.depth);
                result.localAnchorA = float4(jph_inverse_rotate(sa.rotation, contact.pointA - sa.positionCOM.xyz), 0.0f);
                result.localAnchorB = float4(jph_inverse_rotate(sb.rotation, contact.pointB - sb.positionCOM.xyz), 0.0f);
                contacts[slot + point] = result;
            }
        return;
    }

    float3 normal, pa, pb;
    float penetration;
    uint feature = 0u;
    if (shape_a.meta.x == ShapeSphere && shape_b.meta.x == ShapeSphere)
        jph_sphere_sphere(ca, ga.w, cb, gb.w, normal, penetration, pa, pb);
    else if (shape_a.meta.x == ShapeSphere && shape_b.meta.x == ShapeBox)
        jph_sphere_box(ca, ga.w, cb, qb, gb, normal, penetration, pa, pb, feature);
    else if (shape_a.meta.x == ShapeBox && shape_b.meta.x == ShapeSphere)
    {
        jph_sphere_box(cb, gb.w, ca, qa, ga, normal, penetration, pb, pa, feature);
        normal = -normal;
    }
    else
        return;
    if (penetration < -speculative_distance)
        return;
    patch.leaves.z |= (sensor ? 0u : PatchActive) | (penetration >= 0.0f ? PatchOverlapping : 0u);
    contact_patches[patch_index] = patch;
    result.meta.z = sensor ? 0u : 1u;
    result.meta.w = feature;
    result.normalDepth = float4(normal, penetration);
    result.localAnchorA = float4(jph_inverse_rotate(sa.rotation, pa - sa.positionCOM.xyz), 0.0f);
    result.localAnchorB = float4(jph_inverse_rotate(sb.rotation, pb - sb.positionCOM.xyz), 0.0f);
    contacts[slot] = result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void setup_contacts(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint slot = stage_work.range.x + tid.x;
    if (slot >= step_params.solve.z)
        return;
    Contact contact = contacts[slot];
    if (contact.meta.z == 0u)
        return;
    ContactPatch patch = contact_patches[contact_patches[0].leaves.x + slot / MaxContactPointsPerPair];
    ShapeModel shape_a = shape_models[patch.leaves.x];
    ShapeModel shape_b = shape_models[patch.leaves.y];
    uint a = contact.meta.x;
    uint b = contact.meta.y;
    BodyModel ma = jph_effective_model(body_models[a], kinematic_targets[a]);
    BodyModel mb = jph_effective_model(body_models[b], kinematic_targets[b]);
    BodyState sa = body_states[a];
    BodyState sb = body_states[b];
    float3 pa, pb, ra, rb;
    jph_contact_points(contact, sa, sb, pa, pb, ra, rb);
    float3 normal = contact.normalDepth.xyz;
    float normal_velocity = dot(jph_point_velocity(mb, sb, rb) - jph_point_velocity(ma, sa, ra), normal);
    float dt = step_params.gravityDt.w;
    float speculative_bias = max(0.0f, -contact.normalDepth.w / dt);
    float restitution = jph_combine_material(shape_a.material.y, shape_a.surface.w,
                                               shape_b.material.y, shape_b.surface.w);
    float bias = speculative_bias;
    if (restitution > 0.0f && normal_velocity < -step_params.restitution.x
        && normal_velocity < -speculative_bias)
    {
        // ContactConstraintManager compensates the force/gravity velocity
        // already added at the start of this substep before restitution.
        float3 relative_acceleration = jph_acceleration(mb, body_forces[b], step_params.gravityDt.xyz)
                                     - jph_acceleration(ma, body_forces[a], step_params.gravityDt.xyz);
        float force_delta_velocity = min(0.0f, dot(relative_acceleration, normal) * dt);
        bias = restitution * (normal_velocity - force_delta_velocity);
    }
    contact.impulse = float4(0.0f, 0.0f, 0.0f, bias);
    contact.localAnchorA.w = 0.0f;
    contacts[slot] = contact;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_velocity(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.w) return;
    uint world_index = stage_work.range.z + tid.x;
    if (world_index >= step_params.counts.y)
        return;
    WorldRange world = worlds[world_index];
    // One lane owns this world's bounded patch range. Accepted units preserve
    // PGS order and consume prior units' body values before the next iteration.
    uint first = stage_work.range.x;
    if (first >= world.bodies.w) return;
    uint end = first + min(stage_work.range.y, world.bodies.w - first);
    for (uint local = first; local < end; ++local)
    {
        uint patch_index = world.bodies.z + local;
        ContactPatch patch = contact_patches[contact_patches[0].leaves.x + patch_index];
        if ((patch.leaves.z & PatchActive) == 0u)
            continue;
        uint slot = patch_index * MaxContactPointsPerPair;
        // Narrowphase packs active points from the first record of a candidate.
        Contact contact = contacts[slot];
        uint a = contact.meta.x;
        uint b = contact.meta.y;
        BodyModel ma = jph_effective_model(body_models[a], kinematic_targets[a]);
        BodyModel mb = jph_effective_model(body_models[b], kinematic_targets[b]);
        BodyState sa = body_states[a];
        BodyState sb = body_states[b];
        // ContactConstraintManager uses one linear-friction center per patch,
        // plus normal-axis angular friction for multiple contact points. Both
        // precede all non-penetration constraints in this PGS iteration.
        float3 normal = contact.normalDepth.xyz;
        float3 center = float3(0.0f);
        float normal_impulse = 0.0f;
        uint point_count = 0u;
        float3 ja[4];
        float3 jb[4];
        float4 old_normal_impulse = float4(0.0f);
        float4 normal_bias = float4(0.0f);
        for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        {
            Contact current = contacts[slot + point];
            if (current.meta.z == 0u)
                continue;
            float3 pa, pb, ra, rb;
            jph_contact_points(current, sa, sb, pa, pb, ra, rb);
            ja[point] = cross(ra, normal);
            jb[point] = cross(rb, normal);
            old_normal_impulse[point] = current.impulse.x;
            normal_bias[point] = current.impulse.w;
            center += 0.5f * (pa + pb);
            normal_impulse += current.impulse.x;
            ++point_count;
        }
        center /= float(point_count);
        ShapeModel shape_a = shape_models[patch.leaves.x];
        ShapeModel shape_b = shape_models[patch.leaves.y];
        float friction = jph_combine_material(shape_a.material.x, shape_a.surface.z,
                                               shape_b.material.x, shape_b.surface.z);
        float2 linear_impulse = contact.impulse.yz;
        jph_solve_friction(ma, sa, center - sa.positionCOM.xyz, mb, sb, center - sb.positionCOM.xyz,
                           normal, friction * normal_impulse, linear_impulse);
        contact.impulse.yz = linear_impulse;

        float angular_impulse = 0.0f;
        if (point_count > 1u)
        {
            // Read the four point records again instead of holding a private
            // Contact array. Radius is measured only in the contact plane.
            float weighted_radius = 0.0f;
            for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
            {
                Contact current = contacts[slot + point];
                if (current.meta.z == 0u)
                    continue;
                float3 pa, pb, ra, rb;
                jph_contact_points(current, sa, sb, pa, pb, ra, rb);
                float3 delta = 0.5f * (pa + pb) - center;
                delta -= normal * dot(delta, normal);
                weighted_radius += length(delta) * current.impulse.x;
            }
            angular_impulse = contact.localAnchorA.w;
            jph_solve_angular_friction(ma, sa, mb, sb, normal, friction * weighted_radius, angular_impulse);
        }
        contact.localAnchorA.w = angular_impulse;
        contacts[slot] = contact;

        // Solve the pair's unilateral normal QP as one block. Coupled normal
        // impulses must not leave the last scalar point's rocking residual.
        float4 next_normal_impulse = jph_solve_normal_block(ma, sa, mb, sb, normal, ja, jb,
                                                            old_normal_impulse, normal_bias, point_count);
        float delta_linear = 0.0f;
        float3 delta_angular_a = float3(0.0f);
        float3 delta_angular_b = float3(0.0f);
        for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        {
            Contact current = contacts[slot + point];
            if (current.meta.z == 0u)
                continue;
            float delta = next_normal_impulse[point] - old_normal_impulse[point];
            delta_linear += delta;
            delta_angular_a += ja[point] * delta;
            delta_angular_b += jb[point] * delta;
            current.impulse.x = next_normal_impulse[point];
            contacts[slot + point] = current;
        }
        if (jph_dynamic(ma))
        {
            sa.linearVelocity.xyz -= ma.mass.x * delta_linear * normal;
            sa.angularVelocity.xyz -= jph_inverse_inertia(ma, sa, delta_angular_a);
            body_states[a] = sa;
        }
        if (jph_dynamic(mb))
        {
            sb.linearVelocity.xyz += mb.mass.x * delta_linear * normal;
            sb.angularVelocity.xyz += jph_inverse_inertia(mb, sb, delta_angular_b);
            body_states[b] = sb;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void integrate(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (index >= step_params.counts.x)
        return;
    BodyModel model = jph_effective_model(body_models[index], kinematic_targets[index]);
    if (!jph_nonstatic(model) || !jph_resident(model))
        return;
    BodyState state = body_states[index];
    float dt = step_params.gravityDt.w;
    // Jolt clamps Dynamic bodies after impulses. Prescribed Kinematic target
    // velocity must not be truncated by Dynamic speed limits.
    if (jph_dynamic(model))
    {
        state.linearVelocity.xyz = jph_clamp_speed(state.linearVelocity.xyz, model.damping.z);
        state.angularVelocity.xyz = jph_clamp_speed(state.angularVelocity.xyz, model.damping.w);
        state.rotation = jph_rotation_step(state.rotation, state.angularVelocity.xyz * dt);
    }
    else
        state.rotation = jph_kinematic_rotation_step(state.rotation, state.angularVelocity.xyz * dt);
    state.positionCOM.xyz += dt * state.linearVelocity.xyz;
    body_states[index] = state;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_position(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.w) return;
    uint world_index = stage_work.range.z + tid.x;
    if (world_index >= step_params.counts.y)
        return;
    WorldRange world = worlds[world_index];
    // Bounded contact rows from one position iteration, in the original order.
    uint contact_count = world.bodies.w * MaxContactPointsPerPair;
    uint first_contact = world.bodies.z * MaxContactPointsPerPair;
    uint first = stage_work.range.x;
    if (first >= contact_count) return;
    uint end = first + min(stage_work.range.y, contact_count - first);
    for (uint local = first; local < end; ++local)
    {
        Contact contact = contacts[first_contact + local];
        if (contact.meta.z == 0u)
            continue;
        uint a = contact.meta.x;
        uint b = contact.meta.y;
        BodyModel ma = body_models[a];
        BodyModel mb = body_models[b];
        BodyState sa = body_states[a];
        BodyState sb = body_states[b];
        jph_solve_position(ma, sa, mb, sb, contact,
            step_params.contact.y, step_params.contact.z, step_params.contact.w);
        if (jph_dynamic(ma))
            body_states[a] = sa;
        if (jph_dynamic(mb))
            body_states[b] = sb;
    }
}
