// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021-2026 Jorrit Rouwe
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"

ConstantBuffer<StepParams> step_params;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<WorldRange> worlds;
RWStructuredBuffer<BodyForces> body_forces;
RWStructuredBuffer<BodyState> body_states;
RWStructuredBuffer<Aabb> bounds;
RWStructuredBuffer<Contact> contacts;
RWStructuredBuffer<KinematicTarget> kinematic_targets;
RWStructuredBuffer<ContactPatch> contact_patches;

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
    uint index = tid.x;
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
    uint index = tid.x;
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
    uint index = tid.x;
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
    uint index = tid.x;
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
[numthreads(8, 8, 1)]
void collide_pairs(uint3 tid : SV_DispatchThreadID)
{
    if (tid.z >= step_params.counts.y)
        return;
    WorldRange world = worlds[tid.z];
    uint count = world.leaves.y;
    if (tid.x >= count || tid.y >= count)
        return;
    uint patch_index = world.bodies.z + tid.x * count + tid.y;
    uint slot = patch_index * MaxContactPointsPerPair;
    uint leaf_a = world.leaves.x + tid.x;
    uint leaf_b = world.leaves.x + tid.y;
    ShapeModel shape_a = shape_models[leaf_a];
    ShapeModel shape_b = shape_models[leaf_b];
    uint a = shape_a.meta.y;
    uint b = shape_b.meta.y;
    ContactPatch patch;
    patch.leaves = uint4(leaf_a, leaf_b, 0u, 0u);
    contact_patches[patch_index] = patch;
    Contact result;
    result.meta = uint4(a, b, 0u, 0u);
    result.normalDepth = float4(0.0f);
    result.localAnchorA = float4(0.0f);
    result.localAnchorB = float4(0.0f);
    result.impulse = float4(0.0f);
    // Every ordered leaf pair owns a header and four point records, including
    // inactive/lower-triangle/same-body pairs. Both buffers are complete writers
    // without atomics or compaction; no patch combines unrelated leaf normals.
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        contacts[slot + point] = result;
    if (tid.x >= tid.y || a == b)
        return;

    BodyModel ma = body_models[a];
    BodyModel mb = body_models[b];
    if (!jph_resident(ma) || !jph_resident(mb))
        return;
    bool sensor = jph_sensor(shape_a) || jph_sensor(shape_b);
    if (!sensor && !jph_dynamic(ma) && !jph_dynamic(mb))
        return;
    if (!jph_layers_collide(shape_a, shape_b))
        return;
    patch.leaves.z = sensor ? PatchSensor : 0u;
    contact_patches[patch_index] = patch;
    Aabb ba = bounds[leaf_a];
    Aabb bb = bounds[leaf_b];
    if (any(ba.minimum.xyz > bb.maximum.xyz) || any(bb.minimum.xyz > ba.maximum.xyz))
        return;

    BodyState sa = body_states[a];
    BodyState sb = body_states[b];
    float4 ga = shape_a.geometry;
    float4 gb = shape_b.geometry;
    float3 ca = jph_leaf_position(shape_a, sa);
    float3 cb = jph_leaf_position(shape_b, sb);
    float4 qa = jph_leaf_rotation(shape_a, sa);
    float4 qb = jph_leaf_rotation(shape_b, sb);
    float speculative_distance = sensor ? 0.0f : step_params.contact.x;
    if (shape_a.meta.x == ShapeBox && shape_b.meta.x == ShapeBox)
    {
        JphBoxManifold manifold = jph_box_box(ca, qa, ga, cb, qb, gb, speculative_distance);
        if (manifold.count == 0u)
            return;
        patch.leaves.z |= (sensor ? 0u : PatchActive) |
                          (manifold.penetration >= 0.0f ? PatchOverlapping : 0u);
        contact_patches[patch_index] = patch;
        for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
            if (point < manifold.count)
            {
                JphBoxPoint contact = manifold.points[point];
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
    else
    {
        jph_sphere_box(cb, gb.w, ca, qa, ga, normal, penetration, pb, pa, feature);
        normal = -normal;
    }
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
    uint slot = tid.x;
    if (slot >= step_params.solve.z)
        return;
    Contact contact = contacts[slot];
    if (contact.meta.z == 0u)
        return;
    ContactPatch patch = contact_patches[slot / MaxContactPointsPerPair];
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
    uint world_index = tid.x;
    if (world_index >= step_params.counts.y)
        return;
    WorldRange world = worlds[world_index];
    // One lane owns a whole small world, so contacts sharing a dynamic body
    // never race. This dispatch is ONE PGS iteration; the engine repeats it.
    // Preparation proves body count <= MaxBodiesPerWorld (currently 32).
    for (uint local = 0u; local < world.bodies.w; ++local)
    {
        uint patch_index = world.bodies.z + local;
        ContactPatch patch = contact_patches[patch_index];
        if ((patch.leaves.z & PatchActive) == 0u)
            continue;
        uint slot = patch_index * MaxContactPointsPerPair;
        // collide_pairs packs active points from the first record of a pair.
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
    uint index = tid.x;
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
    uint world_index = tid.x;
    if (world_index >= step_params.counts.y)
        return;
    WorldRange world = worlds[world_index];
    // One bounded position iteration, at the same ownership boundary as PGS.
    uint contact_count = world.bodies.w * MaxContactPointsPerPair;
    uint first_contact = world.bodies.z * MaxContactPointsPerPair;
    for (uint local = 0u; local < contact_count; ++local)
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
