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
void clear_forces(uint3 tid : SV_DispatchThreadID)
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
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_bounds(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= step_params.counts.x)
        return;
    BodyModel model = body_models[index];
    BodyState state = body_states[index];
    if (model.meta.y == ShapeEmpty)
    {
        Aabb empty_bounds;
        empty_bounds.minimum = float4(state.positionCOM.xyz, 0.0f);
        empty_bounds.maximum = empty_bounds.minimum;
        bounds[index] = empty_bounds;
        return;
    }
    ShapeModel shape = shape_models[model.meta.x];
    float4 geometry = shape.geometry;
    float3 center = jph_leaf_position(shape, state);
    float4 rotation = jph_leaf_rotation(shape, state);
    float3 extent;
    if (model.meta.y == ShapeSphere)
        extent = float3(geometry.w);
    else
        extent = abs(jph_rotate(rotation, float3(1.0f, 0.0f, 0.0f))) * geometry.x
               + abs(jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f))) * geometry.y
               + abs(jph_rotate(rotation, float3(0.0f, 0.0f, 1.0f))) * geometry.z;
    // Expand each body by half the contact band, so the pair band is exactly
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
    uint count = world.range.y;
    if (tid.x >= count || tid.y >= count)
        return;
    uint slot = world.range.z + tid.x * count + tid.y;
    uint a = world.range.x + tid.x;
    uint b = world.range.x + tid.y;
    Contact result;
    result.meta = uint4(a, b, 0u, 0u);
    result.normalDepth = float4(0.0f);
    result.localAnchorA = float4(0.0f);
    result.localAnchorB = float4(0.0f);
    result.impulse = float4(0.0f);
    contacts[slot] = result;
    if (tid.x >= tid.y)
        return;

    BodyModel ma = body_models[a];
    BodyModel mb = body_models[b];
    if (!jph_resident(ma) || !jph_resident(mb))
        return;
    if (ma.meta.y == ShapeEmpty || mb.meta.y == ShapeEmpty)
        return;
    if (!jph_dynamic(ma) && !jph_dynamic(mb))
        return;
    if ((ma.surface.y & (1u << mb.surface.x)) == 0u || (mb.surface.y & (1u << ma.surface.x)) == 0u)
        return;
    Aabb ba = bounds[a];
    Aabb bb = bounds[b];
    if (any(ba.minimum.xyz > bb.maximum.xyz) || any(bb.minimum.xyz > ba.maximum.xyz))
        return;

    BodyState sa = body_states[a];
    BodyState sb = body_states[b];
    ShapeModel shape_a = shape_models[ma.meta.x];
    ShapeModel shape_b = shape_models[mb.meta.x];
    float4 ga = shape_a.geometry;
    float4 gb = shape_b.geometry;
    float3 ca = jph_leaf_position(shape_a, sa);
    float3 cb = jph_leaf_position(shape_b, sb);
    float4 qa = jph_leaf_rotation(shape_a, sa);
    float4 qb = jph_leaf_rotation(shape_b, sb);
    float3 normal, pa, pb;
    float penetration;
    uint feature = 0u;
    if (ma.meta.y == ShapeSphere && mb.meta.y == ShapeSphere)
        jph_sphere_sphere(ca, ga.w, cb, gb.w, normal, penetration, pa, pb);
    else if (ma.meta.y == ShapeSphere && mb.meta.y == ShapeBox)
        jph_sphere_box(ca, ga.w, cb, qb, gb,
                       normal, penetration, pa, pb, feature);
    else
    {
        // After Empty is skipped, admitted geometry is Sphere or STATIC Box; the only
        // remaining active type combination is static Box / dynamic Sphere.
        // Unsupported shape/motion combinations must be rejected at preparation.
        jph_sphere_box(cb, gb.w, ca, qa, ga,
                       normal, penetration, pb, pa, feature);
        normal = -normal;
    }
    if (penetration < -step_params.contact.x)
        return;
    result.meta.z = 1u;
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
    uint a = contact.meta.x;
    uint b = contact.meta.y;
    BodyModel ma = body_models[a];
    BodyModel mb = body_models[b];
    BodyState sa = body_states[a];
    BodyState sb = body_states[b];
    float3 pa, pb, ra, rb;
    jph_contact_points(contact, sa, sb, pa, pb, ra, rb);
    float3 normal = contact.normalDepth.xyz;
    float normal_velocity = dot(jph_point_velocity(mb, sb, rb) - jph_point_velocity(ma, sa, ra), normal);
    float dt = step_params.gravityDt.w;
    float speculative_bias = max(0.0f, -contact.normalDepth.w / dt);
    float restitution = jph_combine_material(ma.mass.z, ma.surface.w, mb.mass.z, mb.surface.w);
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
    for (uint local = 0u; local < world.range.w; ++local)
    {
        uint slot = world.range.z + local;
        Contact contact = contacts[slot];
        if (contact.meta.z == 0u)
            continue;
        uint a = contact.meta.x;
        uint b = contact.meta.y;
        BodyModel ma = body_models[a];
        BodyModel mb = body_models[b];
        BodyState sa = body_states[a];
        BodyState sb = body_states[b];
        jph_solve_velocity(ma, sa, mb, sb, contact);
        if (jph_dynamic(ma))
            body_states[a] = sa;
        if (jph_dynamic(mb))
            body_states[b] = sb;
        contacts[slot] = contact;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void integrate(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= step_params.counts.x)
        return;
    BodyModel model = body_models[index];
    if (!jph_dynamic(model) || !jph_resident(model))
        return;
    BodyState state = body_states[index];
    float dt = step_params.gravityDt.w;
    // Jolt clamps again after impulses, then rotates and translates using v1.
    state.linearVelocity.xyz = jph_clamp_speed(state.linearVelocity.xyz, model.damping.z);
    state.angularVelocity.xyz = jph_clamp_speed(state.angularVelocity.xyz, model.damping.w);
    state.rotation = jph_rotation_step(state.rotation, state.angularVelocity.xyz * dt);
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
    for (uint local = 0u; local < world.range.w; ++local)
    {
        Contact contact = contacts[world.range.z + local];
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
