// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021-2026 Jorrit Rouwe
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef JPH_GPU_RIGID_MATH_SLANG
#define JPH_GPU_RIGID_MATH_SLANG

#include "../PhysicsTypes.h"

// Pure math, with the same xyzw / world angular velocity conventions as
// Jolt/Shaders/ShaderQuat.h and Physics/Body/Body.inl.
float4 jph_quat_mul(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                  a.w * b.w - dot(a.xyz, b.xyz));
}

float3 jph_rotate(float4 q, float3 v)
{
    float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float3 jph_inverse_rotate(float4 q, float3 v)
{
    return jph_rotate(float4(-q.xyz, q.w), v);
}

float4 jph_rotation_step(float4 q, float3 angular_step)
{
    // Body::AddRotationStep uses an axis-angle left product, not quaternion
    // Euler integration, and skips steps with length <= 1e-6.
    float angle = length(angular_step);
    if (angle <= 1.0e-6f)
        return q;
    float half_angle = 0.5f * angle;
    float4 delta = float4(angular_step * (sin(half_angle) / angle), cos(half_angle));
    return normalize(jph_quat_mul(delta, q));
}

float3 jph_clamp_speed(float3 velocity, float maximum)
{
    float speed_sq = dot(velocity, velocity);
    if (speed_sq > maximum * maximum)
        velocity *= maximum / sqrt(speed_sq);
    return velocity;
}

bool jph_dynamic(BodyModel model)
{
    return (model.meta.z & ~MotionNotResident) == MotionDynamic;
}

bool jph_resident(BodyModel model)
{
    return (model.meta.z & MotionNotResident) == 0u;
}

float jph_inverse_mass(BodyModel model)
{
    return jph_dynamic(model) ? model.mass.x : 0.0f;
}

float3 jph_inverse_inertia(BodyModel model, BodyState state, float3 v)
{
    if (!jph_dynamic(model))
        return float3(0.0f);
    float4 inertia_to_world = jph_quat_mul(state.rotation, model.inertiaRotation);
    return jph_rotate(inertia_to_world,
        model.invInertiaDiagonal.xyz * jph_inverse_rotate(inertia_to_world, v));
}

float3 jph_point_velocity(BodyModel model, BodyState state, float3 r)
{
    // Like Jolt's motion-type specialization, static bodies have zero point
    // velocity even if an input snapshot contains stale velocity padding/data.
    return jph_dynamic(model)
        ? state.linearVelocity.xyz + cross(state.angularVelocity.xyz, r)
        : float3(0.0f);
}

void jph_apply_gyroscopic_force(BodyModel model, inout BodyState state, float dt)
{
    // MotionProperties::ApplyGyroscopicForceInternal. Preparation admits only
    // positive principal inverse inertia; locked axes are outside this slice.
    float4 inertia_to_world = jph_quat_mul(state.rotation, model.inertiaRotation);
    float3 local_velocity = jph_inverse_rotate(inertia_to_world, state.angularVelocity.xyz);
    float3 local_momentum = local_velocity / model.invInertiaDiagonal.xyz;
    float3 new_momentum = local_momentum - dt * cross(local_velocity, local_momentum);
    float new_length_sq = dot(new_momentum, new_momentum);
    new_momentum = new_length_sq > 0.0f
        ? new_momentum * sqrt(dot(local_momentum, local_momentum) / new_length_sq)
        : float3(0.0f);
    state.angularVelocity.xyz = jph_rotate(inertia_to_world, model.invInertiaDiagonal.xyz * new_momentum);
}

float3 jph_acceleration(BodyModel model, BodyForces forces, float3 gravity)
{
    return jph_dynamic(model) ? gravity * model.mass.w + forces.force.xyz * model.mass.x : float3(0.0f);
}

void jph_tangents(float3 normal, out float3 t1, out float3 t2)
{
    // Vec3::GetNormalizedPerpendicular and ContactConstraint::GetTangents.
    t1 = abs(normal.x) > abs(normal.y)
        ? normalize(float3(normal.z, 0.0f, -normal.x))
        : normalize(float3(0.0f, normal.z, -normal.y));
    t2 = cross(normal, t1);
}

float jph_effective_mass(BodyModel ma, BodyState sa, float3 ra,
                         BodyModel mb, BodyState sb, float3 rb, float3 axis)
{
    float3 ja = cross(ra, axis);
    float3 jb = cross(rb, axis);
    float inverse_effective_mass = jph_inverse_mass(ma) + jph_inverse_mass(mb)
        + dot(ja, jph_inverse_inertia(ma, sa, ja))
        + dot(jb, jph_inverse_inertia(mb, sb, jb));
    return inverse_effective_mass > 0.0f ? 1.0f / inverse_effective_mass : 0.0f;
}

float jph_jv(BodyModel ma, BodyState sa, float3 ra,
             BodyModel mb, BodyState sb, float3 rb, float3 axis)
{
    // ContactConstraintPart::SolveVelocityConstraintGetTotalLambda convention.
    return dot(axis, jph_point_velocity(ma, sa, ra) - jph_point_velocity(mb, sb, rb));
}

void jph_apply_impulse(BodyModel ma, inout BodyState sa, float3 ra,
                       BodyModel mb, inout BodyState sb, float3 rb,
                       float3 axis, float delta_lambda)
{
    if (jph_dynamic(ma))
    {
        sa.linearVelocity.xyz -= delta_lambda * ma.mass.x * axis;
        sa.angularVelocity.xyz -= delta_lambda * jph_inverse_inertia(ma, sa, cross(ra, axis));
    }
    if (jph_dynamic(mb))
    {
        sb.linearVelocity.xyz += delta_lambda * mb.mass.x * axis;
        sb.angularVelocity.xyz += delta_lambda * jph_inverse_inertia(mb, sb, cross(rb, axis));
    }
}

void jph_contact_points(Contact contact, BodyState sa, BodyState sb,
                        out float3 pa, out float3 pb, out float3 ra, out float3 rb)
{
    pa = sa.positionCOM.xyz + jph_rotate(sa.rotation, contact.localAnchorA.xyz);
    pb = sb.positionCOM.xyz + jph_rotate(sb.rotation, contact.localAnchorB.xyz);
    float3 midpoint = 0.5f * (pa + pb);
    ra = midpoint - sa.positionCOM.xyz;
    rb = midpoint - sb.positionCOM.xyz;
}

void jph_sphere_sphere(float3 ca, float radius_a, float3 cb, float radius_b,
                       out float3 normal, out float penetration,
                       out float3 pa, out float3 pb)
{
    float3 delta = cb - ca;
    float distance_sq = dot(delta, delta);
    float distance = sqrt(distance_sq);
    // A deterministic finite direction for coincident spheres.
    normal = distance_sq > 1.0e-20f ? delta / distance : float3(1.0f, 0.0f, 0.0f);
    penetration = radius_a + radius_b - distance;
    pa = ca + radius_a * normal;
    pb = cb - radius_b * normal;
}

void jph_sphere_box(float3 sphere_center, float sphere_radius,
                    float3 box_center, float4 box_rotation, float4 box_geometry,
                    out float3 normal_sphere_to_box, out float penetration,
                    out float3 point_sphere, out float3 point_box, out uint feature)
{
    // BoxShape::ExcludeConvexRadius exposes the reduced core plus a Minkowski
    // convex radius. Thus sphere/core-box distance uses the sum of the radii.
    // This retains the rounded edge/corner geometry instead of ignoring the
    // default box convex radius. Deep/degenerate contacts are handled explicitly.
    float convex_radius = box_geometry.w;
    float3 core_half_extent = max(box_geometry.xyz - convex_radius, float3(0.0f));
    float3 center = jph_inverse_rotate(box_rotation, sphere_center - box_center);
    float3 closest_core = clamp(center, -core_half_extent, core_half_extent);
    float3 delta = center - closest_core;
    float distance_sq = dot(delta, delta);
    float3 outward;
    float3 box_point_local;
    if (distance_sq > 1.0e-20f)
    {
        float distance = sqrt(distance_sq);
        outward = delta / distance;
        penetration = sphere_radius + convex_radius - distance;
        box_point_local = closest_core + convex_radius * outward;
        // Outside-core contacts include a face, edge or corner feature.
        feature = 8u + (center.x < -core_half_extent.x ? 1u : center.x > core_half_extent.x ? 2u : 0u)
            + 3u * (center.y < -core_half_extent.y ? 1u : center.y > core_half_extent.y ? 2u : 0u)
            + 9u * (center.z < -core_half_extent.z ? 1u : center.z > core_half_extent.z ? 2u : 0u);
    }
    else
    {
        // An embedded center must leave through its nearest core face. A
        // closest-point clamp alone gives zero and would miss this collision.
        float3 face_distance = core_half_extent - abs(center);
        uint axis = 0u;
        if (face_distance.y < face_distance.x)
            axis = 1u;
        if (face_distance.z < face_distance[axis])
            axis = 2u;
        outward = float3(0.0f);
        outward[axis] = center[axis] >= 0.0f ? 1.0f : -1.0f;
        float distance_to_face = max(face_distance[axis], 0.0f);
        penetration = sphere_radius + convex_radius + distance_to_face;
        box_point_local = center + (distance_to_face + convex_radius) * outward;
        feature = 2u * axis + (center[axis] >= 0.0f ? 1u : 0u);
    }
    float3 outward_world = jph_rotate(box_rotation, outward);
    normal_sphere_to_box = -outward_world;
    point_sphere = sphere_center - sphere_radius * outward_world;
    point_box = box_center + jph_rotate(box_rotation, box_point_local);
}

float jph_combine_material(float value_a, uint mode_a, float value_b, uint mode_b)
{
    // The immutable model boundary validates the four supported modes.
    uint mode = max(mode_a, mode_b);
    if (mode == MaterialCombineAverage)
        return 0.5f * (value_a + value_b);
    if (mode == MaterialCombineMinimum)
        return min(value_a, value_b);
    if (mode == MaterialCombineMultiply)
        return value_a * value_b;
    return max(value_a, value_b);
}

void jph_solve_velocity(BodyModel ma, inout BodyState sa, BodyModel mb, inout BodyState sb,
                        inout Contact contact)
{
    float3 normal = contact.normalDepth.xyz;
    float3 pa, pb, ra, rb;
    jph_contact_points(contact, sa, sb, pa, pb, ra, rb);
    float3 tangent1, tangent2;
    jph_tangents(normal, tangent1, tangent2);
    float friction = jph_combine_material(ma.mass.y, ma.surface.z, mb.mass.y, mb.surface.z);

    // Match Jolt's friction-first / normal-last ordering. The friction bound
    // uses the normal impulse from the PREVIOUS velocity iteration.
    float max_friction = friction * contact.impulse.x;
    float lambda1 = contact.impulse.y
        + jph_effective_mass(ma, sa, ra, mb, sb, rb, tangent1) * jph_jv(ma, sa, ra, mb, sb, rb, tangent1);
    float lambda2 = contact.impulse.z
        + jph_effective_mass(ma, sa, ra, mb, sb, rb, tangent2) * jph_jv(ma, sa, ra, mb, sb, rb, tangent2);
    float friction_sq = lambda1 * lambda1 + lambda2 * lambda2;
    if (friction_sq > max_friction * max_friction)
    {
        float scale = max_friction / sqrt(friction_sq);
        lambda1 *= scale;
        lambda2 *= scale;
    }
    jph_apply_impulse(ma, sa, ra, mb, sb, rb, tangent1, lambda1 - contact.impulse.y);
    jph_apply_impulse(ma, sa, ra, mb, sb, rb, tangent2, lambda2 - contact.impulse.z);
    contact.impulse.y = lambda1;
    contact.impulse.z = lambda2;

    float lambda_n = max(0.0f, contact.impulse.x
        + jph_effective_mass(ma, sa, ra, mb, sb, rb, normal)
            * (jph_jv(ma, sa, ra, mb, sb, rb, normal) - contact.impulse.w));
    jph_apply_impulse(ma, sa, ra, mb, sb, rb, normal, lambda_n - contact.impulse.x);
    contact.impulse.x = lambda_n;
}

void jph_solve_position(BodyModel ma, inout BodyState sa, BodyModel mb, inout BodyState sb,
                        Contact contact, float penetration_slop, float maximum_correction, float baumgarte)
{
    float3 normal = contact.normalDepth.xyz;
    float3 pa, pb, ra, rb;
    jph_contact_points(contact, sa, sb, pa, pb, ra, rb);
    float separation = max(dot(pb - pa, normal) + penetration_slop, -maximum_correction);
    if (separation >= 0.0f)
        return;
    float lambda = -jph_effective_mass(ma, sa, ra, mb, sb, rb, normal) * baumgarte * separation;
    if (jph_dynamic(ma))
    {
        float3 angular_step = -lambda * jph_inverse_inertia(ma, sa, cross(ra, normal));
        sa.positionCOM.xyz -= lambda * ma.mass.x * normal;
        sa.rotation = jph_rotation_step(sa.rotation, angular_step);
    }
    if (jph_dynamic(mb))
    {
        float3 angular_step = lambda * jph_inverse_inertia(mb, sb, cross(rb, normal));
        sb.positionCOM.xyz += lambda * mb.mass.x * normal;
        sb.rotation = jph_rotation_step(sb.rotation, angular_step);
    }
    // Split position correction deliberately leaves the physical velocities alone.
}

#endif // JPH_GPU_RIGID_MATH_SLANG
