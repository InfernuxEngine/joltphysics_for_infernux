// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021-2026 Jorrit Rouwe
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef JPH_GPU_RIGID_JOINTS_SLANG
#define JPH_GPU_RIGID_JOINTS_SLANG

#include "rigid_math.slang"

RWStructuredBuffer<QueryRecordWord> joint_stream;

JointAuthor jph_joint_load_author(uint index)
{
    uint p = 1u + JointRecordWords * index;
    JointAuthor a;
    a.bodies = joint_stream[p].value;
    a.identity = joint_stream[p + 1u].value;
    a.ownerAnchor = asfloat(joint_stream[p + 2u].value);
    a.ownerAxis = asfloat(joint_stream[p + 3u].value);
    a.ownerLocalAnchor = asfloat(joint_stream[p + 4u].value);
    a.rootCOM1 = asfloat(joint_stream[p + 5u].value);
    a.rootCOM2 = asfloat(joint_stream[p + 6u].value);
    a.scale1 = asfloat(joint_stream[p + 7u].value);
    a.scale2 = asfloat(joint_stream[p + 8u].value);
    a.limits = asfloat(joint_stream[p + 9u].value);
    a.motor = asfloat(joint_stream[p + 10u].value);
    a.passive = asfloat(joint_stream[p + 11u].value);
    return a;
}

JointState jph_joint_load_state(uint index)
{
    uint p = joint_stream[0].value.y + JointRecordWords * index;
    JointState s;
    s.flags = joint_stream[p].value;
    s.frame1 = asfloat(joint_stream[p + 1u].value);
    s.frame2 = asfloat(joint_stream[p + 2u].value);
    s.anchor1 = asfloat(joint_stream[p + 3u].value);
    s.anchor2 = asfloat(joint_stream[p + 4u].value);
    s.collapsed1 = asfloat(joint_stream[p + 5u].value);
    s.collapsed2 = asfloat(joint_stream[p + 6u].value);
    s.pointImpulse = asfloat(joint_stream[p + 7u].value);
    s.angularImpulse = asfloat(joint_stream[p + 8u].value);
    s.axisImpulse = asfloat(joint_stream[p + 9u].value);
    s.rotor = asfloat(joint_stream[p + 10u].value);
    s.setup = asfloat(joint_stream[p + 11u].value);
    return s;
}

void jph_joint_store_state(uint index, JointState s)
{
    // The only stores into this resource are in the runtime suffix.
    uint p = joint_stream[0].value.y + JointRecordWords * index;
    joint_stream[p].value = s.flags;
    joint_stream[p + 1u].value = asuint(s.frame1);
    joint_stream[p + 2u].value = asuint(s.frame2);
    joint_stream[p + 3u].value = asuint(s.anchor1);
    joint_stream[p + 4u].value = asuint(s.anchor2);
    joint_stream[p + 5u].value = asuint(s.collapsed1);
    joint_stream[p + 6u].value = asuint(s.collapsed2);
    joint_stream[p + 7u].value = asuint(s.pointImpulse);
    joint_stream[p + 8u].value = asuint(s.angularImpulse);
    joint_stream[p + 9u].value = asuint(s.axisImpulse);
    joint_stream[p + 10u].value = asuint(s.rotor);
    joint_stream[p + 11u].value = asuint(s.setup);
}

float4 jph_joint_conjugate(float4 q) { return float4(-q.xyz, q.w); }

BodyState jph_joint_world_state()
{
    BodyState s;
    s.positionCOM = float4(0.0f);
    s.rotation = float4(0.0f, 0.0f, 0.0f, 1.0f);
    s.linearVelocity = float4(0.0f);
    s.angularVelocity = float4(0.0f);
    return s;
}

BodyModel jph_joint_world_model()
{
    BodyModel m;
    m.meta = uint4(0u);
    m.mass = float4(0.0f);
    m.invInertiaDiagonal = float4(0.0f);
    m.inertiaRotation = float4(0.0f, 0.0f, 0.0f, 1.0f);
    m.damping = float4(0.0f);
    return m;
}

float3 jph_joint_perpendicular(float3 v)
{
    // Vec3::GetNormalizedPerpendicular, including the same deterministic basis.
    return normalize(abs(v.x) > abs(v.y) ? float3(v.z, 0.0f, -v.x) : float3(0.0f, v.z, -v.y));
}

float4 jph_joint_frame(float3 x, float3 y, float3 z)
{
    // Quaternion from rotation columns, avoiding a poorly conditioned trace
    // branch near PI. Frames supplied here are orthonormal.
    float4 q;
    float trace = x.x + y.y + z.z;
    if (trace > 0.0f)
    {
        float t = 2.0f * sqrt(1.0f + trace);
        q = float4((y.z - z.y) / t, (z.x - x.z) / t, (x.y - y.x) / t, 0.25f * t);
    }
    else if (x.x > y.y && x.x > z.z)
    {
        float t = 2.0f * sqrt(1.0f + x.x - y.y - z.z);
        q = float4(0.25f * t, (y.x + x.y) / t, (z.x + x.z) / t, (y.z - z.y) / t);
    }
    else if (y.y > z.z)
    {
        float t = 2.0f * sqrt(1.0f + y.y - x.x - z.z);
        q = float4((y.x + x.y) / t, 0.25f * t, (z.y + y.z) / t, (z.x - x.z) / t);
    }
    else
    {
        float t = 2.0f * sqrt(1.0f + z.z - x.x - y.y);
        q = float4((z.x + x.z) / t, (z.y + y.z) / t, 0.25f * t, (x.y - y.x) / t);
    }
    return normalize(q);
}

void jph_joint_unscale(float3 offset, float3 scale, float3 raw,
                       out float4 anchor, out float4 collapsed)
{
    anchor = float4(raw, 0.0f);
    collapsed = float4(0.0f);
    for (uint axis = 0u; axis < 3u; ++axis)
        if (scale[axis] != 0.0f)
            anchor[axis] = offset[axis] / scale[axis];
        else
            collapsed[axis] = offset[axis];
}

JointState jph_joint_initialize(JointAuthor a, BodyState s1, BodyState s2)
{
    JointState s;
    s.flags = uint4(1u, JointStatusSuccess, asuint(a.passive.z), 0u);
    float3 world_anchor = s2.positionCOM.xyz + jph_rotate(s2.rotation, a.ownerAnchor.xyz - a.rootCOM2.xyz);
    if (a.bodies.x == InvalidQueryBodySlot)
    {
        s.anchor1 = float4(world_anchor, 0.0f);
        s.collapsed1 = float4(0.0f);
    }
    else
    {
        float3 offset = jph_inverse_rotate(s1.rotation, world_anchor - s1.positionCOM.xyz) + a.rootCOM1.xyz;
        jph_joint_unscale(offset, a.scale1.xyz, float3(0.0f), s.anchor1, s.collapsed1);
    }
    jph_joint_unscale(a.ownerAnchor.xyz, a.scale2.xyz, a.ownerLocalAnchor.xyz, s.anchor2, s.collapsed2);
    float4 world_frame = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (a.bodies.z == JointHinge)
    {
        float3 z = normalize(jph_rotate(s2.rotation, a.ownerAxis.xyz));
        float3 x = jph_joint_perpendicular(z);
        world_frame = jph_joint_frame(x, cross(z, x), z);
    }
    if (a.bodies.z == JointSlider)
    {
        float3 x = normalize(jph_rotate(s2.rotation, a.ownerAxis.xyz));
        float3 y = jph_joint_perpendicular(x);
        world_frame = jph_joint_frame(x, y, cross(x, y));
    }
    s.frame1 = normalize(jph_quat_mul(jph_joint_conjugate(s1.rotation), world_frame));
    s.frame2 = normalize(jph_quat_mul(jph_joint_conjugate(s2.rotation), world_frame));
    s.pointImpulse = float4(0.0f);
    s.angularImpulse = float4(0.0f);
    s.axisImpulse = float4(0.0f);
    s.rotor = float4(0.0f);
    s.setup = float4(0.0f);
    if (a.bodies.z == JointHinge)
    {
        float3 axis = jph_rotate(jph_quat_mul(s1.rotation, s.frame1), float3(0.0f, 0.0f, 1.0f));
        s.rotor.x = dot(axis, s2.angularVelocity.xyz - s1.angularVelocity.xyz);
    }
    if (!all(isfinite(s.anchor1)) || !all(isfinite(s.anchor2)) ||
        !all(isfinite(s.frame1)) || !all(isfinite(s.frame2)) ||
        !all(isfinite(s.collapsed1)) || !all(isfinite(s.collapsed2)) || !all(isfinite(s.rotor)))
        s.flags.y = JointStatusNumericalFailure;
    return s;
}

float3 jph_joint_axis(BodyState s1, JointState s)
{
    return jph_rotate(jph_quat_mul(s1.rotation, s.frame1), float3(0.0f, 0.0f, 1.0f));
}

float4 jph_joint_rotation_error(BodyState s1, BodyState s2, JointState s)
{
    float4 inverse_initial = jph_quat_mul(s.frame2, jph_joint_conjugate(s.frame1));
    return jph_quat_mul(jph_quat_mul(s2.rotation, inverse_initial), jph_joint_conjugate(s1.rotation));
}

float jph_joint_angle(BodyState s1, BodyState s2, JointState s)
{
    float4 diff = jph_joint_rotation_error(s1, s2, s);
    // Quat::GetRotationAngle: signed twist, invariant under q -> -q.
    return diff.w == 0.0f ? 3.14159265358979323846f : 2.0f * atan(dot(diff.xyz, jph_joint_axis(s1, s)) / diff.w);
}

float jph_joint_center_angle(float angle)
{
    return angle - 6.28318530717958647692f * floor((angle + 3.14159265358979323846f) / 6.28318530717958647692f);
}

float3 jph_joint_angular_velocity(BodyModel m, BodyState s)
{
    return jph_nonstatic(m) ? s.angularVelocity.xyz : float3(0.0f);
}

void jph_joint_arms(JointAuthor a, JointState s, BodyState s1, BodyState s2, out float3 r1, out float3 r2)
{
    r1 = a.bodies.x == InvalidQueryBodySlot ? s.anchor1.xyz :
        jph_rotate(s1.rotation, s.anchor1.xyz * a.scale1.xyz + s.collapsed1.xyz - a.rootCOM1.xyz);
    r2 = jph_rotate(s2.rotation, s.anchor2.xyz * a.scale2.xyz + s.collapsed2.xyz - a.rootCOM2.xyz);
}

float3 jph_joint_inertia_sum(BodyModel m1, BodyState s1, BodyModel m2, BodyState s2, float3 value)
{
    return jph_inverse_inertia(m1, s1, value) + jph_inverse_inertia(m2, s2, value);
}

float3 jph_joint_point_response(BodyModel m1, BodyState s1, float3 r1,
                              BodyModel m2, BodyState s2, float3 r2, float3 value)
{
    return (jph_inverse_mass(m1) + jph_inverse_mass(m2)) * value
        - cross(r1, jph_inverse_inertia(m1, s1, cross(r1, value)))
        - cross(r2, jph_inverse_inertia(m2, s2, cross(r2, value)));
}

float3 jph_joint_solve3(float3 c0, float3 c1, float3 c2, float3 rhs, inout JointState state)
{
    // Symmetric diagonal equilibration keeps unrelated units / large inertia
    // ratios from deciding the determinant threshold. This is a local block.
    float3 diagonal = float3(c0.x, c1.y, c2.z);
    if (any(diagonal <= 0.0f) || !all(isfinite(diagonal)))
    {
        state.flags.y = JointStatusNumericalFailure;
        return float3(0.0f);
    }
    float3 d = rsqrt(diagonal);
    c0 *= d * d.x;
    c1 *= d * d.y;
    c2 *= d * d.z;
    float3 row0 = cross(c1, c2);
    float3 row1 = cross(c2, c0);
    float3 row2 = cross(c0, c1);
    float det = dot(c0, row0);
    float3 scaled_rhs = d * rhs;
    float3 result = d * float3(dot(row0, scaled_rhs), dot(row1, scaled_rhs), dot(row2, scaled_rhs)) / det;
    if (!(det > 0.0f) || !all(isfinite(result)))
    {
        state.flags.y = JointStatusNumericalFailure;
        return float3(0.0f);
    }
    return result;
}

float3 jph_joint_solve_rotation(BodyModel m1, BodyState s1, BodyModel m2, BodyState s2,
                              float3 rhs, inout JointState state)
{
    return jph_joint_solve3(jph_joint_inertia_sum(m1, s1, m2, s2, float3(1.0f, 0.0f, 0.0f)),
        jph_joint_inertia_sum(m1, s1, m2, s2, float3(0.0f, 1.0f, 0.0f)),
        jph_joint_inertia_sum(m1, s1, m2, s2, float3(0.0f, 0.0f, 1.0f)), rhs, state);
}

float3 jph_joint_solve_point(BodyModel m1, BodyState s1, float3 r1, BodyModel m2, BodyState s2, float3 r2,
                           float3 rhs, inout JointState state)
{
    return jph_joint_solve3(jph_joint_point_response(m1, s1, r1, m2, s2, r2, float3(1.0f, 0.0f, 0.0f)),
        jph_joint_point_response(m1, s1, r1, m2, s2, r2, float3(0.0f, 1.0f, 0.0f)),
        jph_joint_point_response(m1, s1, r1, m2, s2, r2, float3(0.0f, 0.0f, 1.0f)), rhs, state);
}

void jph_joint_apply_angular(BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
                             float3 impulse, bool position)
{
    float3 a = jph_inverse_inertia(m1, s1, impulse);
    float3 b = jph_inverse_inertia(m2, s2, impulse);
    if (jph_dynamic(m1))
        if (position) s1.rotation = jph_rotation_step(s1.rotation, -a);
        else s1.angularVelocity.xyz -= a;
    if (jph_dynamic(m2))
        if (position) s2.rotation = jph_rotation_step(s2.rotation, b);
        else s2.angularVelocity.xyz += b;
}

void jph_joint_apply_point(BodyModel m1, inout BodyState s1, float3 r1,
                          BodyModel m2, inout BodyState s2, float3 r2, float3 impulse, bool position)
{
    float3 a = jph_inverse_inertia(m1, s1, cross(r1, impulse));
    float3 b = jph_inverse_inertia(m2, s2, cross(r2, impulse));
    if (jph_dynamic(m1))
        if (position)
        {
            s1.positionCOM.xyz -= m1.mass.x * impulse;
            s1.rotation = jph_rotation_step(s1.rotation, -a);
        }
        else
        {
            s1.linearVelocity.xyz -= m1.mass.x * impulse;
            s1.angularVelocity.xyz -= a;
        }
    if (jph_dynamic(m2))
        if (position)
        {
            s2.positionCOM.xyz += m2.mass.x * impulse;
            s2.rotation = jph_rotation_step(s2.rotation, b);
        }
        else
        {
            s2.linearVelocity.xyz += m2.mass.x * impulse;
            s2.angularVelocity.xyz += b;
        }
}

void jph_joint_swing_basis(BodyState s1, BodyState s2, JointState state,
                          out float3 j0, out float3 j1, out float2 error)
{
    float3 a1 = jph_joint_axis(s1, state);
    float3 a2 = jph_rotate(jph_quat_mul(s2.rotation, state.frame2), float3(0.0f, 0.0f, 1.0f));
    float alignment = dot(a1, a2);
    if (alignment <= 1.0e-3f)
    {
        float3 perpendicular = a2 - alignment * a1;
        if (dot(perpendicular, perpendicular) < 1.0e-6f)
            perpendicular = jph_joint_perpendicular(a1);
        a2 = normalize(0.99f * normalize(perpendicular) + 0.01f * a1);
    }
    float3 b2 = jph_joint_perpendicular(a2);
    float3 c2 = cross(a2, b2);
    j0 = cross(b2, a1);
    j1 = cross(c2, a1);
    error = float2(dot(a1, b2), dot(a1, c2));
}

float2 jph_joint_symmetric2(float a, float b, float c, float2 rhs, inout JointState state)
{
    float2 d = rsqrt(float2(a, c));
    float off = b * d.x * d.y;
    float det = 1.0f - off * off;
    float2 scaled_rhs = rhs * d;
    float2 result = d * float2(scaled_rhs.x - off * scaled_rhs.y, scaled_rhs.y - off * scaled_rhs.x) / det;
    if (!(a > 0.0f && c > 0.0f && det > 0.0f) || !all(isfinite(result)))
    {
        state.flags.y = JointStatusNumericalFailure;
        return float2(0.0f);
    }
    return result;
}

float2 jph_joint_solve2(BodyModel m1, BodyState s1, BodyModel m2, BodyState s2,
                       float3 j0, float3 j1, float2 rhs, inout JointState state)
{
    float3 i0 = jph_joint_inertia_sum(m1, s1, m2, s2, j0);
    float3 i1 = jph_joint_inertia_sum(m1, s1, m2, s2, j1);
    return jph_joint_symmetric2(dot(j0, i0), dot(j0, i1), dot(j1, i1), rhs, state);
}

// Slider rows use the same body writer and phase schedule as Fixed/Hinge.
// Mathematical source: SliderConstraint / DualAxisConstraintPart / AxisConstraintPart.
void jph_slider_basis(BodyState s1, JointState state, out float3 axis, out float3 n0, out float3 n1)
{
    float4 frame = jph_quat_mul(s1.rotation, state.frame1);
    axis = jph_rotate(frame, float3(1.0f, 0.0f, 0.0f));
    n0 = jph_rotate(frame, float3(0.0f, 1.0f, 0.0f));
    n1 = jph_rotate(frame, float3(0.0f, 0.0f, 1.0f));
}

void jph_slider_limit(JointAuthor a, float position, out float side, out float error)
{
    side = 0.0f;
    error = 0.0f;
    if ((a.bodies.w & JointLimitsEnabled) == 0u) return;
    if (position <= a.limits.x)
    {
        side = -1.0f;
        error = position - a.limits.x;
    }
    else if (position >= a.limits.y)
    {
        side = 1.0f;
        error = position - a.limits.y;
    }
}

float2 jph_slider_transverse(BodyModel m1, BodyState s1, float3 arm1,
    BodyModel m2, BodyState s2, float3 r2, float3 n0, float3 n1, float2 rhs, inout JointState state)
{
    float3 k0 = jph_joint_point_response(m1, s1, arm1, m2, s2, r2, n0);
    float3 k1 = jph_joint_point_response(m1, s1, arm1, m2, s2, r2, n1);
    return jph_joint_symmetric2(dot(n0, k0), dot(n0, k1), dot(n1, k1), rhs, state);
}

void jph_slider_solve_velocity(JointAuthor a, inout JointState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    float3 arm1 = r1 + u; // The rotating reference axis contributes u to body 1's Jacobian.
    float3 axis, n0, n1;
    jph_slider_basis(s1, state, axis, n0, n1);
    float3 relative = jph_point_velocity(m1, s1, arm1) - jph_point_velocity(m2, s2, r2);
    float2 transverse = jph_slider_transverse(m1, s1, arm1, m2, s2, r2, n0, n1,
        float2(dot(n0, relative), dot(n1, relative)), state);
    if (state.flags.y != JointStatusSuccess) return;
    state.pointImpulse.xy += transverse;
    jph_joint_apply_point(m1, s1, arm1, m2, s2, r2, n0 * transverse.x + n1 * transverse.y, false);
    float3 angular = jph_joint_solve_rotation(m1, s1, m2, s2,
        jph_joint_angular_velocity(m1, s1) - jph_joint_angular_velocity(m2, s2), state);
    if (state.flags.y != JointStatusSuccess) return;
    state.angularImpulse.xyz += angular;
    jph_joint_apply_angular(m1, s1, m2, s2, angular, false);
    if (state.setup.w != 0.0f)
    {
        float k = dot(axis, jph_joint_point_response(m1, s1, arm1, m2, s2, r2, axis));
        if (!(k > 0.0f) || !isfinite(k))
        {
            state.flags.y = JointStatusNumericalFailure;
            return;
        }
        float old = state.axisImpulse.x;
        relative = jph_point_velocity(m1, s1, arm1) - jph_point_velocity(m2, s2, r2);
        float next = old + dot(axis, relative) / k;
        if (a.limits.x != a.limits.y)
            next = state.setup.w < 0.0f ? max(0.0f, next) : min(0.0f, next);
        state.axisImpulse.x = next;
        jph_joint_apply_point(m1, s1, arm1, m2, s2, r2, axis * (next - old), false);
    }
}

void jph_slider_solve_position(JointAuthor a, inout JointState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte)
{
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    float3 axis, n0, n1;
    jph_slider_basis(s1, state, axis, n0, n1);
    float2 transverse = jph_slider_transverse(m1, s1, r1 + u, m2, s2, r2, n0, n1,
        -baumgarte * float2(dot(n0, u), dot(n1, u)), state);
    if (state.flags.y != JointStatusSuccess) return;
    jph_joint_apply_point(m1, s1, r1 + u, m2, s2, r2, n0 * transverse.x + n1 * transverse.y, true);
    float4 diff = jph_joint_rotation_error(s1, s2, state);
    float3 error = 2.0f * (diff.w < 0.0f ? -diff.xyz : diff.xyz);
    float3 angular = jph_joint_solve_rotation(m1, s1, m2, s2, -baumgarte * error, state);
    if (state.flags.y != JointStatusSuccess) return;
    jph_joint_apply_angular(m1, s1, m2, s2, angular, true);
    // Rotation correction changes both material points and the reference basis.
    jph_joint_arms(a, state, s1, s2, r1, r2);
    u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    jph_slider_basis(s1, state, axis, n0, n1);
    float side, limit_error;
    jph_slider_limit(a, dot(axis, u), side, limit_error);
    if (side != 0.0f && limit_error != 0.0f)
    {
        float k = dot(axis, jph_joint_point_response(m1, s1, r1 + u, m2, s2, r2, axis));
        if (!(k > 0.0f) || !isfinite(k))
        {
            state.flags.y = JointStatusNumericalFailure;
            return;
        }
        jph_joint_apply_point(m1, s1, r1 + u, m2, s2, r2, axis * (-baumgarte * limit_error / k), true);
    }
}

void jph_joint_limit(JointAuthor a, float angle, out float side, out float error)
{
    side = 0.0f;
    error = 0.0f;
    bool has_limits = (a.bodies.w & JointLimitsEnabled) != 0u
        && (a.limits.x > -3.14159265358979323846f || a.limits.y < 3.14159265358979323846f);
    if (has_limits && (angle <= a.limits.x || angle >= a.limits.y))
    {
        float lower = jph_joint_center_angle(angle - a.limits.x);
        float upper = jph_joint_center_angle(angle - a.limits.y);
        bool use_lower = abs(lower) < abs(upper);
        side = use_lower ? -1.0f : 1.0f;
        error = use_lower ? lower : upper;
    }
}

bool jph_joint_drive_active(JointAuthor a)
{
    return ((a.bodies.w & JointMotorEnabled) != 0u && (a.motor.y > 0.0f || a.motor.z > 0.0f))
        || any(a.passive.xyz > 0.0f);
}

void jph_joint_clear_impulses(inout JointState state)
{
    state.pointImpulse = float4(0.0f);
    state.angularImpulse = float4(0.0f);
    state.axisImpulse = float4(0.0f);
    state.rotor.y = 0.0f;
}

void jph_joint_setup(JointAuthor a, inout JointState state,
                     BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
                     float dt, bool warm_start)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess)
        return;
    float ratio = state.rotor.w > 0.0f ? dt / state.rotor.w : 0.0f;
    if ((a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
    {
        jph_joint_clear_impulses(state);
        return;
    }
    if (a.bodies.z == JointHinge)
    {
        float3 axis = jph_joint_axis(s1, state);
        state.rotor.z = a.passive.z > 0.0f ? state.rotor.x - state.rotor.y / a.passive.z : 0.0f;
        state.rotor.x = state.rotor.z;
        state.setup.x = jph_joint_angle(s1, s2, state);
        float limit_error;
        jph_joint_limit(a, state.setup.x, state.setup.w, limit_error);
        if (state.setup.w == 0.0f)
            state.axisImpulse.x = 0.0f;
        state.setup.yz = float2(0.0f);
        if ((a.bodies.w & JointMotorEnabled) != 0u && (a.motor.y > 0.0f || a.motor.z > 0.0f))
        {
            float target = (a.bodies.w & JointLimitsEnabled) != 0u ? clamp(a.motor.x, a.limits.x, a.limits.y) : a.motor.x;
            state.setup.y = 1.0f / (dt * (a.motor.z + dt * a.motor.y));
            state.setup.z = dt * a.motor.y * state.setup.y * jph_joint_center_angle(state.setup.x - target);
        }
        if (!jph_joint_drive_active(a))
        {
            state.axisImpulse.yzw = float3(0.0f);
            state.rotor.xy = float2(0.0f);
        }
    }
    if (a.bodies.z == JointSlider)
    {
        float3 r1, r2;
        jph_joint_arms(a, state, s1, s2, r1, r2);
        float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
        float3 axis, n0, n1;
        jph_slider_basis(s1, state, axis, n0, n1);
        state.setup = float4(dot(axis, u), 0.0f, 0.0f, 0.0f);
        float error;
        jph_slider_limit(a, state.setup.x, state.setup.w, error);
        if (state.setup.w == 0.0f) state.axisImpulse.x = 0.0f;
        state.pointImpulse.zw = float2(0.0f);
        state.axisImpulse.yzw = float3(0.0f);
        state.rotor.xyz = float3(0.0f);
    }
    if (!warm_start)
    {
        jph_joint_clear_impulses(state);
        return;
    }
    state.pointImpulse *= ratio;
    state.angularImpulse *= ratio;
    state.axisImpulse *= ratio;
    state.rotor.y *= ratio;
}

void jph_joint_warm_start(JointAuthor a, JointState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    if (a.bodies.z == JointSlider)
    {
        float3 axis, n0, n1;
        jph_slider_basis(s1, state, axis, n0, n1);
        float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
        jph_joint_apply_point(m1, s1, r1 + u, m2, s2, r2,
            n0 * state.pointImpulse.x + n1 * state.pointImpulse.y + axis * state.axisImpulse.x, false);
        jph_joint_apply_angular(m1, s1, m2, s2, state.angularImpulse.xyz, false);
        return;
    }
    if (a.bodies.z == JointFixed)
    {
        jph_joint_apply_angular(m1, s1, m2, s2, state.angularImpulse.xyz, false);
        jph_joint_apply_point(m1, s1, r1, m2, s2, r2, state.pointImpulse.xyz, false);
    }
    else
    {
        float3 j0, j1;
        float2 error;
        jph_joint_swing_basis(s1, s2, state, j0, j1, error);
        float3 axis = jph_joint_axis(s1, state);
        jph_joint_apply_angular(m1, s1, m2, s2, axis * state.axisImpulse.w, false);
        jph_joint_apply_point(m1, s1, r1, m2, s2, r2, state.pointImpulse.xyz, false);
        jph_joint_apply_angular(m1, s1, m2, s2, j0 * state.angularImpulse.x + j1 * state.angularImpulse.y, false);
        jph_joint_apply_angular(m1, s1, m2, s2, axis * state.axisImpulse.x, false);
    }
}

struct JphJointDriveImpulse { float motor, passive, total, rotor, velocity; };

void jph_joint_apply_drive_velocity(BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2,
    float3 axis, float3 response1, float3 response2, float k1, float k2, float velocity, inout JointState state)
{
    float k = k1 + k2;
    if (!(k > 0.0f) || k1 < 0.0f || k2 < 0.0f || !isfinite(k)
        || !isfinite(velocity) || !all(isfinite(response1)) || !all(isfinite(response2)))
    {
        state.flags.y = JointStatusNumericalFailure;
        return;
    }
    float3 w1 = jph_joint_angular_velocity(m1, s1);
    float3 w2 = jph_joint_angular_velocity(m2, s2);
    float u = dot(axis, w1), v = dot(axis, w2);
    float weight1 = k1 / k, weight2 = k2 / k;
    // Equal and opposite axial impulses leave this common projection intact.
    // Reconstruct along inverse-inertia response, including its off-axis terms.
    float common = fma(weight2, u, weight1 * v);
    float next_u = fma(-weight1, velocity, common);
    float next_v = fma(weight2, velocity, common);
    float3 next1 = s1.angularVelocity.xyz, next2 = s2.angularVelocity.xyz;
    if (jph_dynamic(m1) && k1 > 0.0f)
    {
        float3 direction = response1 / k1;
        float3 residual = fma(-direction, float3(u), w1);
        next1 = fma(direction, float3(next_u), residual);
    }
    if (jph_dynamic(m2) && k2 > 0.0f)
    {
        float3 direction = response2 / k2;
        float3 residual = fma(-direction, float3(v), w2);
        next2 = fma(direction, float3(next_v), residual);
    }
    if (!isfinite(common) || !isfinite(next_u) || !isfinite(next_v)
        || !all(isfinite(next1)) || !all(isfinite(next2)))
    {
        state.flags.y = JointStatusNumericalFailure;
        return;
    }
    if (jph_dynamic(m1) && k1 > 0.0f) s1.angularVelocity.xyz = next1;
    if (jph_dynamic(m2) && k2 > 0.0f) s2.angularVelocity.xyz = next2;
}

JphJointDriveImpulse jph_joint_drive(float free_velocity, float k, float gamma, float bias,
    float min_motor, float max_motor, float damping, float friction, float armature, float old_speed,
    inout JointState state)
{
    // ConstraintPart/DriveImpulse.h. Keep the stable net-impulse expression:
    // independent co-linear motor/passive rows undo one another on light links.
    // Evaluate net velocity independently of the cancelling free + K * lambda.
    JphJointDriveImpulse result;
    result.motor = result.passive = result.total = result.rotor = result.velocity = 0.0f;
    float shifted = free_velocity + k * armature * old_speed;
    float motor;
    if (gamma > 0.0f) motor = clamp(-bias / gamma, min_motor, max_motor);
    else if (bias < 0.0f) motor = max_motor;
    else if (bias > 0.0f) motor = min_motor;
    else motor = clamp(-shifted / k, min_motor, max_motor);
    float at_rest = shifted + k * motor;
    float friction_bound = k * friction;
    float signed_friction = at_rest > 0.0f ? friction : -friction;
    float b = 1.0f + k * (damping + armature);
    float motor_rhs = shifted - k * signed_friction + b * bias;
    float motor_denominator = b * gamma + k;
    if (!(b > 0.0f) || !(motor_denominator > 0.0f)
        || !all(isfinite(float4(min_motor, max_motor, shifted, at_rest)))
        || !all(isfinite(float4(friction_bound, b, motor_rhs, motor_denominator))))
    {
        state.flags.y = JointStatusNumericalFailure;
        return result;
    }
    if (abs(at_rest) <= friction_bound)
    {
        result.total = -free_velocity / k;
        result.rotor = armature * old_speed;
        result.motor = motor;
        result.passive = result.total - motor - result.rotor;
        result.velocity = 0.0f;
        return result;
    }
    motor = clamp(-motor_rhs / motor_denominator, min_motor, max_motor);
    result.total = (motor - signed_friction - damping * free_velocity + armature * (old_speed - free_velocity)) / b;
    float velocity = (free_velocity + k * (motor - signed_friction + armature * old_speed)) / b;
    result.velocity = velocity;
    result.motor = motor;
    result.passive = -damping * velocity - signed_friction;
    result.rotor = (armature / b) * (old_speed - free_velocity + k * (damping * old_speed - motor + signed_friction));
    return result;
}

void jph_joint_solve_velocity(JointAuthor a, inout JointState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float dt)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;
    if (a.bodies.z == JointSlider)
    {
        jph_slider_solve_velocity(a, state, m1, s1, m2, s2);
        return;
    }
    if (a.bodies.z == JointFixed)
    {
        float3 lambda = jph_joint_solve_rotation(m1, s1, m2, s2,
            jph_joint_angular_velocity(m1, s1) - jph_joint_angular_velocity(m2, s2), state);
        state.angularImpulse.xyz += lambda;
        jph_joint_apply_angular(m1, s1, m2, s2, lambda, false);
    }
    else if (jph_joint_drive_active(a))
    {
        float3 axis = jph_joint_axis(s1, state);
        float3 response1 = jph_inverse_inertia(m1, s1, axis);
        float3 response2 = jph_inverse_inertia(m2, s2, axis);
        float k1 = dot(axis, response1), k2 = dot(axis, response2);
        float k = k1 + k2;
        if (k > 0.0f && isfinite(k) && k1 >= 0.0f && k2 >= 0.0f)
        {
            float free_velocity = dot(axis, jph_joint_angular_velocity(m2, s2) - jph_joint_angular_velocity(m1, s1)) - k * state.axisImpulse.w;
            bool driven = (a.bodies.w & JointMotorEnabled) != 0u && (a.motor.y > 0.0f || a.motor.z > 0.0f);
            float torque = driven ? dt * a.motor.w : 0.0f;
            JphJointDriveImpulse solved = jph_joint_drive(free_velocity, k, state.setup.y, state.setup.z,
                -torque, torque, dt * a.passive.x, dt * a.passive.y, a.passive.z, state.rotor.z, state);
            if (state.flags.y != JointStatusSuccess) return;
            if (!isfinite(solved.motor) || !isfinite(solved.passive) || !isfinite(solved.total)
                || !isfinite(solved.rotor) || !isfinite(solved.velocity))
            {
                state.flags.y = JointStatusNumericalFailure;
                return;
            }
            state.axisImpulse.yzw = float3(solved.motor, solved.passive, solved.total);
            state.rotor.y = solved.rotor;
            jph_joint_apply_drive_velocity(m1, s1, m2, s2, axis, response1, response2, k1, k2,
                solved.velocity, state);
            if (state.flags.y != JointStatusSuccess) return;
        }
        else
        {
            state.flags.y = JointStatusNumericalFailure;
            return;
        }
    }
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 point_lambda = jph_joint_solve_point(m1, s1, r1, m2, s2, r2,
        jph_point_velocity(m1, s1, r1) - jph_point_velocity(m2, s2, r2), state);
    state.pointImpulse.xyz += point_lambda;
    jph_joint_apply_point(m1, s1, r1, m2, s2, r2, point_lambda, false);
    if (a.bodies.z == JointHinge)
    {
        float3 j0, j1;
        float2 error;
        jph_joint_swing_basis(s1, s2, state, j0, j1, error);
        float3 dw = jph_joint_angular_velocity(m1, s1) - jph_joint_angular_velocity(m2, s2);
        float2 lambda = jph_joint_solve2(m1, s1, m2, s2, j0, j1, float2(dot(j0, dw), dot(j1, dw)), state);
        state.angularImpulse.xy += lambda;
        jph_joint_apply_angular(m1, s1, m2, s2, j0 * lambda.x + j1 * lambda.y, false);
        if (state.setup.w != 0.0f)
        {
            float3 axis = jph_joint_axis(s1, state);
            float k = dot(axis, jph_joint_inertia_sum(m1, s1, m2, s2, axis));
            if (k > 0.0f)
            {
                float old_lambda = state.axisImpulse.x;
                float next_lambda = old_lambda + dot(axis, jph_joint_angular_velocity(m1, s1) - jph_joint_angular_velocity(m2, s2)) / k;
                if (a.limits.x != a.limits.y)
                    next_lambda = state.setup.w < 0.0f ? max(0.0f, next_lambda) : min(0.0f, next_lambda);
                state.axisImpulse.x = next_lambda;
                jph_joint_apply_angular(m1, s1, m2, s2, axis * (next_lambda - old_lambda), false);
            }
        }
    }
}

void jph_joint_solve_position(JointAuthor a, inout JointState state,
    BodyModel m1, inout BodyState s1, BodyModel m2, inout BodyState s2, float baumgarte)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess ||
        (a.bodies.w & JointEnabled) == 0u || (!jph_dynamic(m1) && !jph_dynamic(m2)))
        return;
    if (a.bodies.z == JointSlider)
    {
        jph_slider_solve_position(a, state, m1, s1, m2, s2, baumgarte);
        return;
    }
    if (a.bodies.z == JointFixed)
    {
        float4 diff = jph_joint_rotation_error(s1, s2, state);
        float3 error = 2.0f * (diff.w < 0.0f ? -diff.xyz : diff.xyz);
        float3 lambda = jph_joint_solve_rotation(m1, s1, m2, s2, -baumgarte * error, state);
        jph_joint_apply_angular(m1, s1, m2, s2, lambda, true);
    }
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 separation = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    float3 point_lambda = jph_joint_solve_point(m1, s1, r1, m2, s2, r2, -baumgarte * separation, state);
    jph_joint_apply_point(m1, s1, r1, m2, s2, r2, point_lambda, true);
    if (a.bodies.z == JointHinge)
    {
        float3 j0, j1;
        float2 error;
        jph_joint_swing_basis(s1, s2, state, j0, j1, error);
        float2 lambda = jph_joint_solve2(m1, s1, m2, s2, j0, j1, -baumgarte * error, state);
        jph_joint_apply_angular(m1, s1, m2, s2, j0 * lambda.x + j1 * lambda.y, true);
        float side, angle_error;
        jph_joint_limit(a, jph_joint_angle(s1, s2, state), side, angle_error);
        if (side != 0.0f)
        {
            float3 axis = jph_joint_axis(s1, state);
            float k = dot(axis, jph_joint_inertia_sum(m1, s1, m2, s2, axis));
            if (k > 0.0f)
                jph_joint_apply_angular(m1, s1, m2, s2, axis * (-baumgarte * angle_error / k), true);
        }
    }
}

bool jph_joint_finite(BodyState state)
{
    return all(isfinite(state.positionCOM)) && all(isfinite(state.rotation))
        && all(isfinite(state.linearVelocity)) && all(isfinite(state.angularVelocity));
}

bool jph_joint_state_finite(JointState state)
{
    return all(isfinite(state.pointImpulse)) && all(isfinite(state.angularImpulse))
        && all(isfinite(state.axisImpulse)) && all(isfinite(state.rotor)) && all(isfinite(state.setup));
}

bool jph_joint_suppresses_pair(WorldRange world, uint body1, uint body2)
{
    for (uint local = 0u; local < world.leaves.w; ++local)
    {
        JointAuthor a = jph_joint_load_author(world.leaves.z + local);
        if ((a.bodies.w & (JointEnabled | JointCollisionEnabled)) == JointEnabled &&
            ((a.bodies.x == body1 && a.bodies.y == body2) || (a.bodies.x == body2 && a.bodies.y == body1)))
            return true;
    }
    return false;
}

#endif // JPH_GPU_RIGID_JOINTS_SLANG
