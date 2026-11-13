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

float3 jph_kinematic_angular_velocity(float4 target, float4 current, float requested_dt)
{
    float4 delta = jph_quat_mul(target, float4(-current.xyz, current.w));
    // q and -q choose the same shortest arc, including an exact PI tie.
    if (delta.w < 0.0f || (delta.w == 0.0f && (delta.x < 0.0f ||
        (delta.x == 0.0f && (delta.y < 0.0f || (delta.y == 0.0f && delta.z < 0.0f))))))
        delta = -delta;
    float length_squared = dot(delta.xyz, delta.xyz);
    if (length_squared < 1e-12f)
        return (2.0f * delta.xyz) / requested_dt;
    float imaginary_length = sqrt(length_squared);
    float angle = 2.0f * atan2(imaginary_length, delta.w);
    return (delta.xyz * (angle / imaginary_length)) / requested_dt;
}

float4 jph_kinematic_rotation_step(float4 rotation, float3 angular_step)
{
    float length_squared = dot(angular_step, angular_step);
    if (length_squared == 0.0f)
        return rotation;
    float sine_scale;
    float cosine;
    if (length_squared < 1e-12f)
    {
        // Continuous exponential at zero; a small target is not discarded
        // merely because collision substeps make each angular step tiny.
        sine_scale = 0.5f - length_squared / 48.0f;
        cosine = 1.0f - length_squared / 8.0f;
    }
    else
    {
        float angle = sqrt(length_squared);
        sine_scale = sin(0.5f * angle) / angle;
        cosine = cos(0.5f * angle);
    }
    return normalize(jph_quat_mul(float4(angular_step * sine_scale, cosine), rotation));
}

KinematicTarget jph_empty_kinematic_target()
{
    KinematicTarget target;
    target.positionSpeed = float4(0.0f);
    target.rotation = float4(0.0f);
    target.rootCOMDt = float4(0.0f);
    target.state = uint4(0u);
    return target;
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

bool jph_nonstatic(BodyModel model)
{
    return (model.meta.z & ~MotionNotResident) != MotionStatic;
}

bool jph_has_kinematic_move(KinematicTarget target)
{
    return (target.state.x & (TargetPending | TargetActive | TargetDrivenLastTick)) != 0u;
}

BodyModel jph_effective_model(BodyModel author, KinematicTarget target)
{
    // This local view never writes BodyModels: accepted author Static remains
    // Static while the explicit GPU drive state supplies temporary movement.
    if ((author.meta.z & ~MotionNotResident) == MotionStatic &&
        (target.state.x & TargetTemporaryStatic) != 0u &&
        (target.state.x & (TargetActive | TargetDrivenLastTick)) != 0u)
        author.meta.z = (author.meta.z & MotionNotResident) | MotionKinematic;
    return author;
}

bool jph_resident(BodyModel model)
{
    return (model.meta.z & MotionNotResident) == 0u;
}

bool jph_sensor(ShapeModel shape)
{
    return (shape.meta.w & ShapeSensor) != 0u;
}

bool jph_layers_collide(ShapeModel a, ShapeModel b)
{
    return (a.surface.y & (1u << b.surface.x)) != 0u &&
           (b.surface.y & (1u << a.surface.x)) != 0u;
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
    // Moving bodies provide prescribed velocity, but only Dynamic bodies
    // have nonzero inverse mass/inertia and receive solver impulses.
    return jph_nonstatic(model)
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


// Rounded boxes are their reduced OBB cores plus a spherical convex radius.
// All work below is bounded: 15 SAT axes and 16 vertex/OBB + 144 edge/edge
// candidates. SAT separation alone is not a rounded-box distance query.
struct JphCoreBox
{
    float3 center;
    float3 axis[3];
    float3 halfExtent;
    float radius;
};

struct JphBoxPoint
{
    float3 pointA;
    float3 pointB;
    float depth;
    uint feature;
};

struct JphBoxManifold
{
    float3 normal;
    uint count;
    // Exact primary distance test, independent of contact-area point reduction.
    // Patch overlap must not be inferred from the speculative contact band.
    float penetration;
    JphBoxPoint points[MaxContactPointsPerPair];
};

JphCoreBox jph_core_box(float3 center, float4 rotation, float4 geometry)
{
    JphCoreBox box;
    box.center = center;
    box.axis[0] = jph_rotate(rotation, float3(1.0f, 0.0f, 0.0f));
    box.axis[1] = jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f));
    box.axis[2] = jph_rotate(rotation, float3(0.0f, 0.0f, 1.0f));
    box.halfExtent = max(geometry.xyz - geometry.w, float3(0.0f));
    box.radius = geometry.w;
    return box;
}

float3 jph_core_vertex(JphCoreBox box, uint index)
{
    return box.center
        + box.axis[0] * ((index & 1u) != 0u ? box.halfExtent.x : -box.halfExtent.x)
        + box.axis[1] * ((index & 2u) != 0u ? box.halfExtent.y : -box.halfExtent.y)
        + box.axis[2] * ((index & 4u) != 0u ? box.halfExtent.z : -box.halfExtent.z);
}

float3 jph_core_closest(JphCoreBox box, float3 point)
{
    float3 delta = point - box.center;
    float3 local = float3(dot(delta, box.axis[0]), dot(delta, box.axis[1]), dot(delta, box.axis[2]));
    local = clamp(local, -box.halfExtent, box.halfExtent);
    return box.center + box.axis[0] * local.x + box.axis[1] * local.y + box.axis[2] * local.z;
}

void jph_core_edge(JphCoreBox box, uint index, out float3 begin, out float3 end)
{
    uint axis = index / 4u;
    uint side = index % 4u;
    uint u = (axis + 1u) % 3u;
    uint v = (axis + 2u) % 3u;
    float3 center = box.center
        + box.axis[u] * ((side & 1u) != 0u ? box.halfExtent[u] : -box.halfExtent[u])
        + box.axis[v] * ((side & 2u) != 0u ? box.halfExtent[v] : -box.halfExtent[v]);
    begin = center - box.axis[axis] * box.halfExtent[axis];
    end = center + box.axis[axis] * box.halfExtent[axis];
}

void jph_segment_closest(float3 a0, float3 a1, float3 b0, float3 b1,
                          out float3 pointA, out float3 pointB)
{
    float3 da = a1 - a0;
    float3 db = b1 - b0;
    float3 delta = a0 - b0;
    float aa = dot(da, da);
    float bb = dot(db, db);
    float s = 0.0f;
    float t = 0.0f;
    if (aa <= 1e-20f)
    {
        if (bb > 1e-20f)
            t = clamp(dot(db, delta) / bb, 0.0f, 1.0f);
    }
    else
    {
        float ad = dot(da, delta);
        if (bb <= 1e-20f)
            s = clamp(-ad / aa, 0.0f, 1.0f);
        else
        {
            float ab = dot(da, db);
            float bd = dot(db, delta);
            // Cross products avoid subtracting almost equal aa*bb and ab*ab.
            float3 cross_ab = cross(da, db);
            float denominator = dot(cross_ab, cross_ab);
            if (denominator > 1e-20f * aa * bb)
                s = clamp(dot(cross(db, delta), cross_ab) / denominator, 0.0f, 1.0f);
            t = (ab * s + bd) / bb;
            if (t < 0.0f)
            {
                t = 0.0f;
                s = clamp(-ad / aa, 0.0f, 1.0f);
            }
            else if (t > 1.0f)
            {
                t = 1.0f;
                s = clamp((ab - ad) / aa, 0.0f, 1.0f);
            }
        }
    }
    pointA = a0 + s * da;
    pointB = b0 + t * db;
}

void jph_closest_candidate(float3 a, float3 b, uint id, inout float best,
                            inout float3 pointA, inout float3 pointB, inout uint feature)
{
    float3 delta = b - a;
    float distance_squared = dot(delta, delta);
    if (distance_squared < best)
    {
        best = distance_squared;
        pointA = a;
        pointB = b;
        feature = id;
    }
}

void jph_core_distance(JphCoreBox a, JphCoreBox b, out float3 pointA, out float3 pointB,
                        out float distance_squared, out uint feature)
{
    distance_squared = 3.402823466e+38f;
    pointA = a.center;
    pointB = b.center;
    feature = 0u;
    for (uint vertex = 0u; vertex < 8u; ++vertex)
    {
        float3 va = jph_core_vertex(a, vertex);
        float3 vb = jph_core_vertex(b, vertex);
        jph_closest_candidate(va, jph_core_closest(b, va), vertex,
                              distance_squared, pointA, pointB, feature);
        jph_closest_candidate(jph_core_closest(a, vb), vb, 8u + vertex,
                              distance_squared, pointA, pointB, feature);
    }
    for (uint edgeA = 0u; edgeA < 12u; ++edgeA)
    {
        float3 a0, a1;
        jph_core_edge(a, edgeA, a0, a1);
        for (uint edgeB = 0u; edgeB < 12u; ++edgeB)
        {
            float3 b0, b1, pa, pb;
            jph_core_edge(b, edgeB, b0, b1);
            jph_segment_closest(a0, a1, b0, b1, pa, pb);
            jph_closest_candidate(pa, pb, 16u + edgeA * 12u + edgeB,
                                  distance_squared, pointA, pointB, feature);
        }
    }
}

bool jph_core_sat(JphCoreBox a, JphCoreBox b, out float3 normal, out float depth, out uint feature)
{
    float3 delta = b.center - a.center;
    depth = 3.402823466e+38f;
    normal = a.axis[0];
    feature = 0u;
    bool overlapping = true;
    for (uint axis = 0u; axis < 15u; ++axis)
    {
        float3 n;
        if (axis < 3u)
            n = a.axis[axis];
        else if (axis < 6u)
            n = b.axis[axis - 3u];
        else
            n = cross(a.axis[(axis - 6u) / 3u], b.axis[(axis - 6u) % 3u]);
        float length_squared = dot(n, n);
        if (length_squared <= 1e-20f)
            continue; // Parallel edges do not define a separating axis.
        n /= sqrt(length_squared);
        float radiusA = abs(dot(a.axis[0], n)) * a.halfExtent.x
                      + abs(dot(a.axis[1], n)) * a.halfExtent.y
                      + abs(dot(a.axis[2], n)) * a.halfExtent.z;
        float radiusB = abs(dot(b.axis[0], n)) * b.halfExtent.x
                      + abs(dot(b.axis[1], n)) * b.halfExtent.y
                      + abs(dot(b.axis[2], n)) * b.halfExtent.z;
        float center_distance = dot(delta, n);
        float penetration = radiusA + radiusB - abs(center_distance);
        overlapping = overlapping && penetration >= 0.0f;
        // Face axes precede edge axes; exact ties retain that stable order.
        if (penetration < depth)
        {
            depth = penetration;
            normal = center_distance >= 0.0f ? n : -n;
            feature = axis;
        }
    }
    return overlapping;
}

uint jph_clip_box_face(float3 input_vertices[8], uint count, float3 normal, float limit,
                        out float3 output_vertices[8])
{
    // The caller supplies a nonempty polygon. Only the returned vertex range
    // is initialized; unused array tails are not geometry.
    uint output_count = 0u;
    float3 previous = input_vertices[count - 1u];
    float previous_distance = limit - dot(normal, previous);
    for (uint index = 0u; index < 8u; ++index)
    {
        if (index >= count)
            break;
        float3 current = input_vertices[index];
        float current_distance = limit - dot(normal, current);
        bool current_inside = current_distance >= 0.0f;
        // A boundary endpoint is emitted by the vertex branch exactly once.
        // Only a strict sign crossing creates a new polygon vertex.
        if ((previous_distance < 0.0f && current_distance > 0.0f)
            || (previous_distance > 0.0f && current_distance < 0.0f))
        {
            float fraction = previous_distance / (previous_distance - current_distance);
            output_vertices[output_count++] = previous + fraction * (current - previous);
        }
        if (current_inside)
            output_vertices[output_count++] = current;
        previous = current;
        previous_distance = current_distance;
    }
    return output_count;
}

uint jph_box_face_points(JphCoreBox a, JphCoreBox b, float3 normal, float speculative_distance,
                          inout JphBoxPoint points[9], uint count)
{
    // Extra contacts use real flat patches of the rounded shapes. Their plane
    // is at outer half extent, but their tangent extents belong to the CORE.
    // Full outer-box face quads would turn rounded corners into sharp corners.
    uint reference_shape = 0u;
    uint reference_axis = 0u;
    float alignment = 3.402823466e+38f;
    for (uint shape = 0u; shape < 2u; ++shape)
        for (uint axis = 0u; axis < 3u; ++axis)
        {
            float3 direction = shape == 0u ? a.axis[axis] : b.axis[axis];
            float3 perpendicular = cross(direction, normal);
            float error = dot(perpendicular, perpendicular);
            if (error < alignment)
            {
                alignment = error;
                reference_shape = shape;
                reference_axis = axis;
            }
        }
    JphCoreBox reference = a;
    JphCoreBox incident = b;
    if (reference_shape != 0u)
    {
        reference = b;
        incident = a;
    }
    float3 outward = reference_shape == 0u ? normal : -normal;
    float3 reference_normal = dot(reference.axis[reference_axis], outward) >= 0.0f
        ? reference.axis[reference_axis] : -reference.axis[reference_axis];
    float3 plane_origin = reference.center
        + reference_normal * (reference.halfExtent[reference_axis] + reference.radius);
    uint u = (reference_axis + 1u) % 3u;
    uint v = (reference_axis + 2u) % 3u;
    uint incident_axis = 0u;
    if (abs(dot(incident.axis[1], outward)) > abs(dot(incident.axis[incident_axis], outward)))
        incident_axis = 1u;
    if (abs(dot(incident.axis[2], outward)) > abs(dot(incident.axis[incident_axis], outward)))
        incident_axis = 2u;
    float3 incident_normal = dot(incident.axis[incident_axis], outward) >= 0.0f
        ? -incident.axis[incident_axis] : incident.axis[incident_axis];
    float3 face_origin = incident.center
        + incident_normal * (incident.halfExtent[incident_axis] + incident.radius);
    uint iu = (incident_axis + 1u) % 3u;
    uint iv = (incident_axis + 2u) % 3u;
    float3 du = incident.axis[iu] * incident.halfExtent[iu];
    float3 dv = incident.axis[iv] * incident.halfExtent[iv];
    float3 polygon[8];
    polygon[0] = face_origin - du - dv;
    polygon[1] = face_origin + du - dv;
    polygon[2] = face_origin + du + dv;
    polygon[3] = face_origin - du + dv;
    // Map the actual incident flat patch onto the reference plane along the
    // common contact direction BEFORE clipping. A slightly tilted normal must
    // not collapse a resting patch to one point, nor move a clipped endpoint
    // outside the reference patch. The most aligned reference/incident axes
    // guarantee nonzero plane denominators (absolute dot >= 1/sqrt(3)).
    float reference_denominator = dot(reference_normal, outward);
    for (uint vertex = 0u; vertex < 4u; ++vertex)
    {
        float distance = dot(plane_origin - polygon[vertex], reference_normal) / reference_denominator;
        polygon[vertex] += distance * outward;
    }
    uint polygon_count = 4u;
    // A quad clipped by four rectangle planes has at most eight vertices.
    for (uint side = 0u; side < 4u; ++side)
    {
        if (polygon_count == 0u)
            break;
        uint tangent = side < 2u ? u : v;
        float3 side_normal = (side & 1u) == 0u ? reference.axis[tangent] : -reference.axis[tangent];
        float limit = dot(side_normal, reference.center) + reference.halfExtent[tangent];
        float3 clipped[8];
        polygon_count = jph_clip_box_face(polygon, polygon_count, side_normal, limit, clipped);
        for (uint vertex = 0u; vertex < 8u; ++vertex)
            if (vertex < polygon_count)
                polygon[vertex] = clipped[vertex];
    }
    float incident_denominator = dot(incident_normal, outward);
    for (uint vertex = 0u; vertex < 8u; ++vertex)
    {
        if (vertex >= polygon_count)
            break;
        float3 reference_point = polygon[vertex];
        // Invert the projection to recover a point on the ORIGINAL incident
        // flat patch. Clipping in the reference plane preserves both finite
        // patch domains because this plane-to-plane map is affine/invertible.
        float distance = dot(face_origin - reference_point, incident_normal) / incident_denominator;
        float penetration = -distance;
        if (penetration < -speculative_distance)
            continue;
        float3 incident_point = reference_point + distance * outward;
        JphBoxPoint point;
        point.pointA = reference_shape == 0u ? reference_point : incident_point;
        point.pointB = reference_shape == 0u ? incident_point : reference_point;
        point.depth = penetration;
        point.feature = 1024u + reference_shape * 24u + reference_axis * 8u + vertex;
        points[count++] = point;
    }
    return count;
}

JphBoxManifold jph_box_box(float3 centerA, float4 rotationA, float4 geometryA,
                            float3 centerB, float4 rotationB, float4 geometryB, float speculative_distance)
{
    // Pair-relative coordinates keep the narrow phase independent of a large
    // shared world translation. Anchors are converted to world only at output.
    JphCoreBox a = jph_core_box(float3(0.0f), rotationA, geometryA);
    JphCoreBox b = jph_core_box(centerB - centerA, rotationB, geometryB);
    float3 normal;
    float core_depth;
    uint axis_feature;
    bool core_overlap = jph_core_sat(a, b, normal, core_depth, axis_feature);
    float3 pa, pb;
    float distance_squared;
    uint distance_feature;
    float penetration;
    if (core_overlap)
    {
        // At the minimum translation, the two core boundaries touch. Their
        // exact closest feature pair supplies matching support witnesses,
        // including partial face overlap, edge/edge and contained boxes.
        JphCoreBox translated = b;
        translated.center += core_depth * normal;
        jph_core_distance(a, translated, pa, pb, distance_squared, distance_feature);
        pb -= core_depth * normal;
        penetration = core_depth + a.radius + b.radius;
    }
    else
    {
        jph_core_distance(a, b, pa, pb, distance_squared, distance_feature);
        float distance = sqrt(distance_squared);
        if (distance_squared > 1e-20f)
            normal = (pb - pa) / distance;
        penetration = a.radius + b.radius - distance;
    }

    JphBoxManifold manifold;
    manifold.normal = normal;
    manifold.count = 0u;
    manifold.penetration = penetration;
    if (penetration < -speculative_distance)
        return manifold;
    JphBoxPoint candidates[9];
    candidates[0].pointA = pa + a.radius * normal;
    candidates[0].pointB = pb - b.radius * normal;
    candidates[0].depth = dot(candidates[0].pointA - candidates[0].pointB, normal);
    candidates[0].feature = core_overlap ? 64u + axis_feature : 128u + distance_feature;
    uint count = jph_box_face_points(a, b, normal, speculative_distance, candidates, 1u);

    // Remove duplicate boundary points before selecting a support polygon.
    float extent = max(1.0f, max(max(geometryA.x, geometryA.y), max(geometryA.z,
                              max(max(geometryB.x, geometryB.y), geometryB.z))));
    float tolerance_squared = 1e-12f * extent * extent;
    uint unique_count = 0u;
    for (uint point = 0u; point < 9u; ++point)
    {
        if (point >= count)
            break;
        bool duplicate = false;
        for (uint other = 0u; other < 9u; ++other)
        {
            if (other >= unique_count)
                break;
            float3 da = candidates[point].pointA - candidates[other].pointA;
            float3 db = candidates[point].pointB - candidates[other].pointB;
            if (dot(da, da) + dot(db, db) <= tolerance_squared)
            {
                if (candidates[point].depth > candidates[other].depth)
                    candidates[other] = candidates[point];
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
            candidates[unique_count++] = candidates[point];
    }

    // Depth first; equal-depth contacts prefer distance from the patch center.
    // Then retain the longest segment and its extreme points on both sides,
    // following Jolt's tangent-plane area reduction to at most four points.
    float3 center = float3(0.0f);
    for (uint point = 0u; point < 9u; ++point)
        if (point < unique_count)
            center += 0.5f * (candidates[point].pointA + candidates[point].pointB);
    center /= float(unique_count);
    uint first = 0u;
    float first_distance = -1.0f;
    float depth_tolerance = 1e-6f * extent;
    for (uint point = 0u; point < 9u; ++point)
    {
        if (point >= unique_count)
            break;
        float3 delta = 0.5f * (candidates[point].pointA + candidates[point].pointB) - center;
        delta -= normal * dot(delta, normal);
        float distance = dot(delta, delta);
        if (candidates[point].depth > candidates[first].depth + depth_tolerance ||
            (abs(candidates[point].depth - candidates[first].depth) <= depth_tolerance && distance > first_distance))
        {
            first = point;
            first_distance = distance;
        }
    }
    uint selected[4];
    selected[0] = first;
    uint selected_count = 1u;
    if (unique_count > 1u)
    {
        uint second = first;
        float longest = -1.0f;
        float3 p1 = 0.5f * (candidates[first].pointA + candidates[first].pointB);
        for (uint point = 0u; point < 9u; ++point)
        {
            if (point >= unique_count)
                break;
            if (point == first)
                continue;
            float3 delta = 0.5f * (candidates[point].pointA + candidates[point].pointB) - p1;
            delta -= normal * dot(delta, normal);
            float distance = dot(delta, delta);
            if (distance > longest)
            {
                longest = distance;
                second = point;
            }
        }
        selected[selected_count++] = second;
        float3 segment = 0.5f * (candidates[second].pointA + candidates[second].pointB) - p1;
        float3 perpendicular = cross(segment, normal);
        float minimum = 0.0f;
        float maximum = 0.0f;
        uint negative = 0xffffffffu;
        uint positive = 0xffffffffu;
        for (uint point = 0u; point < 9u; ++point)
        {
            if (point >= unique_count)
                break;
            if (point == first || point == second)
                continue;
            float3 delta = 0.5f * (candidates[point].pointA + candidates[point].pointB) - p1;
            float side = dot(perpendicular, delta);
            if (side < minimum)
            {
                minimum = side;
                negative = point;
            }
            else if (side > maximum)
            {
                maximum = side;
                positive = point;
            }
        }
        if (negative != 0xffffffffu)
            selected[selected_count++] = negative;
        if (positive != 0xffffffffu)
            selected[selected_count++] = positive;
    }
    manifold.count = selected_count;
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        if (point < selected_count)
        {
            JphBoxPoint contact = candidates[selected[point]];
            contact.pointA += centerA;
            contact.pointB += centerA;
            manifold.points[point] = contact;
        }
    return manifold;
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

void jph_solve_friction(BodyModel ma, inout BodyState sa, float3 ra,
                        BodyModel mb, inout BodyState sb, float3 rb,
                        float3 normal, float maximum_impulse, inout float2 impulse)
{
    float3 tangent1, tangent2;
    jph_tangents(normal, tangent1, tangent2);

    // ContactConstraintManager solves linear friction at the average contact
    // midpoint, with a circular bound from the previous iteration's SUM of
    // normal impulses. The manifold owns this pair of accumulated lambdas.
    float lambda1 = impulse.x
        + jph_effective_mass(ma, sa, ra, mb, sb, rb, tangent1) * jph_jv(ma, sa, ra, mb, sb, rb, tangent1);
    float lambda2 = impulse.y
        + jph_effective_mass(ma, sa, ra, mb, sb, rb, tangent2) * jph_jv(ma, sa, ra, mb, sb, rb, tangent2);
    float friction_sq = lambda1 * lambda1 + lambda2 * lambda2;
    if (friction_sq > maximum_impulse * maximum_impulse)
    {
        float scale = maximum_impulse / sqrt(friction_sq);
        lambda1 *= scale;
        lambda2 *= scale;
    }
    jph_apply_impulse(ma, sa, ra, mb, sb, rb, tangent1, lambda1 - impulse.x);
    jph_apply_impulse(ma, sa, ra, mb, sb, rb, tangent2, lambda2 - impulse.y);
    impulse = float2(lambda1, lambda2);
}

void jph_solve_angular_friction(BodyModel ma, inout BodyState sa, BodyModel mb, inout BodyState sb,
                                float3 normal, float maximum_impulse, inout float impulse)
{
    float3 inverse_axis_a = jph_inverse_inertia(ma, sa, normal);
    float3 inverse_axis_b = jph_inverse_inertia(mb, sb, normal);
    float inverse_effective_mass = dot(normal, inverse_axis_a + inverse_axis_b);
    float effective_mass = inverse_effective_mass > 0.0f ? 1.0f / inverse_effective_mass : 0.0f;
    float3 angular_a = jph_nonstatic(ma) ? sa.angularVelocity.xyz : float3(0.0f);
    float3 angular_b = jph_nonstatic(mb) ? sb.angularVelocity.xyz : float3(0.0f);
    float lambda = clamp(impulse + effective_mass * dot(normal, angular_a - angular_b),
                         -maximum_impulse, maximum_impulse);
    float delta = lambda - impulse;
    if (jph_dynamic(ma))
        sa.angularVelocity.xyz -= delta * inverse_axis_a;
    if (jph_dynamic(mb))
        sb.angularVelocity.xyz += delta * inverse_axis_b;
    impulse = lambda;
}

float4 jph_solve_normal_block(BodyModel ma, BodyState sa, BodyModel mb, BodyState sb,
                               float3 normal, float3 ja[4], float3 jb[4],
                               float4 old_impulse, float4 bias, uint count)
{
    // All points share n; their midpoint offsets vary in only two tangent
    // coordinates. Thus J, and this pair-local PSD K, have rank at most three.
    // The unilateral QP has a minimizer on an independent active basis of
    // size <= 3, even if a four-point force distribution is non-unique.
    float4 k[4];
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        k[point] = float4(0.0f);
    float inverse_mass = jph_inverse_mass(ma) + jph_inverse_mass(mb);
    for (uint column = 0u; column < MaxContactPointsPerPair; ++column)
    {
        if (column >= count)
            break;
        float3 inverse_a = jph_inverse_inertia(ma, sa, ja[column]);
        float3 inverse_b = jph_inverse_inertia(mb, sb, jb[column]);
        for (uint row = 0u; row < MaxContactPointsPerPair; ++row)
        {
            if (row > column)
                break;
            float value = inverse_mass + dot(ja[row], inverse_a) + dot(jb[row], inverse_b);
            k[row][column] = value;
            k[column][row] = value;
        }
    }

    float3 linear_a = jph_nonstatic(ma) ? sa.linearVelocity.xyz : float3(0.0f);
    float3 linear_b = jph_nonstatic(mb) ? sb.linearVelocity.xyz : float3(0.0f);
    float3 angular_a = jph_nonstatic(ma) ? sa.angularVelocity.xyz : float3(0.0f);
    float3 angular_b = jph_nonstatic(mb) ? sb.angularVelocity.xyz : float3(0.0f);
    float linear_jv = dot(normal, linear_a - linear_b);
    float4 q = float4(0.0f);
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        if (point < count)
            q[point] = linear_jv + dot(ja[point], angular_a) - dot(jb[point], angular_b)
                - bias[point] + dot(k[point], old_impulse);

    // Equilibrate only AFTER reconstructing q with physical K and old lambda.
    // Dii=sqrt(Kii), C=D^-1 K D^-1, r=D^-1 q and y=D lambda.
    // Reuse k for C. Its unit diagonal makes rank tests independent of the
    // different mass/inertia scales of otherwise independent contact rows.
    float4 inverse_scale = float4(0.0f);
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        if (point < count)
            inverse_scale[point] = 1.0f / sqrt(k[point][point]);
    for (uint column = 0u; column < MaxContactPointsPerPair; ++column)
    {
        if (column >= count)
            break;
        for (uint row = 0u; row < MaxContactPointsPerPair; ++row)
        {
            if (row > column)
                break;
            float value = k[row][column] * inverse_scale[row] * inverse_scale[column];
            k[row][column] = value;
            k[column][row] = value;
        }
        k[column][column] = 1.0f;
    }
    float4 scaled_q = q * inverse_scale;

    // Empty set plus 4 singleton, 6 pair and 4 triple bases: at most fifteen
    // candidates. Zero is a normal feasible QP candidate, not a PGS fallback.
    float4 best = float4(0.0f);
    for (uint mask = 1u; mask < 16u; ++mask)
    {
        if (mask >= (1u << count))
            break;
        uint size = (mask & 1u) + ((mask >> 1u) & 1u) + ((mask >> 2u) & 1u) + ((mask >> 3u) & 1u);
        if (size > 3u)
            continue;
        uint indices[3];
        uint next = 0u;
        for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
            if ((mask & (1u << point)) != 0u)
                indices[next++] = point;

        uint i = indices[0];
        // Relative to C's unit diagonal, LDL pivots identify duplicate or
        // collinear rows without treating a differently scaled row as null.
        float rank_threshold = 1e-6f;
        float d0 = k[i][i];
        if (d0 <= rank_threshold)
            continue;
        float y0 = scaled_q[i];
        float z0 = y0 / d0;
        float3 solution = float3(z0, 0.0f, 0.0f);
        if (size > 1u)
        {
            uint j = indices[1];
            float l10 = k[j][i] / d0;
            float d1 = k[j][j] - l10 * k[j][i];
            if (d1 <= rank_threshold)
                continue;
            float y1 = scaled_q[j] - l10 * y0;
            float z1 = y1 / d1;
            solution.y = z1;
            solution.x = z0 - l10 * solution.y;
            if (size > 2u)
            {
                uint third = indices[2];
                float l20 = k[third][i] / d0;
                float l21 = (k[third][j] - l20 * k[j][i]) / d1;
                float d2 = k[third][third] - l20 * k[third][i] - l21 * l21 * d1;
                if (d2 <= rank_threshold)
                    continue;
                float y2 = scaled_q[third] - l20 * y0 - l21 * y1;
                solution.z = y2 / d2;
                solution.y = z1 - l21 * solution.z;
                solution.x = z0 - l10 * solution.y - l20 * solution.z;
            }
        }
        if (any(solution < float3(0.0f)))
            continue;
        float4 candidate = float4(0.0f);
        candidate[indices[0]] = solution.x;
        if (size > 1u)
            candidate[indices[1]] = solution.y;
        if (size > 2u)
            candidate[indices[2]] = solution.z;
        // E(y)=.5*y^T*C*y-r^T*y equals the physical impulse objective.
        // Compare E(candidate)-E(best) directly: subtracting two large total
        // energies can erase a real improvement on a smaller-scale row.
        float4 sum = candidate + best;
        float4 response = float4(dot(k[0], sum), dot(k[1], sum), dot(k[2], sum), dot(k[3], sum));
        float energy_delta = dot(candidate - best, 0.5f * response - scaled_q);
        if (energy_delta < 0.0f)
            best = candidate;
    }
    return best * inverse_scale;
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
