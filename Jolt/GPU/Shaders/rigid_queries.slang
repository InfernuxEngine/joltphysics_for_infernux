// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"

ConstantBuffer<RaycastParams> raycast_params;
StructuredBuffer<RaycastInput> raycast_inputs;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<BodyState> body_states;
RWStructuredBuffer<RaycastHit> raycast_hits;

float3 query_box_face_normal(float3 point, float3 half_extent)
{
    // BoxShape::GetSurfaceNormal: nearest outer face, lowest axis on ties.
    float3 distance = abs(abs(point) - half_extent);
    uint axis = distance.y < distance.x ? 1u : 0u;
    if (distance.z < distance[axis])
        axis = 2u;
    float3 normal = float3(0.0f);
    normal[axis] = point[axis] > 0.0f ? 1.0f : -1.0f;
    return normal;
}

bool query_sphere(float3 origin, float3 direction, float radius, float maximum,
                  out float distance, out float3 normal)
{
    distance = 0.0f;
    normal = float3(0.0f, 1.0f, 0.0f);
    // Scaling the quadratic avoids overflow when finite origins are large.
    float scale = max(max(abs(origin.x), max(abs(origin.y), abs(origin.z))), radius);
    float3 scaled = origin / scale;
    float scaled_radius = radius / scale;
    float c = dot(scaled, scaled) - scaled_radius * scaled_radius;
    if (c > 0.0f)
    {
        float b = dot(scaled, direction);
        float discriminant = b * b - dot(direction, direction) * c;
        if (b >= 0.0f || discriminant < 0.0f)
            return false;
        // Stable entering root: c / (-b + sqrt(b*b-a*c)).
        distance = (c / (-b + sqrt(discriminant))) * scale;
        if (distance > maximum)
            return false;
    }
    float3 point = origin + direction * distance;
    float point_scale = max(abs(point.x), max(abs(point.y), abs(point.z)));
    if (point_scale > 0.0f)
        normal = normalize(point / point_scale);
    return true;
}

bool query_rounded_box(float3 origin, float3 direction, float4 geometry, float maximum,
                       out float distance, out float3 normal)
{
    // Keep finite author coordinates out of squared-distance overflow. A ray's
    // entry is computed in these scaled units; only the final distance leaves
    // them. A tiny shape and large far plane may overflow the division, but
    // no entering surface requires a normalized endpoint beyond float range.
    float scale = max(max(abs(origin.x), max(abs(origin.y), abs(origin.z))),
                      max(geometry.x, max(geometry.y, geometry.z)));
    origin /= scale;
    geometry /= scale;
    maximum = min(maximum / scale, 3.402823466e+38f);
    distance = 0.0f;
    normal = query_box_face_normal(origin, geometry.xyz);
    float radius = geometry.w;
    float3 core = max(geometry.xyz - radius, float3(0.0f));
    float3 delta = origin - clamp(origin, -core, core);
    float delta_scale = max(max(abs(delta.x), max(abs(delta.y), abs(delta.z))), radius);
    if (delta_scale == 0.0f || dot(delta / delta_scale, delta / delta_scale) <= (radius / delta_scale) * (radius / delta_scale))
        return true; // Solid convex volume, including its boundary.

    // Squared distance to an axis-aligned core is piecewise quadratic. Its
    // active outside axes change only at these six core-plane crossings.
    // Solve at most seven intervals analytically: no ray marching/tolerance,
    // and cylindrical edges/spherical corners retain the convex radius.
    float points[8];
    points[0] = 0.0f;
    points[1] = maximum;
    uint count = 2u;
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        if (direction[axis] == 0.0f)
            continue;
        for (uint side = 0u; side < 2u; ++side)
        {
            float plane = side == 0u ? -core[axis] : core[axis];
            float t = (plane - origin[axis]) / direction[axis];
            if (t > 0.0f && t < maximum)
                points[count++] = t;
        }
    }
    for (uint i = 1u; i < count; ++i)
    {
        float value = points[i];
        uint j = i;
        for (uint move = 0u; move < 7u && j > 0u; ++move)
        {
            if (points[j - 1u] <= value)
                break;
            points[j] = points[j - 1u];
            --j;
        }
        points[j] = value;
    }
    for (uint interval = 0u; interval + 1u < count; ++interval)
    {
        float start = points[interval];
        float width = points[interval + 1u] - start;
        if (width <= 0.0f)
            continue;
        float3 at_start = origin + start * direction;
        float3 middle = origin + (start + width * 0.5f) * direction;
        float3 offset = float3(0.0f);
        float3 velocity = float3(0.0f);
        for (uint axis = 0u; axis < 3u; ++axis)
        {
            if (middle[axis] < -core[axis] || middle[axis] > core[axis])
            {
                offset[axis] = at_start[axis] - (middle[axis] < 0.0f ? -core[axis] : core[axis]);
                velocity[axis] = direction[axis];
            }
        }
        float root = 0.0f;
        // Normalize this interval again: a large aspect ratio can otherwise
        // underflow a small rounded edge after the outer coordinate scaling.
        float offset_scale = max(max(abs(offset.x), max(abs(offset.y), abs(offset.z))), radius);
        if (offset_scale > 0.0f)
        {
            float3 scaled_offset = offset / offset_scale;
            float scaled_radius = radius / offset_scale;
            float c = dot(scaled_offset, scaled_offset) - scaled_radius * scaled_radius;
            if (c > 0.0f)
            {
                float velocity_scale = max(abs(velocity.x), max(abs(velocity.y), abs(velocity.z)));
                if (velocity_scale == 0.0f)
                    continue;
                float3 scaled_velocity = velocity / velocity_scale;
                float a = dot(scaled_velocity, scaled_velocity);
                float b = dot(scaled_offset, scaled_velocity);
                if (b >= 0.0f)
                    continue;
                // Closest-approach residual avoids subtracting two large
                // nearly equal quadratic terms at an edge/corner tangent.
                float3 perpendicular = scaled_offset - scaled_velocity * (b / a);
                float discriminant = a * (scaled_radius * scaled_radius - dot(perpendicular, perpendicular));
                if (discriminant < 0.0f)
                    continue;
                root = ((c / (-b + sqrt(discriminant))) * offset_scale) / velocity_scale;
                if (root > width)
                    continue;
            }
        }
        float scaled_distance = start + root;
        float3 point = origin + scaled_distance * direction;
        float3 outward = point - clamp(point, -core, core);
        float normal_scale = max(abs(outward.x), max(abs(outward.y), abs(outward.z)));
        normal = normal_scale > 0.0f ? normalize(outward / normal_scale) : query_box_face_normal(point, geometry.xyz);
        distance = scaled_distance * scale;
        return true;
    }
    return false;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void raycast_closest(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= raycast_params.counts.x)
        return;
    RaycastInput ray = raycast_inputs[index];
    float3 origin = ray.originMaxDistance.xyz;
    float maximum = ray.originMaxDistance.w;
    // Max-component scaling avoids squaring the raw direction magnitude.
    // Subnormal handling still follows the target GPU's floating-point mode.
    float direction_scale = max(abs(ray.direction.x), max(abs(ray.direction.y), abs(ray.direction.z)));
    float3 direction = normalize(ray.direction.xyz / direction_scale);
    RaycastHit hit;
    hit.meta = uint4(InvalidQueryBodySlot, 0u, 0u, 0u);
    hit.positionDistance = float4(0.0f);
    hit.normalFraction = float4(0.0f);
    float closest = maximum;
    for (uint local = 0u; local < raycast_params.counts.z; ++local)
    {
        uint slot = raycast_params.counts.y + local;
        BodyModel model = body_models[slot];
        if (!jph_resident(model))
            continue;
        BodyState state = body_states[slot];
        for (uint local_leaf = 0u; local_leaf < model.meta.y; ++local_leaf)
        {
            uint leaf = model.meta.x + local_leaf;
            ShapeModel shape = shape_models[leaf];
            if ((ray.filter.x & (1u << shape.surface.x)) == 0u ||
                (ray.filter.y == 0u && (shape.meta.w & ShapeSensor) != 0u))
                continue;
            float3 center = state.positionCOM.xyz + jph_rotate(state.rotation, shape.localPositionCOM.xyz);
            float4 rotation = jph_quat_mul(state.rotation, shape.localRotation);
            float3 local_origin = jph_inverse_rotate(rotation, origin - center);
            float3 local_direction = jph_inverse_rotate(rotation, direction);
            float distance;
            float3 normal;
            bool found;
            if (shape.meta.x == ShapeSphere)
                found = query_sphere(local_origin, local_direction, shape.geometry.w, closest, distance, normal);
            else if (shape.meta.x == ShapeBox)
                found = query_rounded_box(local_origin, local_direction, shape.geometry, closest, distance, normal);
            else
                continue; // Only admitted geometry has a query implementation.
            if (found && (hit.meta.y == 0u || distance < closest))
            {
                // Body slots and their admitted leaf ranges are visited in
                // stable order within this immutable publication.
                closest = distance;
                hit.meta = uint4(slot, 1u, leaf, 0u);
                hit.positionDistance = float4(origin + direction * distance, distance);
                hit.normalFraction = float4(jph_rotate(rotation, normal), distance / maximum);
            }
        }
    }
    raycast_hits[index] = hit; // A complete output record for every ray, including misses.
}
