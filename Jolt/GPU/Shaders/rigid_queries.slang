// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"

ConstantBuffer<QueryParams> query_params;
StructuredBuffer<QueryInput> query_inputs;
StructuredBuffer<QueryPrimitive> query_primitives;
StructuredBuffer<QueryBodyIdentity> query_body_ids;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<BodyState> body_states;
RWStructuredBuffer<QueryHit> query_candidates;
RWStructuredBuffer<QuerySpan> query_spans;
RWStructuredBuffer<QuerySummary> query_summary;
RWStructuredBuffer<QueryHit> query_hits;
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

float query_entering_distance(float b, float a, float root, float c, float scale)
{
    // The same quadratic has two conditioned forms. The projection minus
    // chord preserves distant hits; c/q preserves a small near-surface gap.
    return root <= -0.5f * b ? (-b / a) * scale - (root / a) * scale
                             : (c / (-b + root)) * scale;
}

float query_squared_radius_difference(float3 point, float radius, float scale)
{
    uint axis = abs(point.y) > abs(point.x) ? 1u : 0u;
    if (abs(point.z) > abs(point[axis]))
        axis = 2u;
    float coordinate = abs(point[axis]);
    float3 other = point / scale;
    other[axis] = 0.0f;
    return ((coordinate - radius) / scale) * (coordinate / scale + radius / scale) + dot(other, other);
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
    float c = query_squared_radius_difference(origin, radius, scale);
    if (c > 0.0f)
    {
        float b = dot(scaled, direction);
        float a = dot(direction, direction);
        float3 perpendicular = scaled - direction * (b / a);
        float discriminant = a * (scaled_radius * scaled_radius - dot(perpendicular, perpendicular));
        if (b >= 0.0f || discriminant < 0.0f)
            return false;
        float root = sqrt(discriminant);
        distance = query_entering_distance(b, a, root, c, scale);
        if (distance > maximum)
            return false;
        // Use the small entry residual, not origin + distance*direction:
        // distant origins can erase the entire radius in that subtraction.
        float3 entry = perpendicular - direction * (root / a);
        float normal_scale = max(abs(entry.x), max(abs(entry.y), abs(entry.z)));
        normal = normal_scale > 0.0f ? normalize(entry / normal_scale) : -normalize(direction);
        return true;
    }
    float3 point = origin + direction * distance;
    float point_scale = max(abs(point.x), max(abs(point.y), abs(point.z)));
    if (point_scale > 0.0f)
        normal = normalize(point / point_scale);
    return true;
}

bool query_capsule(float3 origin, float3 direction, float half_height, float radius, float maximum,
                   out float distance, out float3 normal)
{
    float3 core = float3(0.0f, clamp(origin.y, -half_height, half_height), 0.0f);
    float3 radial = origin - core;
    float radial_scale = max(max(abs(radial.x), max(abs(radial.y), abs(radial.z))), radius);
    distance = 0.0f;
    normal = float3(1.0f, 0.0f, 0.0f);
    if (query_squared_radius_difference(radial, radius, radial_scale) <= 0.0f)
    {
        float normal_scale = max(abs(radial.x), max(abs(radial.y), abs(radial.z)));
        if (normal_scale > 0.0f)
            normal = normalize(radial / normal_scale);
        return true;
    }

    bool found = false;
    float closest = maximum;
    // A capsule is the union of its finite cylinder and endpoint spheres.
    // Starting outside that union, its first entering component is the true
    // surface; internal sphere hemispheres cannot become an earlier hit.
    for (uint end = 0u; end < 2u; ++end)
    {
        float center_y = end == 0u ? -half_height : half_height;
        float candidate;
        float3 candidate_normal;
        if (query_sphere(origin - float3(0.0f, center_y, 0.0f), direction, radius,
                         closest, candidate, candidate_normal))
        {
            closest = candidate;
            normal = candidate_normal;
            found = true;
        }
    }

    float2 planar_origin = origin.xz;
    float2 planar_direction = direction.xz;
    float velocity_scale = max(abs(planar_direction.x), abs(planar_direction.y));
    if (velocity_scale > 0.0f)
    {
        // Normalize direction and radial coordinates separately so a nearly
        // axial ray does not disappear when its squared planar speed underflows.
        float position_scale = max(max(abs(planar_origin.x), abs(planar_origin.y)), radius);
        float2 offset = planar_origin / position_scale;
        float2 velocity = planar_direction / velocity_scale;
        float r = radius / position_scale;
        float a = dot(velocity, velocity);
        float b = dot(offset, velocity);
        uint axis = abs(planar_origin.y) > abs(planar_origin.x) ? 1u : 0u;
        float coordinate = abs(planar_origin[axis]);
        float other = offset[1u - axis];
        float c = ((coordinate - radius) / position_scale) * (coordinate / position_scale + r) + other * other;
        if (c > 0.0f && b < 0.0f)
        {
            float2 perpendicular = offset - velocity * (b / a);
            float discriminant = a * (r * r - dot(perpendicular, perpendicular));
            if (discriminant >= 0.0f)
            {
                float root = sqrt(discriminant);
                float candidate = query_entering_distance(b, a, root, c, position_scale) / velocity_scale;
                float y = origin.y + direction.y * candidate;
                if (candidate <= closest && y >= -half_height && y <= half_height)
                {
                    float2 surface = perpendicular - velocity * (root / a);
                    float surface_scale = max(abs(surface.x), abs(surface.y));
                    float2 outward = surface_scale > 0.0f ? surface / surface_scale : -velocity;
                    normal = normalize(float3(outward.x, 0.0f, outward.y));
                    closest = candidate;
                    found = true;
                }
            }
        }
    }
    distance = closest;
    return found;
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

QueryHit query_miss()
{
    QueryHit hit;
    hit.meta = uint4(InvalidQueryBodySlot, 0u, InvalidQueryBodySlot, 0u);
    hit.positionDistance = float4(0.0f);
    hit.normalFraction = float4(0.0f);
    hit.pointADepth = float4(0.0f);
    return hit;
}

bool query_accepts(QueryInput request, ShapeModel shape, uint slot)
{
    if ((request.meta.w & QueryExcludeBody) != 0u && slot == request.source.x)
        return false;
    if ((request.meta.w & QueryTargetCollider) != 0u)
        return all(shape.source.xy == request.filter.zw);
    return (request.filter.x & (1u << shape.surface.x)) != 0u &&
           (request.filter.y != 0u || (shape.meta.w & ShapeSensor) == 0u);
}

void query_leaf_pose(ShapeModel shape, BodyState state, out float3 center, out float4 rotation)
{
    center = state.positionCOM.xyz + jph_rotate(state.rotation, shape.localPositionCOM.xyz);
    rotation = jph_quat_mul(state.rotation, shape.localRotation);
}

void query_leaf_bounds(ShapeModel shape, BodyState state, out float3 low, out float3 high)
{
    float3 center;
    float4 rotation;
    query_leaf_pose(shape, state, center, rotation);
    float3 extent = float3(shape.geometry.w);
    if (shape.meta.x == ShapeBox)
    {
        float3 core = max(shape.geometry.xyz - shape.geometry.w, float3(0.0f));
        extent += abs(jph_rotate(rotation, float3(1.0f, 0.0f, 0.0f))) * core.x +
                  abs(jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f))) * core.y +
                  abs(jph_rotate(rotation, float3(0.0f, 0.0f, 1.0f))) * core.z;
    }
    else if (shape.meta.x == ShapeCapsule)
        extent += abs(jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f))) * shape.geometry.y;
    low = center - extent;
    high = center + extent;
}

uint query_metric_status(JphPairMetric metric)
{
    return metric.status == 1u ? QuerySuccess :
           metric.status == 0u ? QueryUnsupported : QueryNumericalFailure;
}

QueryHit query_leaf(QueryInput request, ShapeModel shape, BodyState state, uint leaf, out uint status)
{
    status = QuerySuccess;
    QueryHit hit = query_miss();
    float3 center;
    float4 rotation;
    query_leaf_pose(shape, state, center, rotation);
    if (!all(isfinite(center)) || !all(isfinite(rotation)))
    {
        status = QueryNumericalFailure;
        return hit;
    }
    if (request.meta.x == QueryRay)
    {
        float3 direction = jph_capsule_unit(request.direction.xyz);
        float3 local_origin = jph_inverse_rotate(rotation, request.positionMaximum.xyz - center);
        float3 local_direction = jph_inverse_rotate(rotation, direction);
        if (!all(isfinite(local_origin)) || !all(isfinite(local_direction)))
        {
            status = QueryNumericalFailure;
            return hit;
        }
        float distance;
        float3 normal;
        bool found;
        if (shape.meta.x == ShapeSphere)
            found = query_sphere(local_origin, local_direction, shape.geometry.w, request.positionMaximum.w, distance, normal);
        else if (shape.meta.x == ShapeBox)
            found = query_rounded_box(local_origin, local_direction, shape.geometry, request.positionMaximum.w, distance, normal);
        else if (shape.meta.x == ShapeCapsule)
            found = query_capsule(local_origin, local_direction, shape.geometry.y, shape.geometry.w,
                                  request.positionMaximum.w, distance, normal);
        else
        {
            status = QueryUnsupported;
            return hit;
        }
        if (found)
        {
            hit.meta = uint4(shape.meta.y, 1u, leaf, 0u);
            hit.positionDistance = float4(request.positionMaximum.xyz + direction * distance, distance);
            hit.normalFraction = float4(jph_rotate(rotation, normal), distance / request.positionMaximum.w);
            hit.pointADepth = float4(hit.positionDistance.xyz, 0.0f);
            if (!all(isfinite(hit.positionDistance)) || !all(isfinite(hit.normalFraction)))
                status = QueryNumericalFailure;
        }
        return hit;
    }
    uint query_type = request.meta.x == QueryClosestPoint ? ShapeSphere : request.meta.z;
    float4 query_geometry = request.meta.x == QueryClosestPoint ? float4(0.0f) : request.geometry;
    JphPairMetric metric = jph_pair_metric(query_type, request.positionMaximum.xyz, request.rotation, query_geometry,
                                          shape.meta.x, center, rotation, shape.geometry);
    status = query_metric_status(metric);
    if (status != QuerySuccess)
        return hit;
    if (request.meta.x == QueryOverlap && metric.separation > 0.0f)
        return hit;
    hit.meta = uint4(shape.meta.y, 1u, leaf, metric.feature);
    float3 point = metric.pointB;
    if (request.meta.x == QueryClosestPoint && metric.separation <= 0.0f)
        point = request.positionMaximum.xyz;
    hit.positionDistance = float4(point, max(metric.separation, 0.0f));
    hit.normalFraction = float4(-metric.normal, 0.0f);
    hit.pointADepth = float4(metric.pointA, max(-metric.separation, 0.0f));
    return hit;
}

bool query_less(QueryInput request, QueryHit a, QueryHit b)
{
    if (request.meta.y == QueryUniqueCollider)
    {
        uint2 ca = shape_models[a.meta.z].source.xy;
        uint2 cb = shape_models[b.meta.z].source.xy;
        return ca.y != cb.y ? ca.y < cb.y : ca.x < cb.x;
    }
    if (request.meta.y == QueryUniqueBody)
    {
        uint2 a_id = query_body_ids[a.meta.x].ids.zw;
        uint2 b_id = query_body_ids[b.meta.x].ids.zw;
        if (a_id.y != b_id.y || a_id.x != b_id.x)
            return a_id.y != b_id.y ? a_id.y < b_id.y : a_id.x < b_id.x;
    }
    if (request.meta.y != QueryUniqueBody && a.positionDistance.w != b.positionDistance.w)
        return a.positionDistance.w < b.positionDistance.w;
    uint bodyA = query_body_ids[a.meta.x].ids.x;
    uint bodyB = query_body_ids[b.meta.x].ids.x;
    if (bodyA != bodyB)
        return bodyA < bodyB;
    uint subA = shape_models[a.meta.z].meta.z;
    uint subB = shape_models[b.meta.z].meta.z;
    return subA != subB ? subA < subB : a.meta.z < b.meta.z;
}

void query_collect(QueryInput request, QueryHit hit, uint begin, inout uint count)
{
    if (hit.meta.y == 0u)
        return;
    if (request.meta.y == QueryClosest)
    {
        QueryHit previous = query_candidates[begin];
        // Closest rays preserve the existing strict-distance slot/leaf tie.
        if (previous.meta.y == 0u || hit.positionDistance.w < previous.positionDistance.w)
            query_candidates[begin] = hit;
        return;
    }
    if (request.meta.y == QueryUniqueCollider)
        for (uint old = 0u; old < count; ++old)
            if (all(shape_models[query_candidates[begin + old].meta.z].source.xy == shape_models[hit.meta.z].source.xy))
                return;
    if (request.meta.y == QueryUniqueBody)
        for (uint old = 0u; old < count; ++old)
            if (query_candidates[begin + old].meta.x == hit.meta.x)
                return;
    uint position = count;
    while (position > 0u && query_less(request, hit, query_candidates[begin + position - 1u]))
    {
        query_candidates[begin + position] = query_candidates[begin + position - 1u];
        --position;
    }
    query_candidates[begin + position] = hit;
    ++count;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void query_evaluate(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= query_params.counts.x)
        return;
    QueryInput request = query_inputs[index];
    uint begin = index * query_params.sizes.y;
    for (uint row = 0u; row < query_params.sizes.y; ++row)
        query_candidates[begin + row] = query_miss();
    uint count = request.meta.y == QueryClosest ? 1u : 0u;
    uint status = QuerySuccess;
    if (request.meta.x == QueryPenetration)
    {
        QueryPrimitive other = query_primitives[request.source.y];
        JphPairMetric metric = jph_pair_metric(request.meta.z, request.positionMaximum.xyz, request.rotation, request.geometry,
                                              other.meta.x, other.position.xyz, other.rotation, other.geometry);
        status = query_metric_status(metric);
        if (status == QuerySuccess && metric.separation < 0.0f)
        {
            QueryHit hit = query_miss();
            hit.meta.yw = uint2(1u, metric.feature);
            hit.positionDistance = float4(metric.pointB, 0.0f);
            hit.normalFraction = float4(-metric.normal, 0.0f);
            hit.pointADepth = float4(metric.pointA, -metric.separation);
            query_candidates[begin] = hit;
            count = 1u;
        }
    }
    else
    {
        for (uint local = 0u; local < query_params.counts.z && status == QuerySuccess; ++local)
        {
            uint slot = query_params.counts.y + local;
            BodyModel model = body_models[slot];
            if (!jph_resident(model) || model.meta.y == 0u)
                continue;
            if (request.meta.x == QueryBodyBounds && query_body_ids[slot].ids.y == 0u)
                continue;
            BodyState state = body_states[slot];
            bool accepted = false;
            float3 low = float3(3.402823466e+38f);
            float3 high = -low;
            uint first = model.meta.x;
            for (uint ordinal = 0u; ordinal < model.meta.y; ++ordinal)
            {
                uint leaf = model.meta.x + ordinal;
                ShapeModel shape = shape_models[leaf];
                if (request.meta.x == QueryBodyBounds)
                {
                    float3 a, b;
                    query_leaf_bounds(shape, state, a, b);
                    if (!all(isfinite(a)) || !all(isfinite(b)))
                    {
                        status = QueryNumericalFailure;
                        break;
                    }
                    low = min(low, a);
                    high = max(high, b);
                    accepted = accepted || query_accepts(request, shape, slot);
                    continue;
                }
                if (!query_accepts(request, shape, slot))
                    continue;
                uint leaf_status;
                QueryHit hit = query_leaf(request, shape, state, leaf, leaf_status);
                if (leaf_status != QuerySuccess)
                {
                    status = leaf_status;
                    break;
                }
                query_collect(request, hit, begin, count);
                if (request.meta.y == QueryAny && count != 0u)
                    break;
            }
            if (request.meta.x == QueryBodyBounds && accepted &&
                all(low <= request.positionMaximum.xyz + request.direction.xyz) &&
                all(high >= request.positionMaximum.xyz - request.direction.xyz))
            {
                QueryHit hit = query_miss();
                hit.meta = uint4(slot, 1u, first, 0u);
                query_collect(request, hit, begin, count);
            }
            if (request.meta.y == QueryAny && count != 0u)
                break;
        }
    }
    QuerySpan span;
    span.range = uint4(0u, count, status, 0u);
    query_spans[index] = span;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void query_scan(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u)
        return;
    uint count = 0u;
    uint status = QuerySuccess;
    for (uint index = 0u; index < query_params.counts.x; ++index)
    {
        QuerySpan span = query_spans[index];
        span.range.x = count;
        count += span.range.y;
        status = max(status, span.range.z);
        query_spans[index] = span;
    }
    QuerySummary summary;
    summary.counts = uint4(count, status, query_params.counts.x, 0u);
    query_summary[0] = summary;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void query_compact(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= query_params.sizes.z)
        return;
    QueryHit hit = query_miss();
    if (index < query_summary[0].counts.x)
    {
        uint low = 0u;
        uint high = query_params.counts.x;
        while (low < high)
        {
            uint middle = low + (high - low) / 2u;
            QuerySpan span = query_spans[middle];
            if (span.range.x + span.range.y <= index)
                low = middle + 1u;
            else
                high = middle;
        }
        QuerySpan source = query_spans[low];
        hit = query_candidates[low * query_params.sizes.y + index - source.range.x];
    }
    // One writer per final index, including the unused tail. A sparse scatter
    // plus concurrent clearing would race and would not prove initialization.
    query_hits[index] = hit;
}
