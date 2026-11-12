// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"

ConstantBuffer<PoseParams> pose_params;
StructuredBuffer<PoseBinding> pose_bindings;
StructuredBuffer<BodyState> body_states;
// Four float4 columns per renderer, matching column-major InxModelMatrix.
RWStructuredBuffer<float4> pose_models;
// Two float4 records per renderer: final world-space mesh AABB min and max.
RWStructuredBuffer<float4> pose_bounds;

// The host validates finite affine bindings, nonnegative local extents, valid
// body indices and normalized xyzw quaternions before accepting this work.
// Output capacities cover 4N / 2N records; N <= UINT_MAX / 4 for uint indexing.
// The BodyState input is the frozen final state after the required solve stages,
// not an intermediate substep. Render-frame motion history belongs to Renderer.
[shader("compute")]
[numthreads(64, 1, 1)]
void export_render_poses(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= pose_params.bindingCount.x)
        return;

    PoseBinding binding = pose_bindings[index];
    BodyState state = body_states[binding.bodyIndex.x];
    float4 q = state.rotation;

    // M = Translate(world COM) * Rotate(body-to-world) * MeshLocalToCOM.
    // Work in explicit columns to preserve the renderer's storage convention.
    float3 column0 = jph_rotate(q, binding.localToCOM0.xyz);
    float3 column1 = jph_rotate(q, binding.localToCOM1.xyz);
    float3 column2 = jph_rotate(q, binding.localToCOM2.xyz);
    float3 column3 = state.positionCOM.xyz + jph_rotate(q, binding.localToCOM3.xyz);
    uint model_begin = 4u * index;
    pose_models[model_begin] = float4(column0, 0.0f);
    pose_models[model_begin + 1u] = float4(column1, 0.0f);
    pose_models[model_begin + 2u] = float4(column2, 0.0f);
    pose_models[model_begin + 3u] = float4(column3, 1.0f);

    // Transform the mesh envelope, independent of the speculative collision AABB.
    float3 center = binding.boundsCenter.xyz;
    float3 extent = binding.boundsExtent.xyz;
    float3 world_center = column3 + column0 * center.x + column1 * center.y + column2 * center.z;
    float3 world_extent = abs(column0) * extent.x + abs(column1) * extent.y + abs(column2) * extent.z;
    uint bounds_begin = 2u * index;
    pose_bounds[bounds_begin] = float4(world_center - world_extent, 0.0f);
    pose_bounds[bounds_begin + 1u] = float4(world_center + world_extent, 0.0f);
}
