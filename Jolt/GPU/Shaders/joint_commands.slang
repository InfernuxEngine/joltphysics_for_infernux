// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_joints.slang"

ConstantBuffer<JointParams> joint_params;
StructuredBuffer<uint> joint_remap;
StructuredBuffer<QueryRecordWord> old_joint_stream;
StructuredBuffer<BodyState> body_states;
StructuredBuffer<uint> joint_query_rows;
RWStructuredBuffer<QueryRecordWord> joint_motion;

[shader("compute")]
[numthreads(64, 1, 1)]
void update_joint_state(uint3 tid : SV_DispatchThreadID)
{
    uint request = tid.x;
    if (request >= joint_params.counts.z) return;
    uint index = joint_query_rows[2u * request];
    uint flags = joint_query_rows[2u * request + 1u];
    JointAuthor a = jph_joint_load_author(index);
    JointState state = jph_joint_load_state(index);
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess)
    {
        jph_joint_store_state(index, state);
        return;
    }
    if (a.bodies.z == JointSlider || a.bodies.z == JointDistance)
    {
        jph_joint_clear_impulses(state);
        state.rotor.xyz = float3(0.0f);
        jph_joint_store_state(index, state);
        return;
    }
    // Reset at the ordered author boundary, before a later body command.
    // Deferring this until setup would collapse two distinct setter orders.
    float old_armature = asfloat(state.flags.z);
    float speed = old_armature > 0.0f ? state.rotor.x - state.rotor.y / old_armature : state.rotor.x;
    if ((flags & JointResetRotor) != 0u)
    {
        BodyState s1 = jph_joint_world_state();
        if (a.bodies.x != InvalidQueryBodySlot) s1 = body_states[a.bodies.x];
        BodyState s2 = body_states[a.bodies.y];
        speed = dot(jph_joint_axis(s1, state), s2.angularVelocity.xyz - s1.angularVelocity.xyz);
    }
    jph_joint_clear_impulses(state);
    state.rotor.xyz = float3(speed, 0.0f, speed);
    state.flags.z = asuint(a.passive.z);
    if (!isfinite(speed)) state.flags.y = JointStatusNumericalFailure;
    jph_joint_store_state(index, state);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize_joints(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= joint_params.counts.x) return;
    uint old_index = joint_remap[index];
    uint destination = joint_stream[0].value.y + JointStateWords * index;
    if (old_index != InvalidQueryBodySlot && old_index < joint_params.counts.y)
    {
        uint source = old_joint_stream[0].value.y + JointStateWords * old_index;
        // Copy GPU-owned state directly. Both resources are immutable to the
        // CPU in these ranges; accepted topology decides the identity remap.
        for (uint word = 0u; word < JointStateWords; ++word)
            joint_stream[destination + word] = old_joint_stream[source + word];
        return;
    }
    JointAuthor a = jph_joint_load_author(index);
    BodyState s1 = jph_joint_world_state();
    if (a.bodies.x != InvalidQueryBodySlot) s1 = body_states[a.bodies.x];
    JointState state = jph_joint_initialize(a, s1, body_states[a.bodies.y]);
    if (old_index != InvalidQueryBodySlot)
        state.flags.y = JointStatusInvalid;
    jph_joint_store_state(index, state);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void read_joint_motion(uint3 tid : SV_DispatchThreadID)
{
    uint row = tid.x;
    if (row >= joint_params.counts.z) return;
    QueryRecordWord result;
    result.value = uint4(0u, 0u, JointStatusInvalid, 0u);
    uint index = joint_query_rows[row];
    if (index < joint_stream[0].value.x)
    {
        JointAuthor a = jph_joint_load_author(index);
        JointState state = jph_joint_load_state(index);
        if ((a.bodies.z == JointHinge || a.bodies.z == JointSlider || a.bodies.z == JointDistance) && state.flags.x != 0u)
        {
            result.value.z = state.flags.y;
            if (state.flags.y == JointStatusSuccess)
            {
                BodyState s1 = jph_joint_world_state();
                if (a.bodies.x != InvalidQueryBodySlot) s1 = body_states[a.bodies.x - joint_params.counts.w];
                BodyState s2 = body_states[a.bodies.y - joint_params.counts.w];
                float coordinate, velocity;
                if (a.bodies.z == JointDistance)
                {
                    float3 r1, r2;
                    jph_joint_arms(a, state, s1, s2, r1, r2);
                    float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
                    coordinate = length(u);
                    float3 normal = coordinate > 0.0f ? u / coordinate : state.distanceReference.xyz;
                    velocity = dot(normal, s2.linearVelocity.xyz - s1.linearVelocity.xyz)
                        + dot(cross(r2, normal), s2.angularVelocity.xyz)
                        - dot(cross(r1 + u, normal), s1.angularVelocity.xyz);
                }
                else if (a.bodies.z == JointSlider)
                {
                    float3 r1, r2;
                    jph_joint_arms(a, state, s1, s2, r1, r2);
                    float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
                    float3 axis, n0, n1;
                    jph_slider_basis(s1, state, axis, n0, n1);
                    coordinate = dot(axis, u);
                    velocity = dot(axis, s2.linearVelocity.xyz - s1.linearVelocity.xyz)
                        + dot(cross(r2, axis), s2.angularVelocity.xyz)
                        - dot(cross(r1 + u, axis), s1.angularVelocity.xyz);
                }
                else
                {
                    coordinate = jph_joint_angle(s1, s2, state);
                    float3 owner_axis = jph_rotate(jph_quat_mul(s2.rotation, state.frame2), float3(0.0f, 0.0f, 1.0f));
                    velocity = dot(owner_axis, s2.angularVelocity.xyz - s1.angularVelocity.xyz);
                }
                if (isfinite(coordinate) && isfinite(velocity))
                    result.value.xy = uint2(asuint(coordinate), asuint(velocity));
                else
                    result.value.z = JointStatusNumericalFailure;
            }
        }
    }
    joint_motion[row] = result;
}

groupshared uint joint_status_values[64];
groupshared uint joint_status_failures[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void read_joint_status(uint3 tid : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    uint status = JointStatusSuccess;
    uint first_failure = InvalidQueryBodySlot;
    // Status work: source first/count, destination first, reduction level.
    // Every lane reaches the barriers, including the last group's padding.
    if (tid.x < joint_params.counts.y)
    {
        uint index = joint_params.counts.x + tid.x;
        if (joint_params.counts.w == 0u)
        {
            uint runtime_word = joint_stream[0].value.y + JointStateWords * index;
            status = joint_stream[runtime_word].value.y;
            if (status != JointStatusSuccess)
                first_failure = index;
        }
        else
        {
            uint4 previous = joint_motion[index].value;
            status = previous.x;
            first_failure = previous.y;
        }
    }
    joint_status_values[lane.x] = status;
    joint_status_failures[lane.x] = first_failure;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32u; stride != 0u; stride >>= 1u)
    {
        if (lane.x < stride)
        {
            joint_status_values[lane.x] = max(joint_status_values[lane.x], joint_status_values[lane.x + stride]);
            joint_status_failures[lane.x] = min(joint_status_failures[lane.x], joint_status_failures[lane.x + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane.x == 0u)
    {
        QueryRecordWord result;
        result.value = uint4(joint_status_values[0], joint_status_failures[0], 0u, 0u);
        joint_motion[joint_params.counts.z + group.x] = result;
    }
}
