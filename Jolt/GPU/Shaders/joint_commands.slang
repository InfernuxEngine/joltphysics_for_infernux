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
    uint destination = joint_stream[0].value.y + JointRecordWords * index;
    if (old_index != InvalidQueryBodySlot && old_index < joint_params.counts.y)
    {
        uint source = old_joint_stream[0].value.y + JointRecordWords * old_index;
        // Copy GPU-owned state directly. Both resources are immutable to the
        // CPU in these ranges; accepted topology decides the identity remap.
        for (uint word = 0u; word < JointRecordWords; ++word)
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
        if (a.bodies.z == JointHinge && state.flags.x != 0u)
        {
            result.value.z = state.flags.y;
            if (state.flags.y == JointStatusSuccess)
            {
                BodyState s1 = jph_joint_world_state();
                if (a.bodies.x != InvalidQueryBodySlot) s1 = body_states[a.bodies.x];
                BodyState s2 = body_states[a.bodies.y];
                float angle = jph_joint_angle(s1, s2, state);
                float3 owner_axis = jph_rotate(jph_quat_mul(s2.rotation, state.frame2), float3(0.0f, 0.0f, 1.0f));
                float velocity = dot(owner_axis, s2.angularVelocity.xyz - s1.angularVelocity.xyz);
                if (isfinite(angle) && isfinite(velocity))
                    result.value.xy = uint2(asuint(angle), asuint(velocity));
                else
                    result.value.z = JointStatusNumericalFailure;
            }
        }
    }
    joint_motion[row] = result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void read_joint_status(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint status = JointStatusSuccess;
    uint first_failure = InvalidQueryBodySlot;
    for (uint index = 0u; index < joint_params.counts.x; ++index)
    {
        uint runtime_word = joint_stream[0].value.y + JointRecordWords * index;
        uint current = joint_stream[runtime_word].value.y;
        if (current != JointStatusSuccess && first_failure == InvalidQueryBodySlot)
            first_failure = index;
        status = max(status, current);
    }
    // Publication consumes this small metadata record before accepting the
    // logical step. Local NaN guards do not turn a partial solve into success.
    QueryRecordWord result;
    result.value = uint4(status, first_failure, 0u, 0u);
    joint_motion[0] = result;
}
