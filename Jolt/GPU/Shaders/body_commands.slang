// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"

ConstantBuffer<BodyCommandParams> command_params;
StructuredBuffer<BodyCommand> body_commands;
StructuredBuffer<BodyCommandSpan> command_spans;
StructuredBuffer<BodyModel> body_models;
RWStructuredBuffer<BodyState> body_states;
RWStructuredBuffer<BodyForces> body_forces;

float3 command_world_inertia(BodyModel model, BodyState state, float3 value)
{
    float4 frame = jph_quat_mul(state.rotation, model.inertiaRotation);
    return jph_rotate(frame, jph_inverse_rotate(frame, value) / model.invInertiaDiagonal.xyz);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_body_commands(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= command_params.spans.y)
        return;
    BodyCommandSpan span = command_spans[command_params.spans.x + tid.x];
    uint body = span.range.x;
    BodyModel model = body_models[body];
    BodyState state = body_states[body];
    BodyForces forces = body_forces[body];
    // The validated immutable plan bounds every span to 64. Each dispatch has
    // at most one span for any body, so no atomics or global command scan occur.
    for (uint offset = 0; offset < span.range.z; ++offset)
    {
        BodyCommand command = body_commands[span.range.y + offset];
        uint kind = command.meta.y;
        uint mode = command.meta.z;
        if (kind == CommandSetPosition)
            state.positionCOM.xyz = command.value.xyz + jph_rotate(state.rotation, command.rootCOM.xyz);
        else if (kind == CommandSetRotation)
        {
            float3 origin = state.positionCOM.xyz - jph_rotate(state.rotation, command.rootCOM.xyz);
            state.rotation = command.auxiliary;
            state.positionCOM.xyz = origin + jph_rotate(state.rotation, command.rootCOM.xyz);
        }
        else if (kind == CommandSetPose)
        {
            state.rotation = command.auxiliary;
            state.positionCOM.xyz = command.value.xyz + jph_rotate(state.rotation, command.rootCOM.xyz);
        }
        else if (kind == CommandResetMotion)
        {
            state.linearVelocity = float4(0.0f);
            state.angularVelocity = float4(0.0f);
            forces.force = float4(0.0f);
            forces.torque = float4(0.0f);
        }
        else if (jph_dynamic(model))
        {
            if (kind == CommandSetLinearVelocity)
                state.linearVelocity.xyz = jph_clamp_speed(command.value.xyz, model.damping.z);
            else if (kind == CommandSetAngularVelocity)
                state.angularVelocity.xyz = jph_clamp_speed(command.value.xyz, model.damping.w);
            else if (kind == CommandAddTorque)
            {
                float3 torque = command.value.xyz;
                // Acceleration converts to a world torque NOW. A later pose
                // command must not change the inertia used for this conversion.
                if (mode == CommandAcceleration || mode == CommandVelocityChange)
                    torque = command_world_inertia(model, state, torque);
                if (mode == CommandForce || mode == CommandAcceleration)
                    forces.torque.xyz += torque;
                else
                    state.angularVelocity.xyz = jph_clamp_speed(
                        state.angularVelocity.xyz + jph_inverse_inertia(model, state, torque), model.damping.w);
            }
            else
            {
                float3 force = command.value.xyz;
                if (mode == CommandAcceleration || mode == CommandVelocityChange)
                    force /= model.mass.x;
                float3 torque = kind == CommandAddForceAtPosition
                    ? cross(command.auxiliary.xyz - state.positionCOM.xyz, force) : float3(0.0f);
                if (mode == CommandForce || mode == CommandAcceleration)
                {
                    forces.force.xyz += force;
                    forces.torque.xyz += torque;
                }
                else
                {
                    state.linearVelocity.xyz = jph_clamp_speed(
                        state.linearVelocity.xyz + model.mass.x * force, model.damping.z);
                    if (kind == CommandAddForceAtPosition)
                        state.angularVelocity.xyz = jph_clamp_speed(
                            state.angularVelocity.xyz + jph_inverse_inertia(model, state, torque), model.damping.w);
                }
            }
        }
    }
    body_states[body] = state;
    body_forces[body] = forces;
}
