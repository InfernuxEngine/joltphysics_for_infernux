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
RWStructuredBuffer<KinematicTarget> kinematic_targets;

float3 command_world_inertia(BodyModel model, BodyState state, float3 value)
{
    float4 frame = jph_quat_mul(state.rotation, model.inertiaRotation);
    return jph_rotate(frame, jph_inverse_rotate(frame, value) / model.invInertiaDiagonal.xyz);
}

void command_cancel_move(inout BodyState state, inout KinematicTarget target)
{
    if (jph_has_kinematic_move(target))
    {
        state.linearVelocity = float4(0.0f);
        state.angularVelocity = float4(0.0f);
    }
    target.state = uint4(0u);
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
    KinematicTarget target = kinematic_targets[body];
    // The validated immutable plan bounds every span to 64. Each dispatch has
    // at most one span for any body, so no atomics or global command scan occur.
    for (uint offset = 0; offset < span.range.z; ++offset)
    {
        BodyCommand command = body_commands[span.range.y + offset];
        uint kind = command.meta.y;
        uint mode = command.meta.z;
        if (kind <= CommandSetPose)
        {
            if ((mode & CommandPreserveKinematicMove) == 0u)
                command_cancel_move(state, target);
            if (kind == CommandSetPosition)
                state.positionCOM.xyz = command.value.xyz + jph_rotate(state.rotation, command.rootCOM.xyz);
            else if (kind == CommandSetRotation)
            {
                float3 origin = state.positionCOM.xyz - jph_rotate(state.rotation, command.rootCOM.xyz);
                state.rotation = command.auxiliary;
                state.positionCOM.xyz = origin + jph_rotate(state.rotation, command.rootCOM.xyz);
            }
            else
            {
                state.rotation = command.auxiliary;
                state.positionCOM.xyz = command.value.xyz + jph_rotate(state.rotation, command.rootCOM.xyz);
            }
        }
        else if (kind == CommandResetMotion)
        {
            state.linearVelocity = float4(0.0f);
            state.angularVelocity = float4(0.0f);
            forces.force = float4(0.0f);
            forces.torque = float4(0.0f);
            target.state = uint4(0u);
        }
        else if (kind >= CommandMovePosition && kind <= CommandMovePose)
        {
            uint motion = model.meta.z & ~MotionNotResident;
            bool temporary_static = motion == MotionStatic && (mode & CommandTemporaryStatic) != 0u;
            if (motion == MotionKinematic || temporary_static)
            {
                // Only the first pending partial move samples the actual
                // device pose. Later partial moves compose that queued target.
                if ((target.state.x & TargetPending) == 0u)
                {
                    float3 origin = state.positionCOM.xyz - jph_rotate(state.rotation, command.rootCOM.xyz);
                    target.positionSpeed = float4(origin, 0.0f);
                    target.rotation = state.rotation;
                }
                if (kind != CommandMoveRotation)
                    target.positionSpeed.xyz = command.value.xyz;
                if (kind != CommandMovePosition)
                    target.rotation = command.auxiliary;
                target.positionSpeed.w = command.rootCOM.w;
                target.rootCOMDt = float4(command.rootCOM.xyz, command.value.w);
                target.state = uint4(TargetPending | (temporary_static ? TargetTemporaryStatic : 0u), 0u, 0u, 0u);
            }
        }
        else if (kind == CommandMotionBoundary)
        {
            uint action = mode & MotionBoundaryActionMask;
            bool had_move = jph_has_kinematic_move(target);
            if (action == MotionBoundaryTypeChange)
            {
                // Carry protected author channels from the device target,
                // never from a newer CPU pose. COM mutation rebased rootCOM.
                uint carry = mode & (MotionBoundaryCarryAuthoredPosition | MotionBoundaryCarryAuthoredRotation);
                if (had_move && carry != 0u)
                {
                    float3 origin = state.positionCOM.xyz - jph_rotate(state.rotation, target.rootCOMDt.xyz);
                    if ((carry & MotionBoundaryCarryAuthoredPosition) != 0u)
                        origin = target.positionSpeed.xyz;
                    if ((carry & MotionBoundaryCarryAuthoredRotation) != 0u)
                        state.rotation = target.rotation;
                    state.positionCOM.xyz = origin + jph_rotate(state.rotation, target.rootCOMDt.xyz);
                }
                uint motion = model.meta.z & ~MotionNotResident;
                if (motion == MotionStatic)
                {
                    state.linearVelocity = float4(0.0f);
                    state.angularVelocity = float4(0.0f);
                }
                if (motion != MotionDynamic)
                {
                    forces.force = float4(0.0f);
                    forces.torque = float4(0.0f);
                }
            }
            else if (action == MotionBoundarySuspendReset || had_move)
            {
                // Suspension stops target-owned motion. Explicit manual
                // velocity survives preserve mode; force accumulation survives
                // either suspension mode, matching ordinary body removal.
                state.linearVelocity = float4(0.0f);
                state.angularVelocity = float4(0.0f);
            }
            target.state = uint4(0u);
        }
        else if (kind == CommandSetLinearVelocity || kind == CommandSetAngularVelocity)
        {
            if (jph_nonstatic(model))
            {
                // Explicit velocity supersedes the whole arrival stop while
                // preserving the other channel's actual current velocity.
                target.state = uint4(0u);
                if (kind == CommandSetLinearVelocity)
                    state.linearVelocity.xyz = jph_clamp_speed(command.value.xyz, model.damping.z);
                else
                    state.angularVelocity.xyz = jph_clamp_speed(command.value.xyz, model.damping.w);
            }
        }
        else if (jph_dynamic(model))
        {
            if (kind == CommandAddTorque)
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
    // Every selected span writes its complete slot values, including when
    // individual kinds have no effect for the accepted author motion type.
    body_states[body] = state;
    body_forces[body] = forces;
    kinematic_targets[body] = target;
}
