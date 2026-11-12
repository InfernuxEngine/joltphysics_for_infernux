// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"

ConstantBuffer<BodyMutationParams> mutation_params;
StructuredBuffer<BodyInitialization> body_initializations;
StructuredBuffer<CenterOfMassUpdate> center_of_mass_updates;
RWStructuredBuffer<BodyState> body_states;
RWStructuredBuffer<BodyForces> body_forces;

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize_bodies(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= mutation_params.counts.x)
        return;
    BodyInitialization initial = body_initializations[mutation_params.counts.z + tid.x];
    body_states[initial.bodyIndex.x] = initial.state;
    body_forces[initial.bodyIndex.x] = initial.forces;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_centers_of_mass(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= mutation_params.counts.y)
        return;
    CenterOfMassUpdate change = center_of_mass_updates[tid.x];
    BodyState state = body_states[change.bodyIndex.x];
    // Body::UpdateCenterOfMassInternal preserves the world body origin and
    // does not modify velocities, accumulated forces or the body rotation.
    state.positionCOM.xyz += jph_rotate(state.rotation, change.deltaLocalCOM.xyz);
    body_states[change.bodyIndex.x] = state;
}
