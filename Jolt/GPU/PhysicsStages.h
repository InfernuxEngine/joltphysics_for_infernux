// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/GPU/PhysicsTypes.h>

namespace JPH { namespace GPU {

// A stage recipe only: the integrating engine owns compilation, resources,
// barriers, admission, submission, completion and publication.
enum class Resource : std::uint32_t
{
    StepParams = 1u << 0,
    BodyModels = 1u << 1,
    ShapeModels = 1u << 2,
    Worlds = 1u << 3,
    BodyForces = 1u << 4,
    BodyStates = 1u << 5,
    Bounds = 1u << 6,
    Contacts = 1u << 7,
};

constexpr std::uint32_t Mask(Resource inResource) { return static_cast<std::uint32_t>(inResource); }

enum class Stage : std::uint32_t
{
    ApplyForces,
    BuildBounds,
    CollidePairs,
    SetupContacts,
    SolveVelocity,
    Integrate,
    SolvePosition,
};

enum class Domain : std::uint32_t { Bodies, WorldPairs, Contacts, Worlds };

struct StageDefinition
{
    Stage stage;
    const char *entry; // Host metadata; never part of a device buffer.
    Domain domain;
    std::uint32_t groupSizeX, groupSizeY, groupSizeZ;
    std::uint32_t reads, writes;
};

inline constexpr const char *StepModuleName = "jph_gpu_rigid_step";
inline constexpr const char *StepSourcePath = "Jolt/GPU/Shaders/rigid_step.slang";

inline constexpr StageDefinition Stages[] = {
    {Stage::ApplyForces, "apply_forces", Domain::Bodies, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyForces) | Mask(Resource::BodyStates),
        Mask(Resource::BodyStates)},
    {Stage::BuildBounds, "build_bounds", Domain::Bodies, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::ShapeModels) | Mask(Resource::BodyStates),
        Mask(Resource::Bounds)},
    {Stage::CollidePairs, "collide_pairs", Domain::WorldPairs, 8, 8, 1,
        Mask(Resource::StepParams) | Mask(Resource::Worlds) | Mask(Resource::BodyModels) | Mask(Resource::ShapeModels)
            | Mask(Resource::BodyStates) | Mask(Resource::Bounds), Mask(Resource::Contacts)},
    {Stage::SetupContacts, "setup_contacts", Domain::Contacts, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyForces) | Mask(Resource::BodyStates)
            | Mask(Resource::Contacts), Mask(Resource::Contacts)},
    {Stage::SolveVelocity, "solve_velocity", Domain::Worlds, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::Worlds) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates)
            | Mask(Resource::Contacts), Mask(Resource::BodyStates) | Mask(Resource::Contacts)},
    {Stage::Integrate, "integrate", Domain::Bodies, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates), Mask(Resource::BodyStates)},
    {Stage::SolvePosition, "solve_position", Domain::Worlds, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::Worlds) | Mask(Resource::BodyModels) | Mask(Resource::Contacts)
            | Mask(Resource::BodyStates), Mask(Resource::BodyStates)},
};

// One substep: ApplyForces, BuildBounds, CollidePairs, SetupContacts,
// SolveVelocity repeated counts.w times, Integrate, SolvePosition repeated
// solve.x times. Each solve dispatch performs ONE iteration. No warmstart,
// sleeping, CCD, dynamic boxes, multiple-primitive bodies, filters, joints, locked DOFs,
// gyroscopic forces or kinematic motion are implemented by this bootstrap. A
// host must reject these unsupported options. Each body has one primitive leaf;
// its local pose may differ from the body's COM frame, including an explicit
// center-of-mass offset. Authoring containers with one leaf may be flattened;
// multiple-leaf compound collision and mass/response are not implemented here.
// It must also validate finite inputs, normalized body/shape/inertia rotations,
// zero shape local-position padding, positive dimensions,
// nonnegative material/damping/speed bounds, convex radius <= min half extent,
// disjoint complete world ranges, matching body.meta.w and at most 32 bodies per
// world. This is a bounded, serial-per-world correctness baseline, not a parallel
// contact solver or a published performance result.

} } // namespace JPH::GPU
