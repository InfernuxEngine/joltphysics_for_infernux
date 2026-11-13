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
    KinematicTargets = 1u << 8,
    ContactPatches = 1u << 9,
    Joints = 1u << 10,
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
    FinishTick,
    PrepareKinematics,
    SetupJoints,
    FinishJoints,
};

enum class Domain : std::uint32_t { Bodies, Shapes, WorldPairs, Contacts, Worlds };

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
    {Stage::BuildBounds, "build_bounds", Domain::Shapes, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::ShapeModels) | Mask(Resource::BodyStates),
        Mask(Resource::Bounds)},
    {Stage::CollidePairs, "collide_pairs", Domain::WorldPairs, 8, 8, 1,
        Mask(Resource::StepParams) | Mask(Resource::Worlds) | Mask(Resource::BodyModels) | Mask(Resource::ShapeModels)
            | Mask(Resource::BodyStates) | Mask(Resource::Bounds) | Mask(Resource::Joints),
        Mask(Resource::Contacts) | Mask(Resource::ContactPatches)},
    {Stage::SetupContacts, "setup_contacts", Domain::Contacts, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyForces) | Mask(Resource::BodyStates)
            | Mask(Resource::Contacts) | Mask(Resource::KinematicTargets) | Mask(Resource::ShapeModels)
            | Mask(Resource::ContactPatches), Mask(Resource::Contacts)},
    {Stage::SolveVelocity, "solve_velocity", Domain::Worlds, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::Worlds) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates)
            | Mask(Resource::Contacts) | Mask(Resource::KinematicTargets) | Mask(Resource::ShapeModels)
            | Mask(Resource::ContactPatches) | Mask(Resource::Joints),
        Mask(Resource::BodyStates) | Mask(Resource::Contacts) | Mask(Resource::Joints)},
    {Stage::Integrate, "integrate", Domain::Bodies, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates)
            | Mask(Resource::KinematicTargets), Mask(Resource::BodyStates)},
    {Stage::SolvePosition, "solve_position", Domain::Worlds, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::Worlds) | Mask(Resource::BodyModels) | Mask(Resource::Contacts)
            | Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets) | Mask(Resource::Joints),
        Mask(Resource::BodyStates) | Mask(Resource::Joints)},
    {Stage::FinishTick, "finish_tick", Domain::Bodies, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyForces)
            | Mask(Resource::KinematicTargets), Mask(Resource::BodyForces) | Mask(Resource::KinematicTargets)},
    {Stage::PrepareKinematics, "prepare_kinematics", Domain::Bodies, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates)
            | Mask(Resource::KinematicTargets), Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets)},
    {Stage::SetupJoints, "setup_joints", Domain::Worlds, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::Worlds) | Mask(Resource::BodyModels)
            | Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets) | Mask(Resource::Joints),
        Mask(Resource::BodyStates) | Mask(Resource::Joints)},
    {Stage::FinishJoints, "finish_joints", Domain::Worlds, 64, 1, 1,
        Mask(Resource::StepParams) | Mask(Resource::Worlds) | Mask(Resource::Joints), Mask(Resource::Joints)},
};

// PrepareKinematics runs once before all substeps. One substep consists of
// ApplyForces, BuildBounds, CollidePairs, SetupContacts, SetupJoints,
// SolveVelocity repeated counts.w times, Integrate, SolvePosition repeated
// solve.x times, FinishJoints. Joint-only worlds run both solvers even without
// shapes. Each solve dispatch performs ONE iteration. The logical-tick
// owner appends FinishTick once after all substeps, clearing forces and arming
// the target arrival stop without snapping pose. Control-only jobs use neither
// PrepareKinematics nor FinishTick. Contact warmstart, sleeping, CCD, general
// pair-ignore filters, event delivery and locked DOFs are not implemented by this
// correctness baseline; a host must reject unsupported capabilities. Bodies
// address zero or more Sphere/Box leaves in their COM frame, including explicit
// COM offsets. Every leaf pair owns an independent four-point patch; sensors
// record real overlap but do not participate in response. Dense leaf-pair
// collection and serial-per-world solving are not final throughput algorithms.
// The host must validate finite inputs, normalized body/shape/inertia rotations,
// zero shape local-position padding, positive dimensions, layer indices < 32,
// material combine modes in [MaterialCombineAverage, MaterialCombineMaximum],
// nonnegative material/damping/speed bounds, convex radius <= min half extent,
// disjoint complete body/leaf world ranges with four point records per ordered
// leaf pair, each leaf's owning body and range, matching body.meta.w,
// patch/contact counts within device indexing, and at most 32 bodies per
// world. This is a bounded, serial-per-world correctness baseline, not a parallel
// contact solver or a published performance result.
// JointStream contains a header, twelve words per author, then twelve words
// per runtime record. World joint ranges are disjoint and complete; endpoints
// belong to that world or use InvalidQueryBodySlot for the world endpoint.
// Fixed/Hinge author parameters must be finite, axes unit length and
// stiffness/damping/torque/friction/armature nonnegative. SetupJoints applies
// warmstart once per substep, only with nonzero velocity iterations. Joints
// and contacts share each iteration's single world owner. Runtime stores never
// modify the header or author prefix. No entry uses over eight storage ports.

} } // namespace JPH::GPU
