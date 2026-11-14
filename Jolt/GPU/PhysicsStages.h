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
    StageWork = 1u << 11,
    ConstraintSchedule = 1u << 12,
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
    WarmJoints,
    SolveJointVelocity,
    SolveJointPosition,
};

enum class Domain : std::uint32_t
{
    Bodies, Shapes, Patches, Contacts, IslandPatches, IslandContacts, IslandJoints, Joints
};

// Total rectangular items per dispatch, independent of island/world capacity.
// A unit must satisfy laneCount * localCount <= this quantum, including padding.
// One lane alone may consume the entire quantum. Domain lanes are not capped
// by a world-domain lane limit. Host validation uses checked arithmetic.
constexpr std::uint32_t WorkItemLimit(Domain inDomain)
{
    switch (inDomain) {
    case Domain::IslandJoints: return 64;
    case Domain::IslandPatches: return 256;
    case Domain::IslandContacts: return 1024;
    default: return 2048;
    }
}

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
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels)
            | Mask(Resource::BodyForces) | Mask(Resource::BodyStates),
        Mask(Resource::BodyStates)},
    {Stage::BuildBounds, "build_bounds", Domain::Shapes, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ShapeModels) | Mask(Resource::BodyStates),
        Mask(Resource::Bounds)},
    {Stage::CollidePairs, "narrowphase_candidates", Domain::Patches, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ShapeModels)
            | Mask(Resource::BodyStates) | Mask(Resource::ContactPatches),
        Mask(Resource::Contacts) | Mask(Resource::ContactPatches)},
    {Stage::SetupContacts, "setup_contacts", Domain::Contacts, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels)
            | Mask(Resource::BodyForces) | Mask(Resource::BodyStates)
            | Mask(Resource::Contacts) | Mask(Resource::KinematicTargets) | Mask(Resource::ShapeModels)
            | Mask(Resource::ContactPatches), Mask(Resource::Contacts)},
    {Stage::SolveVelocity, "solve_velocity", Domain::IslandPatches, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ConstraintSchedule)
            | Mask(Resource::BodyModels) | Mask(Resource::BodyStates)
            | Mask(Resource::Contacts) | Mask(Resource::KinematicTargets) | Mask(Resource::ShapeModels)
            | Mask(Resource::ContactPatches),
        Mask(Resource::BodyStates) | Mask(Resource::Contacts)},
    {Stage::Integrate, "integrate", Domain::Bodies, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates)
            | Mask(Resource::KinematicTargets), Mask(Resource::BodyStates)},
    {Stage::SolvePosition, "solve_position", Domain::IslandContacts, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ConstraintSchedule)
            | Mask(Resource::BodyModels) | Mask(Resource::Contacts) | Mask(Resource::BodyStates),
        Mask(Resource::BodyStates)},
    {Stage::FinishTick, "finish_tick", Domain::Bodies, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyForces)
            | Mask(Resource::KinematicTargets), Mask(Resource::BodyForces) | Mask(Resource::KinematicTargets)},
    {Stage::PrepareKinematics, "prepare_kinematics", Domain::Bodies, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels) | Mask(Resource::BodyStates)
            | Mask(Resource::KinematicTargets), Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets)},
    {Stage::SetupJoints, "setup_joints", Domain::Joints, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::BodyModels)
            | Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets) | Mask(Resource::Joints),
        Mask(Resource::Joints)},
    {Stage::FinishJoints, "finish_joints", Domain::Joints, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::Joints), Mask(Resource::Joints)},
    {Stage::WarmJoints, "warm_joints", Domain::IslandJoints, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ConstraintSchedule) | Mask(Resource::BodyModels)
            | Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets) | Mask(Resource::Joints),
        Mask(Resource::BodyStates) | Mask(Resource::Joints)},
    {Stage::SolveJointVelocity, "solve_joint_velocity", Domain::IslandJoints, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ConstraintSchedule) | Mask(Resource::BodyModels)
            | Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets) | Mask(Resource::Joints),
        Mask(Resource::BodyStates) | Mask(Resource::Joints)},
    {Stage::SolveJointPosition, "solve_joint_position", Domain::IslandJoints, 64, 1, 1,
        Mask(Resource::StageWork) | Mask(Resource::StepParams) | Mask(Resource::ConstraintSchedule) | Mask(Resource::BodyModels)
            | Mask(Resource::BodyStates) | Mask(Resource::KinematicTargets) | Mask(Resource::Joints),
        Mask(Resource::BodyStates) | Mask(Resource::Joints)},
};

// PrepareKinematics runs once before all substeps. One substep consists of
// ApplyForces, BuildBounds, the sparse collector, CollidePairs candidate
// narrowphase, SetupContacts, SetupJoints, WarmJoints
// (only with nonzero velocity iterations), then SolveJointVelocity followed by
// SolveVelocity for each counts.w iteration, Integrate, SolveJointPosition then
// SolvePosition for each solve.x iteration, FinishJoints. StageWork bounds each
// dispatch's items/island lanes; all units of an iteration precede its successor.
// Joint-only worlds use joint stages without contact stages. The logical-tick
// owner appends FinishTick once after all substeps, clearing forces and arming
// the target arrival stop without snapping pose. Control-only jobs use neither
// PrepareKinematics nor FinishTick. Contact warmstart, sleeping, CCD, general
// pair-ignore filters, event delivery and locked DOFs are not implemented by this
// correctness baseline; a host must reject unsupported capabilities. Bodies
// address zero or more Sphere/Box/Capsule leaves in their COM frame, including explicit
// COM offsets. Each collected canonical leaf pair owns four Contact records;
// sensors record real overlap but do not participate in response. The collector
// supplies actual candidate counts and runtime world ranges before narrowphase.
// ContactPatches is a word arena: word zero holds the final candidate-bank base,
// and patch loads add that base to candidate ordinals. Contact ordinals remain
// four times the candidate ordinal. Every numerical island lane exclusively
// owns its Dynamic body write domain and preserves each kind's source order.
// The host must validate finite inputs, normalized body/shape/inertia rotations,
// zero shape local-position padding, positive dimensions, layer indices < 32,
// material combine modes in [MaterialCombineAverage, MaterialCombineMaximum],
// nonnegative material/damping/speed bounds, convex radius <= min half extent,
// disjoint complete author body/leaf world ranges with bodies.zw reserved zero,
// each leaf's owning body and range, matching body.meta.w, actual runtime world
// candidate begin/count ranges, StepParams.solve.zw equal to 4C/C, candidate and
// Contact counts within device indexing and buffer/device address limits.
// Cross-island numerical work remains bounded. A single connected island is
// serial in this first slice; this is not a published performance result.
// JointStream contains one header, a three-word directory per joint, then
// compact author and runtime payloads sized by the shared kind schema. Word
// offsets are local to this resource generation. World joint ordinal ranges
// are disjoint and complete; endpoints
// belong to that world or use InvalidQueryBodySlot for the world endpoint.
// Fixed/Hinge author parameters must be finite, axes unit length and
// stiffness/damping/torque/friction/armature nonnegative. SetupJoints writes
// runtime independently; WarmJoints applies warmstart once per substep, only
// with nonzero velocity iterations. Joints
// and contacts use the same combined island partition. Runtime stores never
// modify the header or author prefix. No entry uses over eight storage ports.

// Existing substep and iteration order is unchanged. Build and independently
// validate the combined Dynamic-writer constraint plan after all SetupContacts
// and all selected SetupJoints and before WarmJoints. Setup/Finish retain
// complete selected-world joint coverage. No failed/uninitialized accepted joint may be compacted into
// an apparently successful empty schedule; the original failure gate remains.
// All units of a phase/iteration precede its successor. Each island phase uses
// StageWork = (localFirst, localCount, islandTableFirst, islandLaneCount).
// Accepted units cover every island's ascending local sequence exactly once;
// contact position's local ordinal is patchLocal * 4 + point. Bound each unit's
// rectangular items by WorkItemLimit and dispatch ceil(islandLaneCount / 64).
// Dynamic body writes are exclusive across island lanes. Static/kinematic/world
// endpoints remain read-only. Graph rows refer to original accepted JointStream
// and canonical candidate ordinals; ContactPatch word addresses still add the
// ContactPatches[0] bank base. No impulses, Contact records or Joint runtime move.
// ConstraintSchedule replaces the numerical consumer's Worlds storage binding.
// ContactVelocity uses seven storage ports, Joint numerical stages five and
// ContactPosition four. No numerical consumer binds labels, ranges or ID lists
// through separate storage buffers. Graph stages have independent recipes.
// The complete plan workset owns frozen input identities and source allocations
// through all accepted graph/numerical receipts. An invalid plan is never
// installed or dispatched; consumer guard returns do not establish validity.
// There is one production numerical path. Independent original-trajectory GPU
// diagnostics are required before claiming exactness or measuring throughput.

} } // namespace JPH::GPU
