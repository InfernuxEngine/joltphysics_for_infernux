// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef JPH_GPU_PHYSICS_TYPES_H
#define JPH_GPU_PHYSICS_TYPES_H

// Shared, pointer-free C++ / Slang storage contract. This header intentionally
// does not include Jolt.h, Body, PhysicsSystem, a graphics API or an allocator.
// Body positions are centers of mass. A body has one primitive leaf or Empty;
// the primitive's local pose is relative to the body's center of mass.
#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace JPH { namespace GPU {

struct alignas(16) Float4 { float x, y, z, w; };
struct alignas(16) UInt4 { std::uint32_t x, y, z, w; };
#define JPH_GPU_ALIGN alignas(16)
#define JPH_GPU_FLOAT4 Float4
#define JPH_GPU_UINT4 UInt4
#define JPH_GPU_CONSTANT(name, value) inline constexpr std::uint32_t name = value
#else
#define JPH_GPU_ALIGN
#define JPH_GPU_FLOAT4 float4
#define JPH_GPU_UINT4 uint4
#define JPH_GPU_CONSTANT(name, value) static const uint name = value
#endif

JPH_GPU_CONSTANT(ShapeSphere, 0);
JPH_GPU_CONSTANT(ShapeBox, 1);
JPH_GPU_CONSTANT(ShapeEmpty, 2);
JPH_GPU_CONSTANT(MotionStatic, 0);
JPH_GPU_CONSTANT(MotionDynamic, 2);
// Residency is separate from motion type. A suspended actor retains its
// dynamic properties and state, but does not integrate or produce contacts.
JPH_GPU_CONSTANT(MotionNotResident, 4);
// Infernux PhysicsMaterialCombine values and priority (higher mode wins).
JPH_GPU_CONSTANT(MaterialCombineAverage, 0);
JPH_GPU_CONSTANT(MaterialCombineMinimum, 1);
JPH_GPU_CONSTANT(MaterialCombineMultiply, 2);
JPH_GPU_CONSTANT(MaterialCombineMaximum, 3);
// Correctness bootstrap bound, not a permanent limit on user physics worlds.
JPH_GPU_CONSTANT(MaxBodiesPerWorld, 32);
// FIFO body commands. These values match Infernux ForceMode.
JPH_GPU_CONSTANT(CommandSetPosition, 0);
JPH_GPU_CONSTANT(CommandSetRotation, 1);
JPH_GPU_CONSTANT(CommandSetPose, 2);
JPH_GPU_CONSTANT(CommandSetLinearVelocity, 3);
JPH_GPU_CONSTANT(CommandSetAngularVelocity, 4);
JPH_GPU_CONSTANT(CommandAddForce, 5);
JPH_GPU_CONSTANT(CommandAddTorque, 6);
JPH_GPU_CONSTANT(CommandAddForceAtPosition, 7);
// Internal motion-type boundary: preserve pose and clear velocities/forces,
// including after accepting a Static model that ignores dynamic API commands.
JPH_GPU_CONSTANT(CommandResetMotion, 8);
JPH_GPU_CONSTANT(CommandForce, 0);
JPH_GPU_CONSTANT(CommandAcceleration, 1);
JPH_GPU_CONSTANT(CommandImpulse, 2);
JPH_GPU_CONSTANT(CommandVelocityChange, 3);
JPH_GPU_CONSTANT(MaxCommandsPerBodyDispatch, 64);
JPH_GPU_CONSTANT(MaxCommandBodiesPerDispatch, 64);

struct JPH_GPU_ALIGN BodyModel
{
    // Shape index, shape type, motion type | MotionNotResident, world index. Each body has
    // one primitive leaf or Empty; a shared index has an identical local pose.
    JPH_GPU_UINT4 meta;
    // Inverse mass, friction, restitution, gravity factor.
    JPH_GPU_FLOAT4 mass;
    // Principal inverse inertia; w is padding.
    JPH_GPU_FLOAT4 invInertiaDiagonal;
    // Principal inertia frame relative to the body frame, quaternion xyzw.
    JPH_GPU_FLOAT4 inertiaRotation;
    // Linear damping, angular damping, maximum linear/angular speed.
    JPH_GPU_FLOAT4 damping;
    // Layer index [0, 31], collision mask, friction combine, bounce combine.
    // Both masks must admit the opposite layer for a pair to collide.
    JPH_GPU_UINT4 surface;
};

struct JPH_GPU_ALIGN ShapeModel
{
    // Sphere: xyz padding, w radius. Box: xyz half extents, w convex radius.
    // Empty has no collision geometry; all four geometry values are padding.
    // Box core = half extents - convex radius. The radius is not discarded.
    JPH_GPU_FLOAT4 geometry;
    // Primitive center in the body COM frame, xyz; w is zero padding. A collider
    // center already included in the body's COM must not be added a second time.
    JPH_GPU_FLOAT4 localPositionCOM;
    // Primitive frame relative to the body frame, normalized quaternion xyzw.
    // This does not rotate the body's principal inertia frame.
    JPH_GPU_FLOAT4 localRotation;
};

struct JPH_GPU_ALIGN BodyState
{
    JPH_GPU_FLOAT4 positionCOM;      // xyz position; w padding.
    JPH_GPU_FLOAT4 rotation;         // Normalized quaternion xyzw.
    JPH_GPU_FLOAT4 linearVelocity;   // xyz; w padding.
    JPH_GPU_FLOAT4 angularVelocity;  // xyz; w padding.
};

struct JPH_GPU_ALIGN BodyForces
{
    // Accumulator for one logical tick, shared by all of its substeps. The
    // tick owner clears it after the last substep. Continuous controllers
    // submit their contribution for each logical tick.
    JPH_GPU_FLOAT4 force;   // World space xyz; w padding.
    JPH_GPU_FLOAT4 torque;  // World space xyz; w padding.
};

struct JPH_GPU_ALIGN WorldRange
{
    // Body begin/count, contact begin/count. Contact count = body count squared.
    // Each ordered local pair has one slot; only i < j may become active.
    JPH_GPU_UINT4 range;
};

struct JPH_GPU_ALIGN Aabb
{
    JPH_GPU_FLOAT4 minimum; // xyz; w padding.
    JPH_GPU_FLOAT4 maximum; // xyz; w padding.
};

struct JPH_GPU_ALIGN Contact
{
    // Global body A/B indices, active (0/1), primitive contact feature.
    JPH_GPU_UINT4 meta;
    // Normal points from A towards B; positive depth means penetration.
    JPH_GPU_FLOAT4 normalDepth;
    JPH_GPU_FLOAT4 localAnchorA; // Body COM frame xyz; w padding.
    JPH_GPU_FLOAT4 localAnchorB; // Body COM frame xyz; w padding.
    // Accumulated normal/tangent1/tangent2 impulses, fixed normal velocity bias.
    // The bootstrap resets these every substep; it does not implement warmstart.
    JPH_GPU_FLOAT4 impulse;
};

struct JPH_GPU_ALIGN StepParams
{
    // Body count, world count, max body count in any world, velocity iterations.
    JPH_GPU_UINT4 counts;
    // Position iterations, substep count, total contact count, padding.
    JPH_GPU_UINT4 solve;
    // Gravity xyz and the delta time of this substep.
    JPH_GPU_FLOAT4 gravityDt;
    // Speculative distance, penetration slop, max position correction, Baumgarte.
    JPH_GPU_FLOAT4 contact;
    // Minimum restitution speed, then three padding values.
    JPH_GPU_FLOAT4 restitution;
};

// One renderer's immutable mesh-to-body-COM binding. The body index addresses
// the frozen BodyState generation supplied to the render-pose dispatch.
struct JPH_GPU_ALIGN PoseBinding
{
    // x is the body index; yzw are padding and must be zero.
    JPH_GPU_UINT4 bodyIndex;
    // Affine MeshLocalToCOM columns; w = 0, 0, 0, 1 respectively.
    JPH_GPU_FLOAT4 localToCOM0;
    JPH_GPU_FLOAT4 localToCOM1;
    JPH_GPU_FLOAT4 localToCOM2;
    JPH_GPU_FLOAT4 localToCOM3;
    // Mesh-local AABB center and nonnegative half extent; w is zero padding.
    JPH_GPU_FLOAT4 boundsCenter;
    JPH_GPU_FLOAT4 boundsExtent;
};

struct JPH_GPU_ALIGN PoseParams
{
    // x is the binding count; yzw are padding and must be zero.
    JPH_GPU_UINT4 bindingCount;
};

// Sparse body mutations at an owner-controlled boundary between logical ticks.
// Each list addresses unique slots. Neither operation advances simulation time.
struct JPH_GPU_ALIGN BodyMutationParams
{
    // Initialization count, COM update count, first initialization record, padding.
    JPH_GPU_UINT4 counts;
};

struct JPH_GPU_ALIGN BodyInitialization
{
    // x is the body slot; yzw are zero padding. The slot's actor identity is
    // owned by the engine registry, not encoded in the device array index.
    JPH_GPU_UINT4 bodyIndex;
    BodyState state;
    BodyForces forces;
};

struct JPH_GPU_ALIGN CenterOfMassUpdate
{
    JPH_GPU_UINT4 bodyIndex; // x is the body slot; yzw are zero padding.
    // New minus old root COM in the authored body frame, including baked scale.
    JPH_GPU_FLOAT4 deltaLocalCOM;
};

// Cold command payload, separate from the hot 96-byte BodyModel. Pose commands
// carry the accepted root COM in the authored body frame, including baked scale.
// Positions supplied by the public API are body origins, not COM positions.
struct JPH_GPU_ALIGN BodyCommand
{
    JPH_GPU_UINT4 meta; // Body slot, command kind, ForceMode, zero padding.
    JPH_GPU_FLOAT4 value; // Position, velocity, force or torque xyz; w is zero.
    // SetRotation/SetPose: normalized quaternion xyzw. Force-at-position:
    // world point xyz with zero w. Other commands do not consume this field.
    JPH_GPU_FLOAT4 auxiliary;
    JPH_GPU_FLOAT4 rootCOM; // Pose commands' local root COM xyz; w is zero.
};

struct JPH_GPU_ALIGN BodyCommandSpan
{
    // Body slot, first command, command count (1..64), zero padding.
    JPH_GPU_UINT4 range;
};

struct JPH_GPU_ALIGN BodyCommandParams
{
    // First span, span count (1..64); zw are zero padding.
    JPH_GPU_UINT4 spans;
};

JPH_GPU_CONSTANT(InvalidQueryBodySlot, 0xFFFFFFFFu);

struct JPH_GPU_ALIGN RaycastParams
{
    // Ray count, first body slot, body slots in this world (<=32), zero.
    JPH_GPU_UINT4 counts;
};

struct JPH_GPU_ALIGN RaycastInput
{
    JPH_GPU_FLOAT4 originMaxDistance; // World origin xyz; finite positive maximum distance.
    JPH_GPU_FLOAT4 direction; // Finite nonzero world direction xyz, normalized by the query; w zero.
    JPH_GPU_UINT4 filter; // Query layer mask; yzw zero. Uses BodyModel.surface.x, not collision mask.
};

struct JPH_GPU_ALIGN RaycastHit
{
    JPH_GPU_UINT4 meta; // Body SLOT (not public body ID), hit flag, zero, zero.
    JPH_GPU_FLOAT4 positionDistance; // World point xyz and distance; zero on miss.
    JPH_GPU_FLOAT4 normalFraction; // Outward world normal xyz and ray fraction; zero on miss.
};

#ifdef __cplusplus
static_assert(sizeof(Float4) == 16 && alignof(Float4) == 16);
static_assert(sizeof(UInt4) == 16 && alignof(UInt4) == 16);
#define JPH_GPU_CHECK_TYPE(type, bytes) \
    static_assert(sizeof(type) == bytes && alignof(type) == 16); \
    static_assert(std::is_standard_layout_v<type> && std::is_trivially_copyable_v<type>)
JPH_GPU_CHECK_TYPE(BodyModel, 96);
JPH_GPU_CHECK_TYPE(ShapeModel, 48);
JPH_GPU_CHECK_TYPE(BodyState, 64);
JPH_GPU_CHECK_TYPE(BodyForces, 32);
JPH_GPU_CHECK_TYPE(WorldRange, 16);
JPH_GPU_CHECK_TYPE(Aabb, 32);
JPH_GPU_CHECK_TYPE(Contact, 80);
JPH_GPU_CHECK_TYPE(StepParams, 80);
JPH_GPU_CHECK_TYPE(PoseBinding, 112);
JPH_GPU_CHECK_TYPE(PoseParams, 16);
JPH_GPU_CHECK_TYPE(BodyMutationParams, 16);
JPH_GPU_CHECK_TYPE(BodyInitialization, 112);
JPH_GPU_CHECK_TYPE(CenterOfMassUpdate, 32);
JPH_GPU_CHECK_TYPE(BodyCommand, 64);
JPH_GPU_CHECK_TYPE(BodyCommandSpan, 16);
JPH_GPU_CHECK_TYPE(BodyCommandParams, 16);
JPH_GPU_CHECK_TYPE(RaycastParams, 16);
JPH_GPU_CHECK_TYPE(RaycastInput, 48);
JPH_GPU_CHECK_TYPE(RaycastHit, 48);
static_assert(offsetof(RaycastInput, direction) == 16 && offsetof(RaycastInput, filter) == 32);
static_assert(offsetof(RaycastHit, positionDistance) == 16 && offsetof(RaycastHit, normalFraction) == 32);
static_assert(offsetof(BodyCommand, value) == 16 && offsetof(BodyCommand, auxiliary) == 32
    && offsetof(BodyCommand, rootCOM) == 48);
static_assert(offsetof(BodyInitialization, state) == 16 && offsetof(BodyInitialization, forces) == 80);
static_assert(offsetof(CenterOfMassUpdate, deltaLocalCOM) == 16);
static_assert(offsetof(BodyModel, meta) == 0 && offsetof(BodyModel, mass) == 16
    && offsetof(BodyModel, invInertiaDiagonal) == 32 && offsetof(BodyModel, inertiaRotation) == 48
    && offsetof(BodyModel, damping) == 64 && offsetof(BodyModel, surface) == 80);
static_assert(offsetof(ShapeModel, geometry) == 0 && offsetof(ShapeModel, localPositionCOM) == 16
    && offsetof(ShapeModel, localRotation) == 32);
static_assert(offsetof(BodyState, rotation) == 16 && offsetof(BodyState, angularVelocity) == 48);
static_assert(offsetof(Contact, normalDepth) == 16 && offsetof(Contact, impulse) == 64);
static_assert(offsetof(StepParams, gravityDt) == 32 && offsetof(StepParams, restitution) == 64);
static_assert(offsetof(PoseBinding, bodyIndex) == 0 && offsetof(PoseBinding, localToCOM0) == 16
    && offsetof(PoseBinding, localToCOM1) == 32 && offsetof(PoseBinding, localToCOM2) == 48
    && offsetof(PoseBinding, localToCOM3) == 64 && offsetof(PoseBinding, boundsCenter) == 80
    && offsetof(PoseBinding, boundsExtent) == 96);
static_assert(offsetof(PoseParams, bindingCount) == 0);
#undef JPH_GPU_CHECK_TYPE
} } // namespace JPH::GPU
#endif

#undef JPH_GPU_ALIGN
#undef JPH_GPU_FLOAT4
#undef JPH_GPU_UINT4
#undef JPH_GPU_CONSTANT
#endif // JPH_GPU_PHYSICS_TYPES_H
