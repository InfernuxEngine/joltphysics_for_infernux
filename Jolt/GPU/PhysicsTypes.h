// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef JPH_GPU_PHYSICS_TYPES_H
#define JPH_GPU_PHYSICS_TYPES_H

// Shared, pointer-free C++ / Slang storage contract. This header intentionally
// does not include Jolt.h, Body, PhysicsSystem, a graphics API or an allocator.
// Body positions are centers of mass. Each body addresses a contiguous leaf
// range; every leaf's local pose is relative to that body's center of mass.
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
JPH_GPU_CONSTANT(ShapeCapsule, 2);
JPH_GPU_CONSTANT(ShapeSensor, 1);
JPH_GPU_CONSTANT(InvalidGeometryPayload, 0xffffffffu);
JPH_GPU_CONSTANT(PatchActive, 1);
JPH_GPU_CONSTANT(PatchSensor, 2);
JPH_GPU_CONSTANT(PatchOverlapping, 4);
JPH_GPU_CONSTANT(MotionStatic, 0);
JPH_GPU_CONSTANT(MotionKinematic, 1);
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
// Fixed point records per independent ordered leaf pair.
JPH_GPU_CONSTANT(MaxContactPointsPerPair, 4);
// Sentinel for a render hierarchy node without a physical ancestor.
JPH_GPU_CONSTANT(PoseNoParent, 0xffffffffu);
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
JPH_GPU_CONSTANT(CommandMovePosition, 9);
JPH_GPU_CONSTANT(CommandMoveRotation, 10);
JPH_GPU_CONSTANT(CommandMovePose, 11);
JPH_GPU_CONSTANT(CommandMotionBoundary, 12);
// meta.z is decoded by command kind, independently of ForceMode.
JPH_GPU_CONSTANT(CommandPreserveKinematicMove, 1);
JPH_GPU_CONSTANT(CommandTemporaryStatic, 1);
JPH_GPU_CONSTANT(MotionBoundaryActionMask, 3);
JPH_GPU_CONSTANT(MotionBoundaryTypeChange, 0);
JPH_GPU_CONSTANT(MotionBoundarySuspendPreserve, 1);
JPH_GPU_CONSTANT(MotionBoundarySuspendReset, 2);
JPH_GPU_CONSTANT(MotionBoundaryCarryAuthoredPosition, 4);
JPH_GPU_CONSTANT(MotionBoundaryCarryAuthoredRotation, 8);
JPH_GPU_CONSTANT(TargetPending, 1);
JPH_GPU_CONSTANT(TargetActive, 2);
JPH_GPU_CONSTANT(TargetDrivenLastTick, 4);
JPH_GPU_CONSTANT(TargetTemporaryStatic, 8);
JPH_GPU_CONSTANT(CommandForce, 0);
JPH_GPU_CONSTANT(CommandAcceleration, 1);
JPH_GPU_CONSTANT(CommandImpulse, 2);
JPH_GPU_CONSTANT(CommandVelocityChange, 3);
JPH_GPU_CONSTANT(MaxCommandsPerBodyDispatch, 64);
JPH_GPU_CONSTANT(MaxCommandBodiesPerDispatch, 64);

struct JPH_GPU_ALIGN BodyModel
{
    // First leaf, leaf count, motion type | MotionNotResident, world index.
    // Zero leaves represent an empty collision body without a dummy primitive.
    JPH_GPU_UINT4 meta;
    // Inverse mass, zero padding, zero padding, gravity factor.
    JPH_GPU_FLOAT4 mass;
    // Principal inverse inertia; w is padding.
    JPH_GPU_FLOAT4 invInertiaDiagonal;
    // Principal inertia frame relative to the body frame, quaternion xyzw.
    JPH_GPU_FLOAT4 inertiaRotation;
    // Linear damping, angular damping, maximum linear/angular speed.
    JPH_GPU_FLOAT4 damping;
};

struct JPH_GPU_ALIGN ShapeModel
{
    // Shape type, owning body slot, root SubShapeID prefix, ShapeSensor flags.
    JPH_GPU_UINT4 meta;
    // Sphere: xyz padding, w radius. Box: xyz half extents, w convex radius.
    // Capsule: (0, half cylinder height, 0, radius), with the core along local Y.
    // Box core = half extents - convex radius. The radius is not discarded.
    JPH_GPU_FLOAT4 geometry;
    // Primitive center in the body COM frame, xyz; w is zero padding. A collider
    // center already included in the body's COM must not be added a second time.
    JPH_GPU_FLOAT4 localPositionCOM;
    // Primitive frame relative to the body frame, normalized quaternion xyzw.
    // This does not rotate the body's principal inertia frame.
    JPH_GPU_FLOAT4 localRotation;
    // Friction, restitution, then two zero padding values.
    JPH_GPU_FLOAT4 material;
    // Layer index [0, 31], collision mask, friction combine, bounce combine.
    // Both masks must admit the opposite leaf's layer for a pair to collide.
    JPH_GPU_UINT4 surface;
    // Collider ID low/high words, geometry payload index, SubShapeID prefix
    // bits written. Primitive leaves use InvalidGeometryPayload. The full
    // Collider identity is preserved without requiring device uint64 math.
    JPH_GPU_UINT4 source;
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

// Device-owned target/arrival-stop state. No CPU dynamic pose mirror is used.
// The inactive payload is not read; initialization and retirement clear it.
struct JPH_GPU_ALIGN KinematicTarget
{
    JPH_GPU_FLOAT4 positionSpeed; // Target body origin xyz; w maximum drive speed (0 = uncapped).
    JPH_GPU_FLOAT4 rotation;      // Target normalized quaternion xyzw.
    JPH_GPU_FLOAT4 rootCOMDt;     // Accepted local root COM xyz; w requested duration.
    JPH_GPU_UINT4 state;          // Target flags in x; yzw zero.
};

struct JPH_GPU_ALIGN WorldRange
{
    // Body begin/count, patch begin/count. The dense correctness collector
    // allocates leaf count squared patches, including inactive lower pairs.
    JPH_GPU_UINT4 bodies;
    // Leaf begin/count, then two zero padding values.
    JPH_GPU_UINT4 leaves;
};

struct JPH_GPU_ALIGN Aabb
{
    JPH_GPU_FLOAT4 minimum; // xyz; w padding.
    JPH_GPU_FLOAT4 maximum; // xyz; w padding.
};

struct JPH_GPU_ALIGN ContactPatch
{
    // Global leaf A/B indices, patch flags, zero padding. Each patch owns
    // MaxContactPointsPerPair consecutive Contact records. Sensor patches
    // retain real overlap/source but never activate response points.
    JPH_GPU_UINT4 leaves;
};

struct JPH_GPU_ALIGN Contact
{
    // Global body A/B indices, active (0/1), primitive contact feature.
    JPH_GPU_UINT4 meta;
    // Normal points from A towards B; positive depth means penetration.
    JPH_GPU_FLOAT4 normalDepth;
    // Body COM frame xyz. In the first record of a pair, w stores the
    // accumulated manifold angular-friction impulse; other records keep w=0.
    JPH_GPU_FLOAT4 localAnchorA;
    JPH_GPU_FLOAT4 localAnchorB; // Body COM frame xyz; w padding.
    // x is this point's normal impulse; w is its fixed normal velocity bias.
    // yz in the first pair record are the manifold's two linear-friction
    // impulses; other records keep yz=0. Collision/setup reset all impulse
    // scratch every substep; the bootstrap does not implement warmstart.
    JPH_GPU_FLOAT4 impulse;
};

struct JPH_GPU_ALIGN StepParams
{
    // Body count, world count, total leaf count, velocity iterations.
    JPH_GPU_UINT4 counts;
    // Position iterations, substep count, total contact count, total patch count.
    JPH_GPU_UINT4 solve;
    // Gravity xyz and the delta time of this substep.
    JPH_GPU_FLOAT4 gravityDt;
    // Speculative distance, penetration slop, max position correction, Baumgarte.
    JPH_GPU_FLOAT4 contact;
    // Minimum restitution speed, then three padding values.
    JPH_GPU_FLOAT4 restitution;
};

// Immutable render hierarchy nodes address one frozen BodyState generation.
// Parent chains are acyclic and contain at most MaxBodiesPerWorld nodes.
// Ordinary parent chains retain both their affine and their authored TRS inverse:
// a collapsed scale uses zero reciprocal, rather than a generic matrix inverse.
struct JPH_GPU_ALIGN PoseNode
{
    // Body-state slot, parent PoseNode index or PoseNoParent, then zero padding.
    JPH_GPU_UINT4 bodyParent;
    // Accepted body-origin-to-COM vector in the body frame; w is zero padding.
    JPH_GPU_FLOAT4 rootCOM;
    // This Transform's authored signed local scale; w is zero padding.
    JPH_GPU_FLOAT4 localScale;
    // Quaternion xyzw of the ordinary chain from the parent body root to this
    // Transform's direct parent. With PoseNoParent, the direct parent's world
    // quaternion instead. Independent of the affine matrix; do not decompose it.
    JPH_GPU_FLOAT4 parentRotation;
    // Affine columns mapping this body's direct parent into its ancestor body
    // root, or into world space for PoseNoParent. w = 0, 0, 0, 1 respectively.
    JPH_GPU_FLOAT4 parentLocalToAncestor0;
    JPH_GPU_FLOAT4 parentLocalToAncestor1;
    JPH_GPU_FLOAT4 parentLocalToAncestor2;
    JPH_GPU_FLOAT4 parentLocalToAncestor3;
    // Reverse map composed from each ordinary Transform's authored inverse TRS,
    // in reverse order. abs(localScale) < 1e-9 has zero reciprocal per axis.
    // This is not the generic inverse or pseudoinverse of the forward matrix.
    // Columns have w = 0, 0, 0, 1 respectively.
    JPH_GPU_FLOAT4 parentAncestorToLocal0;
    JPH_GPU_FLOAT4 parentAncestorToLocal1;
    JPH_GPU_FLOAT4 parentAncestorToLocal2;
    JPH_GPU_FLOAT4 parentAncestorToLocal3;
};

// One renderer's ordinary local chain from its nearest exported body root.
// Mesh draw and existing local bounds deliberately use separate pivot paths.
struct JPH_GPU_ALIGN PoseBinding
{
    // PoseNode index in x; yzw are zero padding.
    JPH_GPU_UINT4 nodeIndex;
    // Affine object-local-to-body-root columns; w = 0, 0, 0, 1 respectively.
    // The chain contains no other exported body node.
    JPH_GPU_FLOAT4 objectLocal0;
    JPH_GPU_FLOAT4 objectLocal1;
    JPH_GPU_FLOAT4 objectLocal2;
    JPH_GPU_FLOAT4 objectLocal3;
    // Selected-submesh draw pivot, or zero for an unshifted draw. w is zero.
    JPH_GPU_FLOAT4 drawPivot;
    // Existing object-local AABB center and nonnegative half extent. These
    // bounds already contain any selected-submesh pivot. Both w are zero.
    JPH_GPU_FLOAT4 boundsCenter;
    JPH_GPU_FLOAT4 boundsExtent;
};

struct JPH_GPU_ALIGN PoseParams
{
    // Node count, renderer binding count, then zero padding.
    JPH_GPU_UINT4 counts;
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
    JPH_GPU_UINT4 meta; // Body slot, kind, kind-specific mode/flags, zero padding.
    // Position/velocity/force/torque xyz. Move kinds use w for requested dt;
    // all other consumed vector w values are zero.
    JPH_GPU_FLOAT4 value;
    // SetRotation/SetPose: normalized quaternion xyzw. Force-at-position:
    // world point xyz with zero w. MoveRotation/MovePose also consume quaternion.
    // Other commands do not consume this field.
    JPH_GPU_FLOAT4 auxiliary;
    // Pose/Move local root COM xyz. Move kinds use w for maximum drive speed;
    // ordinary pose commands use zero w. Boundary carries use device Target.
    JPH_GPU_FLOAT4 rootCOM;
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
JPH_GPU_CONSTANT(QueryRay, 0);
JPH_GPU_CONSTANT(QueryOverlap, 1);
JPH_GPU_CONSTANT(QueryClosestPoint, 2);
JPH_GPU_CONSTANT(QueryBodyBounds, 3);
JPH_GPU_CONSTANT(QueryPenetration, 4);
JPH_GPU_CONSTANT(QueryGatherShapes, 5);
JPH_GPU_CONSTANT(QueryGatherMotion, 6);
JPH_GPU_CONSTANT(QueryAllShapeTypes, 0xFFFFFFFFu);
JPH_GPU_CONSTANT(QueryOutputHit, 0);
JPH_GPU_CONSTANT(QueryOutputShape, 1);
JPH_GPU_CONSTANT(QueryOutputMotion, 2);
JPH_GPU_CONSTANT(QueryShapeSensor, 1);
JPH_GPU_CONSTANT(QueryShapeMoving, 2);
JPH_GPU_CONSTANT(QueryExportMotionPosition, 0);
JPH_GPU_CONSTANT(QueryExportMotionRotation, 1);
JPH_GPU_CONSTANT(QueryExportMotionCenterOfMass, 2);
JPH_GPU_CONSTANT(QueryExportMotionLinearVelocity, 3);
JPH_GPU_CONSTANT(QueryExportMotionAngularVelocity, 4);
JPH_GPU_CONSTANT(QueryExportMotionInverseMass, 5);
JPH_GPU_CONSTANT(QueryExportMotionInverseInertia, 6);
JPH_GPU_CONSTANT(QueryExportShapeBodyIndex, 7);
JPH_GPU_CONSTANT(QueryExportShapeCenter, 8);
JPH_GPU_CONSTANT(QueryExportShapeHalfExtents, 9);
JPH_GPU_CONSTANT(QueryExportShapeRotation, 10);
JPH_GPU_CONSTANT(QueryExportShapeFriction, 11);
JPH_GPU_CONSTANT(QueryExportShapeBounciness, 12);
JPH_GPU_CONSTANT(QueryClosest, 0);
JPH_GPU_CONSTANT(QueryAll, 1);
JPH_GPU_CONSTANT(QueryUniqueCollider, 2);
JPH_GPU_CONSTANT(QueryUniqueBody, 3);
JPH_GPU_CONSTANT(QueryAny, 4);
JPH_GPU_CONSTANT(QueryTargetCollider, 1);
JPH_GPU_CONSTANT(QueryExcludeBody, 2);
JPH_GPU_CONSTANT(QuerySuccess, 0);
JPH_GPU_CONSTANT(QueryUnsupported, 1);
JPH_GPU_CONSTANT(QueryNumericalFailure, 2);

struct JPH_GPU_ALIGN QueryParams
{
    // Request count, first body slot, body slots, published leaf count.
    JPH_GPU_UINT4 counts;
    // Explicit primitive count, per-request scratch record stride, record capacity, output format.
    JPH_GPU_UINT4 sizes;
};

struct JPH_GPU_ALIGN QueryInput
{
    // Operation, collector, query primitive type, filter flags.
    JPH_GPU_UINT4 meta;
    // Layer mask, include sensors, exact Collider ID low/high when requested.
    JPH_GPU_UINT4 filter;
    // Excluded body slot; second primitive / gather target slot; requested motion fields; zero.
    JPH_GPU_UINT4 source;
    // Ray origin / point / query shape center / bounds center; w maximum travel.
    JPH_GPU_FLOAT4 positionMaximum;
    // Ray direction or bounds half extent; w zero padding.
    JPH_GPU_FLOAT4 direction;
    JPH_GPU_FLOAT4 rotation; // Query primitive quaternion xyzw.
    JPH_GPU_FLOAT4 geometry; // Same cooked geometry law as ShapeModel.
};

struct JPH_GPU_ALIGN QueryPrimitive
{
    JPH_GPU_UINT4 meta; // Shape type, then zero padding.
    JPH_GPU_FLOAT4 geometry;
    JPH_GPU_FLOAT4 position;
    JPH_GPU_FLOAT4 rotation;
};

struct JPH_GPU_ALIGN QueryBodyIdentity
{
    JPH_GPU_UINT4 ids; // Public body ID, has Rigidbody, Rigidbody component ID low/high.
};

struct JPH_GPU_ALIGN QueryHit
{
    JPH_GPU_UINT4 meta; // Body slot, valid flag, global leaf ordinal, feature.
    JPH_GPU_FLOAT4 positionDistance; // Target surface point xyz and travel/distance.
    JPH_GPU_FLOAT4 normalFraction; // Target outward normal xyz and travel fraction.
    JPH_GPU_FLOAT4 pointADepth; // Query surface witness xyz and positive penetration.
};

struct JPH_GPU_ALIGN QueryRecordWord
{
    JPH_GPU_UINT4 value;
};

struct JPH_GPU_ALIGN QueryShapeState
{
    JPH_GPU_UINT4 meta;
    JPH_GPU_UINT4 surface;
    JPH_GPU_FLOAT4 geometry;
    JPH_GPU_FLOAT4 center;
    JPH_GPU_FLOAT4 rotation;
    JPH_GPU_FLOAT4 bodyCenterOfMass;
    JPH_GPU_FLOAT4 linearVelocity;
    JPH_GPU_FLOAT4 angularVelocity;
    JPH_GPU_FLOAT4 material;
};

struct JPH_GPU_ALIGN QueryMotionState
{
    JPH_GPU_UINT4 meta;
    JPH_GPU_FLOAT4 positionInverseMass;
    JPH_GPU_FLOAT4 rotation;
    JPH_GPU_FLOAT4 centerOfMass;
    JPH_GPU_FLOAT4 linearVelocity;
    JPH_GPU_FLOAT4 angularVelocity;
    JPH_GPU_FLOAT4 inverseInertia0;
    JPH_GPU_FLOAT4 inverseInertia1;
    JPH_GPU_FLOAT4 inverseInertia2;
    JPH_GPU_FLOAT4 inertia0;
    JPH_GPU_FLOAT4 inertia1;
    JPH_GPU_FLOAT4 inertia2;
};

struct JPH_GPU_ALIGN QueryExportParams
{
    JPH_GPU_UINT4 output; // format, field, record count, scalar offset.
    JPH_GPU_UINT4 sizes; // request count, record words, scalars per output row, zero.
};

struct JPH_GPU_ALIGN QuerySpan
{
    JPH_GPU_UINT4 range; // Compact offset, record count, status, zero padding.
};

struct JPH_GPU_ALIGN QuerySummary
{
    JPH_GPU_UINT4 counts; // Total records, status, request count, zero padding.
};

#ifdef __cplusplus
static_assert(sizeof(Float4) == 16 && alignof(Float4) == 16);
static_assert(sizeof(UInt4) == 16 && alignof(UInt4) == 16);
#define JPH_GPU_CHECK_TYPE(type, bytes) \
    static_assert(sizeof(type) == bytes && alignof(type) == 16); \
    static_assert(std::is_standard_layout_v<type> && std::is_trivially_copyable_v<type>)
JPH_GPU_CHECK_TYPE(BodyModel, 80);
JPH_GPU_CHECK_TYPE(ShapeModel, 112);
JPH_GPU_CHECK_TYPE(BodyState, 64);
JPH_GPU_CHECK_TYPE(BodyForces, 32);
JPH_GPU_CHECK_TYPE(KinematicTarget, 64);
JPH_GPU_CHECK_TYPE(WorldRange, 32);
JPH_GPU_CHECK_TYPE(Aabb, 32);
JPH_GPU_CHECK_TYPE(ContactPatch, 16);
JPH_GPU_CHECK_TYPE(Contact, 80);
JPH_GPU_CHECK_TYPE(StepParams, 80);
JPH_GPU_CHECK_TYPE(PoseNode, 192);
JPH_GPU_CHECK_TYPE(PoseBinding, 128);
JPH_GPU_CHECK_TYPE(PoseParams, 16);
JPH_GPU_CHECK_TYPE(BodyMutationParams, 16);
JPH_GPU_CHECK_TYPE(BodyInitialization, 112);
JPH_GPU_CHECK_TYPE(CenterOfMassUpdate, 32);
JPH_GPU_CHECK_TYPE(BodyCommand, 64);
JPH_GPU_CHECK_TYPE(BodyCommandSpan, 16);
JPH_GPU_CHECK_TYPE(BodyCommandParams, 16);
JPH_GPU_CHECK_TYPE(QueryParams, 32);
JPH_GPU_CHECK_TYPE(QueryInput, 112);
JPH_GPU_CHECK_TYPE(QueryPrimitive, 64);
JPH_GPU_CHECK_TYPE(QueryBodyIdentity, 16);
JPH_GPU_CHECK_TYPE(QueryHit, 64);
JPH_GPU_CHECK_TYPE(QueryRecordWord, 16);
JPH_GPU_CHECK_TYPE(QueryShapeState, 144);
JPH_GPU_CHECK_TYPE(QueryMotionState, 192);
JPH_GPU_CHECK_TYPE(QueryExportParams, 32);
static_assert(offsetof(QueryRecordWord, value) == 0 && offsetof(QueryExportParams, sizes) == 16);
static_assert(offsetof(QueryShapeState, surface) == 16 && offsetof(QueryShapeState, geometry) == 32
    && offsetof(QueryShapeState, center) == 48 && offsetof(QueryShapeState, rotation) == 64
    && offsetof(QueryShapeState, bodyCenterOfMass) == 80 && offsetof(QueryShapeState, linearVelocity) == 96
    && offsetof(QueryShapeState, angularVelocity) == 112 && offsetof(QueryShapeState, material) == 128);
static_assert(offsetof(QueryMotionState, positionInverseMass) == 16 && offsetof(QueryMotionState, rotation) == 32
    && offsetof(QueryMotionState, centerOfMass) == 48 && offsetof(QueryMotionState, linearVelocity) == 64
    && offsetof(QueryMotionState, angularVelocity) == 80 && offsetof(QueryMotionState, inverseInertia0) == 96
    && offsetof(QueryMotionState, inverseInertia1) == 112 && offsetof(QueryMotionState, inverseInertia2) == 128
    && offsetof(QueryMotionState, inertia0) == 144 && offsetof(QueryMotionState, inertia1) == 160
    && offsetof(QueryMotionState, inertia2) == 176);
JPH_GPU_CHECK_TYPE(QuerySpan, 16);
JPH_GPU_CHECK_TYPE(QuerySummary, 16);
static_assert(offsetof(QueryInput, geometry) == 96 && offsetof(QueryInput, filter) == 16);
static_assert(offsetof(QueryHit, positionDistance) == 16 && offsetof(QueryHit, pointADepth) == 48);
static_assert(offsetof(BodyCommand, value) == 16 && offsetof(BodyCommand, auxiliary) == 32
    && offsetof(BodyCommand, rootCOM) == 48);
static_assert(offsetof(KinematicTarget, positionSpeed) == 0 && offsetof(KinematicTarget, rotation) == 16
    && offsetof(KinematicTarget, rootCOMDt) == 32 && offsetof(KinematicTarget, state) == 48);
static_assert(offsetof(BodyInitialization, state) == 16 && offsetof(BodyInitialization, forces) == 80);
static_assert(offsetof(CenterOfMassUpdate, deltaLocalCOM) == 16);
static_assert(offsetof(BodyModel, meta) == 0 && offsetof(BodyModel, mass) == 16
    && offsetof(BodyModel, invInertiaDiagonal) == 32 && offsetof(BodyModel, inertiaRotation) == 48
    && offsetof(BodyModel, damping) == 64);
static_assert(offsetof(ShapeModel, meta) == 0 && offsetof(ShapeModel, geometry) == 16
    && offsetof(ShapeModel, localPositionCOM) == 32 && offsetof(ShapeModel, localRotation) == 48
    && offsetof(ShapeModel, material) == 64 && offsetof(ShapeModel, surface) == 80
    && offsetof(ShapeModel, source) == 96);
static_assert(offsetof(WorldRange, bodies) == 0 && offsetof(WorldRange, leaves) == 16);
static_assert(offsetof(ContactPatch, leaves) == 0);
static_assert(offsetof(BodyState, rotation) == 16 && offsetof(BodyState, angularVelocity) == 48);
static_assert(offsetof(Contact, normalDepth) == 16 && offsetof(Contact, impulse) == 64);
static_assert(offsetof(StepParams, gravityDt) == 32 && offsetof(StepParams, restitution) == 64);
static_assert(offsetof(PoseNode, bodyParent) == 0 && offsetof(PoseNode, rootCOM) == 16
    && offsetof(PoseNode, localScale) == 32 && offsetof(PoseNode, parentRotation) == 48
    && offsetof(PoseNode, parentLocalToAncestor0) == 64 && offsetof(PoseNode, parentLocalToAncestor1) == 80
    && offsetof(PoseNode, parentLocalToAncestor2) == 96 && offsetof(PoseNode, parentLocalToAncestor3) == 112
    && offsetof(PoseNode, parentAncestorToLocal0) == 128 && offsetof(PoseNode, parentAncestorToLocal1) == 144
    && offsetof(PoseNode, parentAncestorToLocal2) == 160 && offsetof(PoseNode, parentAncestorToLocal3) == 176);
static_assert(offsetof(PoseBinding, nodeIndex) == 0 && offsetof(PoseBinding, objectLocal0) == 16
    && offsetof(PoseBinding, objectLocal1) == 32 && offsetof(PoseBinding, objectLocal2) == 48
    && offsetof(PoseBinding, objectLocal3) == 64 && offsetof(PoseBinding, drawPivot) == 80
    && offsetof(PoseBinding, boundsCenter) == 96 && offsetof(PoseBinding, boundsExtent) == 112);
static_assert(offsetof(PoseParams, counts) == 0);
#undef JPH_GPU_CHECK_TYPE
} } // namespace JPH::GPU
#endif

#undef JPH_GPU_ALIGN
#undef JPH_GPU_FLOAT4
#undef JPH_GPU_UINT4
#undef JPH_GPU_CONSTANT
#endif // JPH_GPU_PHYSICS_TYPES_H
