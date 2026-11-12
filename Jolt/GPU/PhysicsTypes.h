// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef JPH_GPU_PHYSICS_TYPES_H
#define JPH_GPU_PHYSICS_TYPES_H

// Shared, pointer-free C++ / Slang storage contract. This header intentionally
// does not include Jolt.h, Body, PhysicsSystem, a graphics API or an allocator.
// All positions are centers of mass; the bootstrap primitives are COM centered.
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
JPH_GPU_CONSTANT(MotionStatic, 0);
JPH_GPU_CONSTANT(MotionDynamic, 2);
// Correctness bootstrap bound, not a permanent limit on user physics worlds.
JPH_GPU_CONSTANT(MaxBodiesPerWorld, 32);

struct JPH_GPU_ALIGN BodyModel
{
    // shape index, shape type, motion type, world index.
    JPH_GPU_UINT4 meta;
    // Inverse mass, friction, restitution, gravity factor.
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
    // Sphere: xyz padding, w radius. Box: xyz half extents, w convex radius.
    // Box core = half extents - convex radius. The radius is not discarded.
    JPH_GPU_FLOAT4 geometry;
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

#ifdef __cplusplus
static_assert(sizeof(Float4) == 16 && alignof(Float4) == 16);
static_assert(sizeof(UInt4) == 16 && alignof(UInt4) == 16);
#define JPH_GPU_CHECK_TYPE(type, bytes) \
    static_assert(sizeof(type) == bytes && alignof(type) == 16); \
    static_assert(std::is_standard_layout_v<type> && std::is_trivially_copyable_v<type>)
JPH_GPU_CHECK_TYPE(BodyModel, 80);
JPH_GPU_CHECK_TYPE(ShapeModel, 16);
JPH_GPU_CHECK_TYPE(BodyState, 64);
JPH_GPU_CHECK_TYPE(BodyForces, 32);
JPH_GPU_CHECK_TYPE(WorldRange, 16);
JPH_GPU_CHECK_TYPE(Aabb, 32);
JPH_GPU_CHECK_TYPE(Contact, 80);
JPH_GPU_CHECK_TYPE(StepParams, 80);
JPH_GPU_CHECK_TYPE(PoseBinding, 112);
JPH_GPU_CHECK_TYPE(PoseParams, 16);
static_assert(offsetof(BodyModel, mass) == 16 && offsetof(BodyModel, damping) == 64);
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
