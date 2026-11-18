// SPDX-License-Identifier: MIT
#ifndef JPH_GPU_CONTINUOUS_TYPES_H
#define JPH_GPU_CONTINUOUS_TYPES_H
#include "PhysicsTypes.h"

#ifdef __cplusplus
namespace JPH { namespace GPU {
#define JPH_CCD_ALIGN alignas(16)
#define JPH_CCD_UINT4 UInt4
#define JPH_CCD_FLOAT4 Float4
#else
#define JPH_CCD_ALIGN
#define JPH_CCD_UINT4 uint4
#define JPH_CCD_FLOAT4 float4
#endif

struct JPH_CCD_ALIGN ContinuousCpuBody
{
    BodyModel model;
    BodyState state;
    JPH_CCD_FLOAT4 displacement;
};
struct JPH_CCD_ALIGN ContinuousSample
{
    BodyState state;
    JPH_CCD_FLOAT4 displacement;
    JPH_CCD_UINT4 identity; // Global body slot, component root, first TOI bits or UINT32_MAX, zero.
};
struct JPH_CCD_ALIGN ContinuousRootPolicy
{
    JPH_CCD_UINT4 decision; // Component root, shared first-impact time bits, zero, zero.
};
struct JPH_CCD_ALIGN ContinuousMixedHit
{
    JPH_CCD_UINT4 bodies; // CPU body row, GPU slot, zero, zero.
    JPH_CCD_FLOAT4 normalFriction;
    JPH_CCD_FLOAT4 pointRestitution;
    JPH_CCD_FLOAT4 time; // TOI, travel fraction, minimum restitution speed, zero.
};
struct JPH_CCD_ALIGN ContinuousReceipt
{
    JPH_CCD_FLOAT4 linear; // Impulse on CPU xyz.
    JPH_CCD_FLOAT4 angular;
    JPH_CCD_FLOAT4 relative; // Incoming GPU point velocity minus CPU point velocity.
};
#ifdef __cplusplus
static_assert(sizeof(ContinuousCpuBody) == 160);
static_assert(sizeof(ContinuousSample) == 96);
static_assert(sizeof(ContinuousRootPolicy) == 16);
static_assert(sizeof(ContinuousMixedHit) == 64);
static_assert(sizeof(ContinuousReceipt) == 48);
} }
#endif
#undef JPH_CCD_ALIGN
#undef JPH_CCD_UINT4
#undef JPH_CCD_FLOAT4
#endif
