// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT
#ifndef JPH_GPU_CONSTRAINT_TYPES_H
#define JPH_GPU_CONSTRAINT_TYPES_H

#include "PhysicsTypes.h"

#ifdef __cplusplus
namespace JPH { namespace GPU {
#define CS_ALIGN alignas(16)
#define CS_U4 UInt4
#define CS_CONST(name, value) inline constexpr std::uint32_t name = value
#else
#define CS_ALIGN
#define CS_U4 uint4
#define CS_CONST(name, value) static const uint name = value
#endif

CS_CONST(ConstraintBlockSize, 64u);
CS_CONST(ConstraintItemLimit, 2048u);
CS_CONST(ConstraintBlockLimit, 32u);
CS_CONST(ConstraintConnectivityQuantum, 32u);
CS_CONST(ConstraintLinksPending, 1u);
CS_CONST(ConstraintRootsPending, 2u);
CS_CONST(ConstraintScheduleHeaderWords, 4u);
CS_CONST(IslandJointQuantum, 64u);
CS_CONST(IslandPatchQuantum, 256u);
CS_CONST(IslandContactQuantum, 1024u);
CS_CONST(ConstraintInvalid, 0xffffffffu);
CS_CONST(ConstraintJoint, 0u);
CS_CONST(ConstraintPatch, 1u);
CS_CONST(ConstraintBadModel, 1u);
CS_CONST(ConstraintBadJoint, 2u);
CS_CONST(ConstraintBadManifold, 4u);
CS_CONST(ConstraintBadLabel, 8u);
CS_CONST(ConstraintNotConverged, 16u);
CS_CONST(ConstraintOverflow, 32u);
CS_CONST(ConstraintBadSchedule, 64u);
CS_CONST(ConstraintNotReady, 0x80000000u);

struct CS_ALIGN ConstraintScheduleWord { CS_U4 value; };

// All addresses are UInt4 word offsets in ConstraintSchedule, except labels
// (independent scalar uint32 allocations, stride 4) and source ordinals.
// All ranges / products / dispatch dimensions must be checked in host uint64.
// Checked constraints: 2*(J+P), 4*P, 12*J, B*2, totalWords all fit uint32;
// every nonempty read/write range fits storage/device limits and is disjoint
// from concurrent writers. Counts describe this accepted source, not capacity
// tails. B and J span the selected world. Tombstones are nonresident models.
struct CS_ALIGN ConstraintParams
{
    CS_U4 counts;       // B local slots, J local joints, actual P candidates, packed JointStream directory size.
    CS_U4 limits;       // total schedule words, local shape count, contact rows=4P, body-root word base.
    CS_U4 layout;       // header base=0, connectivity control, body-check base, edge cursors/ref-check base.
    CS_U4 references;   // joint ref base, patch ref base (each 2 words/item), final joint sort bank, final patch sort bank.
    CS_U4 sources;      // joint source word base, patch source word base, joint ID-table base, patch ID-table base.
    CS_U4 islands;      // 2B-word range base, B root-row base, B exclusive-prefix base, B range-summary-row base.
    CS_U4 scan;         // input base, exclusive-prefix base, block-total base, input length.
    CS_U4 scanParent;   // scanned parent-prefix base, combine mode (0 or 1), 0, 0.
    CS_U4 merge;        // source bank, target bank, length (J or P), run width power-of-two<=2^31.
    CS_U4 select;       // kind (Joint=0/Patch=1), sort seed bank, numeric refs, zero.
    CS_U4 summary;      // root-scan total word, range-reduction total word, ref-validation total word, body-check total word.
    CS_U4 selection;    // accepted packed world row and body, leaf, joint bases.
};

// Frozen numerical consumer header (four words, base=0):
// h0=(islandCount,jointIslandCount,contactIslandCount,status)
// h1=(activeJ,activeP,maxJointItems,maxPatchItems)
// h2=(rangeWordBase,jointTableWordBase,contactTableWordBase,jointSourceWordBase)
// h3=(patchSourceWordBase,wordCount,bodyRootWordBase,worldBodySlots)
// status==0 is the ONLY installable state. Initialization sets NotReady.
// range[i].word0=(world,root,jointBegin,jointCount)
// range[i].word1=(patchBegin,patchCount,0,0)
// begin/count index the respective source-word stream (not absolute words).
// Tables contain (islandID,0,0,0); sources contain (sourceOrdinal,0,0,0).
// Joint ordinals address the unchanged accepted JointStream. Patch ordinals
// address the unchanged candidate bank and four original Contact records.
// Within each range all source ordinals strictly increase, independently for
// Joint and patch phases. No numeric state is copied, reordered or solved here.
// Island IDs follow increasing root BODY SLOT (world order need not match).
//
// Scratch ref: word0=(world,writerA,writerB,sourceOrdinal),
// word1=(kind,active01,status,0). Readonly writer endpoints use Invalid.
// Sort key=(world,root,sourceOrdinal,kind). Inactive keys are all Invalid;
// stable merge places all active keys before initialized inactive tails.
// Root flag scalar (reuses the sleep vote bank after wake):
// bit0=has Joint, bit1=has patch. Only active refs mark roots.
// Labels and body-root rows are local to the selected world.
// Parents use B scalars. The separate vote bank has B+1: connectivity uses
// its last scalar for pending work; sleep uses [0,B) for votes and B for status.
// References and island ranges retain packed body slots.
// Scan mode0 record=(sum0,sum1,sum2,ORstatus), checked u32 sums.
// Scan mode1 record=(sumJ,sumP,maxJ,maxP); sums bounded by checked J/P.
// Ref validation total (mode0)=(activeJ,activeP,0,status).
// Connectivity control=(pending flags,completed passes,status,validated01).
// Each pass joins bounded parent chains from persistent edge cursors, then
// flattens parents once every edge is joined. The device enables the validation
// tail through indirect argument records indexed by workgroup count (1..32).
// The host reads the control after the tail; the numerical header is published later.
// Both atomic banks have scalar uint32 layout, with dispatch dependencies
// between reset, linking, flattening, and publication.
// Reflected storage access remains read_write even in load-only label entries.
// Do not bind an atomic label array as WGSL storage read-only or use a UInt4
// field as its backing storage. CPU POD remains scalar uint32, not a C++ atomic.

#ifdef __cplusplus
static_assert(sizeof(ConstraintScheduleWord) == 16);
static_assert(alignof(ConstraintScheduleWord) == 16);
static_assert(std::is_trivially_copyable_v<ConstraintScheduleWord>);
static_assert(sizeof(ConstraintParams) == 12 * 16);
static_assert(alignof(ConstraintParams) == 16);
} }
#endif
#undef CS_ALIGN
#undef CS_U4
#undef CS_CONST
#endif
