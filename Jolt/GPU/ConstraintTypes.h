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
CS_CONST(ConstraintRoundWindow, 4u);
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
// tails. B is accepted stable-slot extent; tombstones are nonresident models.
struct CS_ALIGN ConstraintParams
{
    CS_U4 counts;       // B slots, accepted J, actual candidate P, accepted W.
    CS_U4 limits;       // total schedule words, accepted shape count, contact rows=4P, 0.
    CS_U4 layout;       // header base=0, CC control word, body-check row base, ref-check row base.
    CS_U4 references;   // joint ref base, patch ref base (each 2 words/item), final joint sort bank, final patch sort bank.
    CS_U4 sources;      // joint source word base, patch source word base, joint ID-table base, patch ID-table base.
    CS_U4 islands;      // 2B-word range base, B root-row base, B exclusive-prefix base, B range-summary-row base.
    CS_U4 scan;         // input base, exclusive-prefix base, block-total base, input length.
    CS_U4 scanParent;   // scanned parent-prefix base, combine mode (0 or 1), 0, 0.
    CS_U4 merge;        // source bank, target bank, length (J or P), run width power-of-two<=2^31.
    CS_U4 select;       // kind (Joint=0/Patch=1), sort seed bank, 0, 0.
    CS_U4 summary;      // root-scan total word, range-reduction total word, ref-validation total word, CC-check total word.
    CS_U4 round;        // round body-check total word; yzw zero padding.
    // One accepted layout address, never a stable RuntimeWorld identity.
    // selection=(world row, body base, leaf base, joint base).
    // selectedCounts=(body capacity, leaf count, joint count, zero).
    // Counts above remain TOTAL array extents for global address validation.
    CS_U4 selection;
    CS_U4 selectedCounts;
};

// Frozen numerical consumer header (four words, base=0):
// h0=(islandCount,jointIslandCount,contactIslandCount,status)
// h1=(activeJ,activeP,maxJointItems,maxPatchItems)
// h2=(rangeWordBase,jointTableWordBase,contactTableWordBase,jointSourceWordBase)
// h3=(patchSourceWordBase,wordCount,0,0)
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
// Root flag scalar (reuses LabelsNext AFTER final CC validation):
// bit0=has Joint, bit1=has patch. Only active refs mark roots.
// Scan mode0 record=(sum0,sum1,sum2,ORstatus), checked u32 sums.
// Scan mode1 record=(sumJ,sumP,maxJ,maxP); sums bounded by checked J/P.
// Ref validation total (mode0)=(activeJ,activeP,0,status).
// CC control=(changed,roundsCompleted,status,validated01). Control is separate
// from numerical header. A round always has shortcut -> edge atomics -> check
// -> full hierarchical reduction -> commit dispatch boundaries. Host swaps
// label bindings only AFTER commit, advances accepted cursor once, and reads
// one 16B control record after <=4 complete rounds. It must not install after
// exhaustion. At most B complete rounds (including unchanged confirmation)
// suffice for any valid graph; this is a linear bound, not a log bound.
// A control.changed==0 round freezes subsequent rounds in the same window:
// shortcut copies Old, edge kernel skips, commit preserves completed count.
// Zero body extent: skip CC, seed checked zero reduction totals, validate refs.
// Both label buffers must be different allocations (never alias an arena field).
// Slang declares both as RWStructuredBuffer<Atomic<uint>>. Atomic wrapper layout
// delegates to uint (size/stride 4); all accesses are explicit load/store/min/or,
// default MemoryOrder.Relaxed, with real dispatch dependencies for visibility.
// Reflected storage access remains read_write even in load-only label entries.
// Do not bind an atomic label array as WGSL storage read-only or use a UInt4
// field as its backing storage. CPU POD remains scalar uint32, not a C++ atomic.

#ifdef __cplusplus
static_assert(sizeof(ConstraintScheduleWord) == 16);
static_assert(alignof(ConstraintScheduleWord) == 16);
static_assert(std::is_trivially_copyable_v<ConstraintScheduleWord>);
static_assert(sizeof(ConstraintParams) == 14 * 16);
static_assert(alignof(ConstraintParams) == 16);
} }
#endif
#undef CS_ALIGN
#undef CS_U4
#undef CS_CONST
#endif
