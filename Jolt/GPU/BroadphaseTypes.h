// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

// Shared C++ / Slang POD; the host owns scheduling and allocation.
#ifndef JPH_GPU_BROADPHASE_TYPES_H
#define JPH_GPU_BROADPHASE_TYPES_H
#include "PhysicsTypes.h"
#ifdef __cplusplus
namespace JPH { namespace GPU {
#define BP_ALIGN alignas(16)
#define BP_U4 UInt4
#else
#define BP_ALIGN
#define BP_U4 uint4
#endif

#ifdef __cplusplus
inline constexpr std::uint32_t BroadphaseBlockSize = 64u;
inline constexpr std::uint32_t BroadphaseItemLimit = 2048u;
inline constexpr std::uint32_t BroadphaseBlockLimit = BroadphaseItemLimit / BroadphaseBlockSize;
inline constexpr std::uint32_t BroadphaseNeighborLimit = 128u;
#else
static const uint BroadphaseBlockSize = 64u;
static const uint BroadphaseItemLimit = 2048u;
static const uint BroadphaseBlockLimit = BroadphaseItemLimit / BroadphaseBlockSize;
static const uint BroadphaseNeighborLimit = 128u;
#endif

// All addresses below are ContactPatch / uint4 WORD addresses, not bytes,
// Contact ordinals, candidate ordinals, leaf ordinals or body slots.
struct BP_ALIGN BroadphaseParams
{
    BP_U4 counts;       // accepted leaf N, joint J, world W, body B.
    BP_U4 workspace;    // arena word capacity, 2-word header base, final SAP bank, final suppression bank.
    BP_U4 rankRows;     // N row0 base, N row1 base, J joint-status base, zero.
    BP_U4 activeQueues; // N source queue base, N target queue base, host-admitted source count bound, zero.
    BP_U4 round;        // Neighbor quantum [1,128], completed neighbor passes in this window [1,4], zero, zero.
    BP_U4 candidates;   // candidate bank0 base, bank1 base, accepted C capacity/count, zero.
    BP_U4 merge;        // source base, target base, valid length, run width [1,2^31].
    BP_U4 mergeKey;     // 0=SAP(world,minX,leaf), 1=suppression(world,bodyA,bodyB,joint), 2=pair(world,leafA,leafB); yzw zero.
    BP_U4 scan;         // input base, exclusive-prefix base, block-total base, valid input length.
    BP_U4 scanParent;   // globally scanned parent prefix base for add_scan_offsets; yzw zero.
    BP_U4 activeState;  // Input/output active-meta word bases, scan level [0,6], dynamic-scan flag [0,1].
    BP_U4 summary;      // final 1-record reduction address, world-summary W-record base, fill-check N-record base, zero.
    // One accepted layout address, never a stable RuntimeWorld identity.
    // selection=(world row, body base, leaf base, joint base).
    // selectedCounts=(body capacity, leaf count, joint count, Collider-pair ignore count).
    // Counts above remain TOTAL array extents for global address validation.
    BP_U4 selection;
    BP_U4 selectedCounts;
};

// 16-byte arena header0 = (candidateBankWordBase, actualC, 0, 0).
// header1 = (totalLo, totalHi, statusBits, maxWorldC after world summary).
// Active metadata is separate: (activeCount, completedNeighborRounds, statusBits, 0).
// A nonempty input window counts every completed pass, including empty passes
// after individual ranks reach end. A zero input window does not advance it.
// Input metadata stays immutable until this window completes and banks swap.
// StageWork for prepare_indirect_arguments = (consumerFirst, recordCount, 0,
// blockMode), recordCount <= 15, blockMode 0=item / 1=scan-block domain.
// The host checks every generated consumerFirst in uint64 before admission.
// Indirect records live in a separate allocation: (groupsX, 1, 1, 0), with
// exactly the first 12 bytes consumed as indirect arguments.
// N row0 = (exclusiveSapEnd, cursor, expectedCount, emittedCount).
// N row1 = (exclusiveCandidateOffsetLo, exclusiveCandidateOffsetHi, status, 0).
// All scan/reduce records = (sumLo,sumHi,statusBits,maxValue). The associative
// operator performs checked uint2 sum, bitwise OR, max. Sum overflow sets bit 1.
// Queue words = (rank,0,0,0); only accepted active prefix may be read.
// Pair words = (canonicalLeafA,canonicalLeafB,0,world) before narrowphase.
// Narrowphase overwrites patch flags and clears w. Keeping world in the sort
// key preserves runtime world segments even if author leaf ranges are not
// ordered by world index. Within each world the key is exactly (leafA,leafB).

#ifdef __cplusplus
static_assert(sizeof(BroadphaseParams) == 14 * 16);
static_assert(alignof(BroadphaseParams) == 16);
static_assert(sizeof(WorldRange) == 32);
} }
#endif
#undef BP_ALIGN
#undef BP_U4
#endif
