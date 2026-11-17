// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

// Host validates all word ranges, nonoverlap, domains and frozen bindings.
#include "../BroadphaseTypes.h"
#include "key_sort.slang"

ConstantBuffer<BroadphaseParams> broadphase;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<WorldRange> author_worlds;
StructuredBuffer<Aabb> bounds;
StructuredBuffer<QueryRecordWord> joint_stream;
StructuredBuffer<ColliderPairKey> collider_pair_ignores;
RWStructuredBuffer<WorldRange> runtime_worlds;
RWStructuredBuffer<ContactPatch> contact_patches;
// Separate from the writable arena: indirect consumers never bind this as
// writable storage. Each direct producer completely writes its used records.
RWStructuredBuffer<QueryRecordWord> indirect_arguments;
RWStructuredBuffer<QueryRecordWord> resident_step;
RWStructuredBuffer<Contact> resident_contacts;
RWStructuredBuffer<ContactPatch> resident_patches;

static const uint BP_BAD_BOUNDS = 1u;
static const uint BP_SUM_OVERFLOW = 2u;
static const uint BP_BAD_MODEL = 4u;
static const uint BP_BAD_CURSOR = 8u;
static const uint BP_FILL_MISMATCH = 16u;
static const uint BP_INDEX_OVERFLOW = 32u;
static const uint BP_BAD_QUANTUM = 64u;
static const uint BP_CAPACITY_NEEDED = 128u;

uint4 bp_load(uint p) { return contact_patches[p].leaves; }
void bp_store(uint p, uint4 v) { contact_patches[p].leaves = v; }
uint bp_header() { return broadphase.workspace.y; }

bool bp_suffix_valid()
{
    uint4 header = bp_load(bp_header());
    return header.z == 0u && header.y == 0u && header.x <= broadphase.candidates.z;
}

uint2 bp_add64(uint2 a, uint2 b, out bool overflow)
{
    uint lo = a.x + b.x;
    uint carry = lo < a.x ? 1u : 0u;
    uint hi0 = a.y + b.y;
    uint hi = hi0 + carry;
    overflow = hi0 < a.y || hi < hi0;
    return uint2(lo, hi);
}

uint4 bp_combine(uint4 a, uint4 b)
{
    bool overflow;
    uint2 s = bp_add64(a.xy, b.xy, overflow);
    return uint4(s.x, s.y, a.z | b.z | (overflow ? BP_SUM_OVERFLOW : 0u), max(a.w, b.w));
}

bool bp_finite3(float3 v)
{
    return all((asuint(v) & uint3(0x7fffffffu)) < uint3(0x7f800000u));
}

bool bp_valid_bounds(Aabb a)
{
    return bp_finite3(a.minimum.xyz) && bp_finite3(a.maximum.xyz)
        && all(a.minimum.xyz <= a.maximum.xyz);
}

uint bp_ordered(float v)
{
    uint u = v == 0.0f ? 0u : asuint(v); // -0 and +0 share one key.
    return (u & 0x80000000u) != 0u ? ~u : (u ^ 0x80000000u);
}

bool bp_in_range(uint index, uint first, uint count)
{
    return index >= first && index - first < count;
}

bool bp_world_selected(uint world)
{
    return world == broadphase.selection.x;
}

bool bp_item(uint lane, uint length, out uint item)
{
    item = 0u;
    if (lane >= stage_work.range.y || stage_work.range.x >= length)
        return false;
    if (lane >= length - stage_work.range.x)
        return false;
    item = stage_work.range.x + lane;
    return true;
}

// Only for independent globally selected word domains. The host initializes
// every subsequently read header/scratch interval through exact finite chunks;
// an unwritten spare capacity/tail is never an initialized logical prefix.
[shader("compute")]
[numthreads(64, 1, 1)]
void init_words(uint3 tid : SV_DispatchThreadID)
{
    uint word;
    if (bp_item(tid.x, broadphase.workspace.x, word)) bp_store(word, uint4(0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void init_contacts(uint3 tid : SV_DispatchThreadID)
{
    uint item;
    uint points = broadphase.round.y != 0u ? 1u : MaxContactPointsPerPair;
    if (!bp_item(tid.x, max(1u, broadphase.candidates.z * points), item)) return;
    Contact empty;
    empty.meta = uint4(0u);
    empty.normalDepth = empty.localAnchorA = empty.localAnchorB = empty.impulse = float4(0.0f);
    resident_contacts[item] = empty;
}

// The CPU uploaded the other StepParams words. This producer modifies only
// solve.zw; the uniform consumer follows the real storage-write dependency.
[shader("compute")]
[numthreads(1, 1, 1)]
void publish_resident_count(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 header = bp_load(bp_header());
    header.z &= ~BP_CAPACITY_NEEDED;
    if (header.y == 0u && header.x > broadphase.candidates.z) header.z |= BP_CAPACITY_NEEDED;
    bp_store(bp_header(), header);
    uint4 solve = resident_step[1u].value;
    uint points = broadphase.round.y != 0u ? 1u : MaxContactPointsPerPair;
    solve.zw = header.z == 0u ? uint2(header.x * points, header.x) : uint2(0u);
    resident_step[1u].value = solve;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void init_candidates(uint3 tid : SV_DispatchThreadID)
{
    uint word;
    if (bp_item(tid.x, 1u + 2u * broadphase.candidates.z, word))
        resident_patches[word].leaves = uint4(0u);
}

uint4 bp_merge_load(uint word)
{
    return broadphase.mergeKey.x == 2u ? resident_patches[word].leaves : bp_load(word);
}

void bp_merge_store(uint word, uint4 value)
{
    if (broadphase.mergeKey.x == 2u) resident_patches[word].leaves = value;
    else bp_store(word, value);
}

uint bp_blocks(uint count)
{
    return count / BroadphaseBlockSize + (count % BroadphaseBlockSize != 0u ? 1u : 0u);
}

uint bp_active_level_length()
{
    uint length = bp_load(broadphase.activeState.x).x;
    // The host validates scan level <= 6. A positive length stays at one
    // through higher levels, carrying the aggregate to a fixed root.
    for (uint level = 0u; level < 6u; ++level)
    {
        if (level >= broadphase.activeState.z) break;
        length = bp_blocks(length);
    }
    return length;
}

uint bp_scan_length()
{
    if (broadphase.activeState.w == 1u) return bp_active_level_length();
    return broadphase.scan.w;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void seed_active(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint status = bp_load(bp_header()).z;
    uint count = broadphase.round.z == 1u && !bp_suffix_valid() ? 0u : broadphase.selectedCounts.y;
    bp_store(broadphase.activeState.x, uint4(count, 0u, status, 0u));
    bp_store(broadphase.activeState.y, uint4(0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prepare_indirect_arguments(uint3 tid : SV_DispatchThreadID)
{
    uint record = tid.x;
    if (record >= stage_work.range.y) return;
    bool candidates = stage_work.range.z == 1u;
    uint length = candidates ? (bp_suffix_valid() ? bp_load(bp_header()).x : 0u) : bp_active_level_length();
    uint limit = BroadphaseItemLimit;
    bool block_mode = stage_work.range.w == 1u;
    if (block_mode)
    {
        length = bp_blocks(length);
        limit = BroadphaseBlockLimit;
    }
    // Host admission checks recordCount <= 15, blockMode in [0,1], and
    // consumerFirst + (recordCount - 1) * limit without uint32 overflow.
    uint first = stage_work.range.x + record * limit;
    uint count = 0u;
    if (first < length) count = min(limit, length - first);
    uint groups = count;
    if (!block_mode) groups = bp_blocks(count);
    indirect_arguments[record].value = uint4(groups, 1u, 1u, 0u);
}

uint4 bp_sort_record(uint leaf)
{
    ShapeModel shape = shape_models[leaf];
    uint status = 0u;
    if (shape.surface.x >= 32u) status |= BP_BAD_MODEL;
    uint world = 0xffffffffu;
    if (shape.meta.y >= broadphase.counts.w) status |= BP_BAD_MODEL;
    else
    {
        BodyModel body = body_models[shape.meta.y];
        world = body.meta.w;
        if (world >= broadphase.counts.z || !bp_world_selected(world)) status |= BP_BAD_MODEL;
        else
        {
            WorldRange owner = author_worlds[world];
            if (!bp_in_range(leaf, owner.leaves.x, owner.leaves.y)
                || !bp_in_range(shape.meta.y, owner.bodies.x, owner.bodies.y)
                || !bp_in_range(leaf, body.meta.x, body.meta.y)) status |= BP_BAD_MODEL;
        }
    }
    Aabb box = bounds[leaf];
    if (!bp_valid_bounds(box)) status |= BP_BAD_BOUNDS;
    // Invalid records still get an initialized, unique key and an explicit
    // failure. They never silently become a successful zero-candidate result.
    uint min_x = (status & BP_BAD_BOUNDS) == 0u ? bp_ordered(box.minimum.x) : 0u;
    return uint4(world, min_x, leaf, status);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_sort_records(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    uint item;
    bool valid = bp_item(tid.x, broadphase.selectedCounts.y, item);
    uint4 record = uint4(0xffffffffu);
    if (valid) record = bp_sort_record(broadphase.selection.z + item);
    record = jph_sort_tile(lane, record);
    if (valid) bp_store(broadphase.merge.x + item, record);
}

uint4 bp_suppression_record(uint joint, out uint status)
{
    uint p = 1u + JointDirectoryWords * joint;
    uint4 endpoints = joint_stream[p].value;
    uint world = joint_stream[p + 1u].value.z;
    status = 0u;
    bool suppress = (endpoints.w & (JointEnabled | JointCollisionEnabled)) == JointEnabled;
    // World endpoints cannot match any two real bodies, exactly as before.
    suppress = suppress && endpoints.x != InvalidQueryBodySlot;
    if (world >= broadphase.counts.z || !bp_world_selected(world) || endpoints.y >= broadphase.counts.w
        || (endpoints.x != InvalidQueryBodySlot && endpoints.x >= broadphase.counts.w))
        status |= BP_BAD_MODEL;
    else
    {
        WorldRange owner = author_worlds[world];
        if (!bp_in_range(joint, owner.leaves.z, owner.leaves.w)
            || body_models[endpoints.y].meta.w != world
            || (endpoints.x != InvalidQueryBodySlot && body_models[endpoints.x].meta.w != world))
            status |= BP_BAD_MODEL;
    }
    if (suppress && status == 0u)
        return uint4(world, min(endpoints.x, endpoints.y), max(endpoints.x, endpoints.y), joint);
    return uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, joint);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_suppression(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    uint item;
    bool valid = bp_item(tid.x, broadphase.selectedCounts.z, item);
    uint4 key = uint4(0xffffffffu);
    if (valid)
    {
        uint status;
        key = bp_suppression_record(broadphase.selection.w + item, status);
        bp_store(broadphase.scan.x + item, uint4(0u, 0u, status, 0u));
    }
    key = jph_sort_tile(lane, key);
    if (valid) bp_store(broadphase.merge.x + item, key);
}

bool bp_less(uint4 a, uint4 b)
{
    return broadphase.mergeKey.x == 2u ? jph_key_less(a.wxyz, b.wxyz) : jph_key_less(a, b);
}

// Filled pairs are sorted by (world, first leaf, second leaf). The input and
// output of a block are the same disjoint range of the accepted candidate bank.
[shader("compute")]
[numthreads(64, 1, 1)]
void sort_candidate_tiles(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    uint item;
    uint length = bp_suffix_valid() ? bp_load(bp_header()).x : 0u;
    bool valid = bp_item(tid.x, length, item);
    uint4 key = uint4(0xffffffffu);
    if (valid) key = resident_patches[broadphase.merge.x + item].leaves.wxyz;
    key = jph_sort_tile(lane, key);
    if (valid) resident_patches[broadphase.merge.x + item].leaves = key.yzwx;
}

// One stable merge destination per source item. Left items use lower_bound,
// right items upper_bound, so even equal keys cannot collide in the scatter.
[shader("compute")]
[numthreads(64, 1, 1)]
void merge_items(uint3 tid : SV_DispatchThreadID)
{
    uint length = broadphase.mergeKey.x == 2u ? (bp_suffix_valid() ? bp_load(bp_header()).x : 0u) : broadphase.merge.z;
    uint item;
    if (!bp_item(tid.x, length, item)) return;
    uint width = broadphase.merge.w; // Host validates power-of-two and <= 2^31.
    uint run = item / width;
    uint own_begin = run * width;
    uint own_end = own_begin + min(width, length - own_begin);
    bool right = (run & 1u) != 0u;
    uint other_begin = right ? own_begin - width : own_end;
    uint other_end = right ? own_begin : other_begin + min(width, length - other_begin);
    uint merged_begin = right ? other_begin : own_begin;
    uint4 key = bp_merge_load(broadphase.merge.x + item);
    uint low = other_begin, high = other_end;
    for (uint step = 0u; step < 32u && low < high; ++step)
    {
        uint middle = low + (high - low) / 2u;
        uint4 other = bp_merge_load(broadphase.merge.x + middle);
        bool before = right ? !bp_less(key, other) : bp_less(other, key);
        if (before) low = middle + 1u; else high = middle;
    }
    uint destination = merged_begin + (item - own_begin) + (low - other_begin);
    bp_merge_store(broadphase.merge.y + destination, key);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void find_sap_ends(uint3 tid : SV_DispatchThreadID)
{
    uint rank;
    if (!bp_item(tid.x, broadphase.selectedCounts.y, rank)) return;
    uint4 key = bp_load(broadphase.workspace.z + rank);
    uint status = key.w;
    uint end = rank + 1u;
    if (key.x >= broadphase.counts.z || key.z >= broadphase.counts.x) status |= BP_BAD_MODEL;
    if (status == 0u)
    {
        uint max_x = bp_ordered(bounds[key.z].maximum.x);
        uint low = rank + 1u, high = broadphase.selectedCounts.y;
        for (uint step = 0u; step < 32u && low < high; ++step)
        {
            uint middle = low + (high - low) / 2u;
            uint4 other = bp_load(broadphase.workspace.z + middle);
            if (other.x == key.x && other.y <= max_x) low = middle + 1u;
            else high = middle;
        }
        end = low;
    }
    bp_store(broadphase.rankRows.x + rank, uint4(end, rank + 1u, 0u, 0u));
    bp_store(broadphase.rankRows.y + rank, uint4(0u, 0u, status, 0u));
    // The initial active prefix includes empty spans within the selected World.
    bp_store(broadphase.activeQueues.x + rank, uint4(rank, 0u, 0u, 0u));
}

bool bp_suppressed(uint world, uint body_a, uint body_b)
{
    uint a = min(body_a, body_b), b = max(body_a, body_b);
    uint low = 0u, high = broadphase.selectedCounts.z;
    for (uint step = 0u; step < 32u && low < high; ++step)
    {
        uint middle = low + (high - low) / 2u;
        uint4 key = bp_load(broadphase.workspace.w + middle);
        bool before = key.x < world || (key.x == world &&
            (key.y < a || (key.y == a && key.z < b)));
        if (before) low = middle + 1u; else high = middle;
    }
    if (low == broadphase.selectedCounts.z) return false;
    uint4 found = bp_load(broadphase.workspace.w + low);
    // Duplicates retain sourceJointOrdinal in the full sort key. Any matching
    // suppressing joint wins, preserving the original multi-joint OR rule.
    return found.x == world && found.y == a && found.z == b;
}

bool bp_collider_suppressed(uint4 a, uint4 b)
{
    ColliderPairKey key = jph_collider_pair_key(a, b);
    uint low = 0u, high = broadphase.selectedCounts.w;
    for (uint step = 0u; step < 32u && low < high; ++step) {
        uint middle = low + (high - low) / 2u;
        if (jph_collider_pair_less(collider_pair_ignores[middle], key)) low = middle + 1u;
        else high = middle;
    }
    if (low >= broadphase.selectedCounts.w) return false;
    return jph_collider_pair_equal(collider_pair_ignores[low], key);
}

// Count and fill call this exact function against the same frozen buffers.
groupshared uint bp_small_leaf_status[BroadphaseSmallSetLimit];
groupshared uint4 bp_small_joint_keys[BroadphaseSmallSetLimit];
groupshared uint bp_small_pair_masks[BroadphaseSmallSetLimit];

bool bp_small_suppressed(uint world, uint a, uint b)
{
    uint2 pair = uint2(min(a, b), max(a, b));
    for (uint i = 0u; i < broadphase.selectedCounts.z; ++i)
    {
        uint4 key = bp_small_joint_keys[i];
        if (key.x == world && all(key.yz == pair)) return true;
    }
    return false;
}

// AABBs already include the existing half speculative-band expansion.
bool bp_eligible_leaves(uint first, uint second, uint world, bool small)
{
    Aabb ba = bounds[first], bb = bounds[second];
    if (any(ba.minimum.xyz > bb.maximum.xyz) || any(bb.minimum.xyz > ba.maximum.xyz)) return false;
    ShapeModel a = shape_models[first], b = shape_models[second];
    if (a.meta.y == b.meta.y) return false;
    BodyModel ma = body_models[a.meta.y], mb = body_models[b.meta.y];
    if ((ma.meta.z & MotionNotResident) != 0u || (mb.meta.z & MotionNotResident) != 0u) return false;
    bool sensor = ((a.meta.w | b.meta.w) & ShapeSensor) != 0u;
    if (broadphase.round.y != 0u && (sensor ||
        !((ba.minimum.w > 0.0f && bb.minimum.w == 0.0f) || (bb.minimum.w > 0.0f && ba.minimum.w == 0.0f))))
        return false;
    if (!sensor && (ma.meta.z & ~MotionNotResident) != MotionDynamic
        && (mb.meta.z & ~MotionNotResident) != MotionDynamic) return false;
    if ((a.surface.y & (1u << b.surface.x)) == 0u
        || (b.surface.y & (1u << a.surface.x)) == 0u) return false;
    if (small)
    {
        if (bp_small_suppressed(world, a.meta.y, b.meta.y)) return false;
    }
    else if (bp_suppressed(world, a.meta.y, b.meta.y)) return false;
    return !bp_collider_suppressed(a.source, b.source);
}

// A finite leaf set fits one group and one pair mask per row. Lane-order
// prefix placement yields the same (leafA, leafB) order as the global sort.
[shader("compute")]
[numthreads(64, 1, 1)]
void build_small_pairs(uint3 local : SV_GroupThreadID)
{
    uint lane = local.x;
    uint first = broadphase.selection.z;
    uint leaves = broadphase.selectedCounts.y;
    uint joints = broadphase.selectedCounts.z;
    uint world = broadphase.selection.x;
    if (lane < leaves)
        bp_small_leaf_status[lane] = bp_sort_record(first + lane).w;
    if (lane < joints)
    {
        uint status;
        uint4 key = bp_suppression_record(broadphase.selection.w + lane, status);
        key.w = status;
        bp_small_joint_keys[lane] = key;
    }
    GroupMemoryBarrierWithGroupSync();

    if (lane < leaves)
    {
        uint mask = 0u;
        if (bp_small_leaf_status[lane] == 0u)
        {
            for (uint other = lane + 1u; other < leaves; ++other)
            {
                if (bp_small_leaf_status[other] != 0u) continue;
                if (bp_eligible_leaves(first + lane, first + other, world, true))
                    mask |= 1u << other;
            }
        }
        bp_small_pair_masks[lane] = mask;
    }
    GroupMemoryBarrierWithGroupSync();

    if (lane < leaves)
    {
        uint offset = broadphase.candidates.x;
        for (uint earlier = 0u; earlier < lane; ++earlier)
            offset += countbits(bp_small_pair_masks[earlier]);
        uint mask = bp_small_pair_masks[lane];
        for (uint other = lane + 1u; other < leaves; ++other)
            if ((mask & (1u << other)) != 0u)
                resident_patches[offset++].leaves = uint4(first + lane, first + other, 0u, world);
    }
    if (lane == 0u)
    {
        uint count = 0u, status = 0u;
        for (uint i = 0u; i < leaves; ++i)
        {
            count += countbits(bp_small_pair_masks[i]);
            status |= bp_small_leaf_status[i];
        }
        for (uint i = 0u; i < joints; ++i)
            status |= bp_small_joint_keys[i].w;
        bp_store(bp_header(), uint4(count, 0u, status, count));
        resident_patches[0u].leaves = uint4(broadphase.candidates.x, status == 0u ? count : 0u, 0u, 0u);
    }
}

void bp_round(uint lane, bool fill)
{
    uint ordinal;
    uint active_count = bp_load(broadphase.activeState.x).x;
    if (!bp_item(lane, active_count, ordinal)) return;
    uint rank = bp_load(broadphase.activeQueues.x + ordinal).x;
    // Host proves active queue uniqueness and rank<N. Handle corrupt rank
    // locally without following it into a storage access.
    if (rank >= broadphase.selectedCounts.y)
    {
        bp_store(broadphase.scan.x + ordinal, uint4(0u, 0u, BP_BAD_CURSOR, 0u));
        return;
    }
    uint4 row = bp_load(broadphase.rankRows.x + rank);
    uint4 offset_status = bp_load(broadphase.rankRows.y + rank);
    uint status = offset_status.z;
    uint quantum = broadphase.round.x;
    if (quantum == 0u || quantum > BroadphaseNeighborLimit) status |= BP_BAD_QUANTUM;
    if (row.y < rank + 1u || row.y > row.x || row.x > broadphase.selectedCounts.y) status |= BP_BAD_CURSOR;
    uint end = row.y;
    if (status == 0u) end += min(quantum, row.x - row.y);
    uint4 key_a = bp_load(broadphase.workspace.z + rank);
    for (uint other = row.y; other < end; ++other)
    {
        uint4 key_b = bp_load(broadphase.workspace.z + other);
        if (key_b.w != 0u || key_b.x != key_a.x) { status |= BP_BAD_MODEL; continue; }
        if (!bp_eligible_leaves(key_a.z, key_b.z, key_a.x, false)) continue;
        if (fill)
        {
            bool overflow;
            uint2 destination = bp_add64(offset_status.xy, uint2(row.w, 0u), overflow);
            if (overflow || destination.y != 0u || destination.x >= broadphase.candidates.z || row.w >= row.z)
                status |= BP_FILL_MISMATCH;
            else
                resident_patches[broadphase.candidates.x + destination.x].leaves =
                    uint4(min(key_a.z, key_b.z), max(key_a.z, key_b.z), 0u, key_a.x);
            if (row.w == 0xffffffffu) status |= BP_SUM_OVERFLOW; else ++row.w;
        }
        else
        {
            if (row.z == 0xffffffffu) status |= BP_SUM_OVERFLOW; else ++row.z;
        }
    }
    row.y = end;
    offset_status.z = status;
    bp_store(broadphase.rankRows.x + rank, row);
    bp_store(broadphase.rankRows.y + rank, offset_status);
    // Failure makes this rank inactive, but its status is part of this exact
    // round's scanned prefix and is accumulated before the host continues.
    uint active = status == 0u && row.y < row.x ? 1u : 0u;
    bp_store(broadphase.scan.x + ordinal, uint4(active, 0u, status, active != 0u ? row.x - row.y : 0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void count_round(uint3 tid : SV_DispatchThreadID) { bp_round(tid.x, false); }
[shader("compute")]
[numthreads(64, 1, 1)]
void fill_round(uint3 tid : SV_DispatchThreadID) { bp_round(tid.x, true); }

groupshared uint4 bp_shared[BroadphaseBlockSize];

// For these two entries only, StageWork item first/count means BLOCKS.
// Every group participates in all barriers, including the masked final tail.
[shader("compute")]
[numthreads(64, 1, 1)]
void scan_blocks(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint length = bp_scan_length();
    uint block_count = bp_blocks(length);
    uint block;
    // The active length can come from storage. Mask reads and writes without
    // using that value to control whether a lane reaches a barrier.
    bool active_block = bp_item(gid.x, block_count, block);
    uint first = block * BroadphaseBlockSize;
    bool valid = active_block && lane < length - first;
    uint4 input = uint4(0u);
    if (valid) input = bp_load(broadphase.scan.x + first + lane);
    bp_shared[lane] = input;
    GroupMemoryBarrierWithGroupSync();
    for (uint distance = 1u; distance < BroadphaseBlockSize; distance *= 2u)
    {
        uint4 value = bp_shared[lane];
        uint4 previous = uint4(0u);
        if (lane >= distance) previous = bp_shared[lane - distance];
        GroupMemoryBarrierWithGroupSync();
        bp_shared[lane] = bp_combine(previous, value);
        GroupMemoryBarrierWithGroupSync();
    }
    if (valid)
    {
        uint4 exclusive = uint4(0u);
        if (lane != 0u) exclusive = bp_shared[lane - 1u];
        bp_store(broadphase.scan.y + first + lane, exclusive);
    }
    if (active_block && lane == BroadphaseBlockSize - 1u)
        bp_store(broadphase.scan.z + block, bp_shared[BroadphaseBlockSize - 1u]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reduce_blocks(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint length = bp_scan_length();
    uint block_count = bp_blocks(length);
    uint block;
    bool active_block = bp_item(gid.x, block_count, block);
    uint first = block * BroadphaseBlockSize;
    bool valid = active_block && lane < length - first;
    uint4 input = uint4(0u);
    if (valid) input = bp_load(broadphase.scan.x + first + lane);
    bp_shared[lane] = input;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = BroadphaseBlockSize / 2u; stride != 0u; stride /= 2u)
    {
        if (lane < stride) bp_shared[lane] = bp_combine(bp_shared[lane], bp_shared[lane + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (active_block && lane == 0u) bp_store(broadphase.scan.z + block, bp_shared[0u]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void add_scan_offsets(uint3 tid : SV_DispatchThreadID)
{
    uint item;
    uint length = bp_scan_length();
    if (!bp_item(tid.x, length, item)) return;
    uint4 parent = bp_load(broadphase.scanParent.x + item / BroadphaseBlockSize);
    bp_store(broadphase.scan.y + item, bp_combine(parent, bp_load(broadphase.scan.y + item)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void compact_active(uint3 tid : SV_DispatchThreadID)
{
    uint item;
    uint active_count = bp_load(broadphase.activeState.x).x;
    if (!bp_item(tid.x, active_count, item)) return;
    uint4 flag = bp_load(broadphase.scan.x + item);
    if (flag.x == 0u) return;
    uint4 prefix = bp_load(broadphase.scan.y + item);
    // Prefix high must be zero because each input is 0/1 and count<=N.
    // No capacity tail or target queue suffix is read by this entry.
    bp_store(broadphase.activeQueues.y + prefix.x, bp_load(broadphase.activeQueues.x + item));
}

// Dedicated fixed-one-record publication entries. Dispatch exactly one group;
// only lane zero accesses the records. summary.x points to a fully reduced
// record, or to a zero initialized record for an empty domain.
[shader("compute")]
[numthreads(1, 1, 1)]
void accumulate_status_summary(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 header = bp_load(bp_header());
    header.z |= bp_load(broadphase.summary.x).z;
    bp_store(bp_header(), header);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void publish_active_summary(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 input = bp_load(broadphase.activeState.x);
    uint4 header = bp_load(bp_header());
    uint4 total = uint4(0u);
    // Zero indirect groups have not produced a root for this window. Never
    // load the stale root, even though its memory was initialized earlier.
    if (input.x != 0u) total = bp_load(broadphase.summary.x);
    uint status = input.z | header.z | total.z;
    if (total.y != 0u) status |= BP_INDEX_OVERFLOW;
    if (total.x > input.x)
        status |= BP_BAD_CURSOR;
    header.z = status;
    bp_store(bp_header(), header);
    bp_store(broadphase.activeState.y, uint4(total.x, total.w, status, 0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void materialize_counts(uint3 tid : SV_DispatchThreadID)
{
    uint rank;
    if (!bp_item(tid.x, broadphase.selectedCounts.y, rank)) return;
    uint4 row = bp_load(broadphase.rankRows.x + rank);
    uint status = bp_load(broadphase.rankRows.y + rank).z;
    if (row.y != row.x) status |= BP_BAD_CURSOR;
    bp_store(broadphase.scan.x + rank, uint4(row.z, 0u, status, 0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void store_rank_offsets(uint3 tid : SV_DispatchThreadID)
{
    uint rank;
    if (!bp_item(tid.x, broadphase.selectedCounts.y, rank)) return;
    uint4 prefix = bp_load(broadphase.scan.y + rank);
    uint4 row = bp_load(broadphase.rankRows.y + rank);
    bp_store(broadphase.rankRows.y + rank, uint4(prefix.x, prefix.y, row.z | prefix.z, 0u));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void publish_count_summary(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 total = bp_load(broadphase.summary.x);
    uint4 header = bp_load(bp_header());
    // Keep the exact required C even when it cannot fit downstream indices.
    uint status = header.z | total.z;
    if (total.y != 0u || total.x > 0xffffffffu / MaxContactPointsPerPair) status |= BP_INDEX_OVERFLOW;
    bp_store(bp_header(), uint4(total.x, total.y, status, total.x));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_world_ranges(uint3 tid : SV_DispatchThreadID)
{
    uint world;
    if (!bp_item(tid.x, broadphase.counts.z, world)) return;
    WorldRange runtime = author_worlds[world];
    runtime.bodies.zw = uint2(0u, bp_world_selected(world) ? bp_load(bp_header()).x : 0u);
    runtime_worlds[world] = runtime;
}

// Preserve the count and offset for each rank and reset its cursor for fill.
// Both sweeps start from the same selected-world rank prefix.
[shader("compute")]
[numthreads(64, 1, 1)]
void reset_fill_ranks(uint3 tid : SV_DispatchThreadID)
{
    uint rank;
    if (!bp_suffix_valid() || !bp_item(tid.x, broadphase.selectedCounts.y, rank)) return;
    uint4 row = bp_load(broadphase.rankRows.x + rank);
    row.y = rank + 1u;
    row.w = 0u;
    bp_store(broadphase.rankRows.x + rank, row);
    bp_store(broadphase.activeQueues.x + rank, uint4(rank, 0u, 0u, 0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void check_fill(uint3 tid : SV_DispatchThreadID)
{
    uint rank;
    if (!bp_item(tid.x, broadphase.selectedCounts.y, rank)) return;
    uint4 row = bp_load(broadphase.rankRows.x + rank);
    uint status = bp_load(broadphase.rankRows.y + rank).z;
    if (!bp_suffix_valid()) { bp_store(broadphase.scan.x + rank, uint4(0u)); return; }
    if (row.y != row.x || row.w != row.z) status |= BP_FILL_MISMATCH;
    bp_store(broadphase.scan.x + rank, uint4(row.w, 0u, status, 0u));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void publish_fill_summary(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    if (!bp_suffix_valid()) return;
    uint4 emitted = bp_load(broadphase.summary.x);
    uint4 header = bp_load(bp_header());
    header.z |= emitted.z;
    if (emitted.x != header.x || emitted.y != header.y) header.z |= BP_FILL_MISMATCH;
    bp_store(bp_header(), header);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void set_candidate_bank(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 header = bp_load(bp_header());
    uint status = header.z;
    if ((status & BP_CAPACITY_NEEDED) == 0u && (header.y != 0u || header.x > broadphase.candidates.z)) status |= BP_FILL_MISMATCH;
    header.z = status;
    bp_store(bp_header(), header);
    // Host calls only after every canonical-sort chunk has completed.
    resident_patches[0u].leaves = uint4(broadphase.merge.x, status == 0u ? header.x : 0u, 0u, 0u);
}
