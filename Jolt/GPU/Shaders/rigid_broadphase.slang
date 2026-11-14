// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

// Host validates all word ranges, nonoverlap, domains and frozen bindings.
#include "../BroadphaseTypes.h"

ConstantBuffer<BroadphaseParams> broadphase;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<WorldRange> author_worlds;
StructuredBuffer<Aabb> bounds;
StructuredBuffer<QueryRecordWord> joint_stream;
RWStructuredBuffer<WorldRange> runtime_worlds;
RWStructuredBuffer<ContactPatch> contact_patches;
// Separate from the writable arena: indirect consumers never bind this as
// writable storage. Each direct producer completely writes its used records.
RWStructuredBuffer<QueryRecordWord> indirect_arguments;

static const uint BP_BAD_BOUNDS = 1u;
static const uint BP_SUM_OVERFLOW = 2u;
static const uint BP_BAD_MODEL = 4u;
static const uint BP_BAD_CURSOR = 8u;
static const uint BP_FILL_MISMATCH = 16u;
static const uint BP_INDEX_OVERFLOW = 32u;
static const uint BP_BAD_QUANTUM = 64u;

uint4 bp_load(uint p) { return contact_patches[p].leaves; }
void bp_store(uint p, uint4 v) { contact_patches[p].leaves = v; }
uint bp_h0() { return broadphase.workspace.y; }
uint bp_h1() { return broadphase.workspace.y + 1u; }

uint2 bp_add64(uint2 a, uint2 b, out bool overflow)
{
    uint lo = a.x + b.x;
    uint carry = lo < a.x ? 1u : 0u;
    uint hi0 = a.y + b.y;
    uint hi = hi0 + carry;
    overflow = hi0 < a.y || hi < hi0;
    return uint2(lo, hi);
}

uint2 bp_sub64(uint2 a, uint2 b, out bool underflow)
{
    uint borrow = a.x < b.x ? 1u : 0u;
    underflow = a.y < b.y || (a.y == b.y && borrow != 0u);
    return uint2(a.x - b.x, a.y - b.y - borrow);
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
    uint status = bp_load(bp_h1()).z;
    bp_store(broadphase.activeState.x, uint4(broadphase.selectedCounts.y, 0u, status, 0u));
    bp_store(broadphase.activeState.y, uint4(0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prepare_indirect_arguments(uint3 tid : SV_DispatchThreadID)
{
    uint record = tid.x;
    if (record >= stage_work.range.y) return;
    uint length = bp_active_level_length();
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

[shader("compute")]
[numthreads(64, 1, 1)]
void build_sort_records(uint3 tid : SV_DispatchThreadID)
{
    uint leaf;
    if (!bp_item(tid.x, broadphase.counts.x, leaf)) return;
    if (!bp_in_range(leaf, broadphase.selection.z, broadphase.selectedCounts.y))
    {
        bp_store(broadphase.merge.x + leaf, uint4(0xffffffffu, 0xffffffffu, leaf, 0u));
        return;
    }
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
    bp_store(broadphase.merge.x + leaf, uint4(world, min_x, leaf, status));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_suppression(uint3 tid : SV_DispatchThreadID)
{
    uint joint;
    if (!bp_item(tid.x, broadphase.counts.y, joint)) return;
    if (!bp_in_range(joint, broadphase.selection.w, broadphase.selectedCounts.z))
    {
        bp_store(broadphase.merge.x + joint, uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, joint));
        bp_store(broadphase.rankRows.z + joint, uint4(0u));
        return;
    }
    uint p = 1u + JointAuthorWords * joint;
    uint4 endpoints = joint_stream[p].value;
    uint world = joint_stream[p + 1u].value.z;
    uint status = 0u;
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
    uint4 key = uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, joint);
    if (suppress && status == 0u)
        key = uint4(world, min(endpoints.x, endpoints.y), max(endpoints.x, endpoints.y), joint);
    bp_store(broadphase.merge.x + joint, key);
    bp_store(broadphase.rankRows.z + joint, uint4(0u, 0u, status, 0u));
}

bool bp_less(uint4 a, uint4 b)
{
    if (broadphase.mergeKey.x == 2u && a.w != b.w) return a.w < b.w;
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    if (broadphase.mergeKey.x == 2u) return false;
    if (a.z != b.z) return a.z < b.z;
    return broadphase.mergeKey.x == 1u && a.w < b.w;
}

// One stable merge destination per source item. Left items use lower_bound,
// right items upper_bound, so even equal keys cannot collide in the scatter.
[shader("compute")]
[numthreads(64, 1, 1)]
void merge_items(uint3 tid : SV_DispatchThreadID)
{
    uint item;
    if (!bp_item(tid.x, broadphase.merge.z, item)) return;
    uint width = broadphase.merge.w; // Host validates power-of-two and <= 2^31.
    uint run = item / width;
    uint own_begin = run * width;
    uint own_end = own_begin + min(width, broadphase.merge.z - own_begin);
    bool right = (run & 1u) != 0u;
    uint other_begin = right ? own_begin - width : own_end;
    uint other_end = right ? own_begin : other_begin + min(width, broadphase.merge.z - other_begin);
    uint merged_begin = right ? other_begin : own_begin;
    uint4 key = bp_load(broadphase.merge.x + item);
    uint low = other_begin, high = other_end;
    for (uint step = 0u; step < 32u && low < high; ++step)
    {
        uint middle = low + (high - low) / 2u;
        uint4 other = bp_load(broadphase.merge.x + middle);
        bool before = right ? !bp_less(key, other) : bp_less(other, key);
        if (before) low = middle + 1u; else high = middle;
    }
    uint destination = merged_begin + (item - own_begin) + (low - other_begin);
    bp_store(broadphase.merge.y + destination, key);
}

uint bp_world_bound(uint world, bool upper)
{
    uint low = 0u, high = broadphase.counts.x;
    for (uint step = 0u; step < 32u && low < high; ++step)
    {
        uint middle = low + (high - low) / 2u;
        uint value = bp_load(broadphase.workspace.z + middle).x;
        if (value < world || (upper && value == world)) low = middle + 1u; else high = middle;
    }
    return low;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void find_sap_ends(uint3 tid : SV_DispatchThreadID)
{
    uint rank;
    if (!bp_item(tid.x, broadphase.counts.x, rank)) return;
    uint4 key = bp_load(broadphase.workspace.z + rank);
    uint status = key.w;
    uint end = rank + 1u;
    if (key.x >= broadphase.counts.z || key.z >= broadphase.counts.x) status |= BP_BAD_MODEL;
    if (rank >= broadphase.selectedCounts.y)
    {
        bp_store(broadphase.rankRows.x + rank, uint4(end, end, 0u, 0u));
        bp_store(broadphase.rankRows.y + rank, uint4(0u));
        return;
    }
    if (status == 0u)
    {
        uint max_x = bp_ordered(bounds[key.z].maximum.x);
        uint low = rank + 1u, high = broadphase.counts.x;
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
    // The initial active prefix is all N, including empty spans, so every rank
    // contributes its status before stable compaction can remove it.
    bp_store(broadphase.activeQueues.x + rank, uint4(rank, 0u, 0u, 0u));
}

bool bp_suppressed(uint world, uint body_a, uint body_b)
{
    uint a = min(body_a, body_b), b = max(body_a, body_b);
    uint low = 0u, high = broadphase.counts.y;
    for (uint step = 0u; step < 32u && low < high; ++step)
    {
        uint middle = low + (high - low) / 2u;
        uint4 key = bp_load(broadphase.workspace.w + middle);
        bool before = key.x < world || (key.x == world &&
            (key.y < a || (key.y == a && key.z < b)));
        if (before) low = middle + 1u; else high = middle;
    }
    if (low == broadphase.counts.y) return false;
    uint4 found = bp_load(broadphase.workspace.w + low);
    // Duplicates retain sourceJointOrdinal in the full sort key. Any matching
    // suppressing joint wins, preserving the original multi-joint OR rule.
    return found.x == world && found.y == a && found.z == b;
}

// Count and fill call this exact function against the same frozen buffers.
// AABBs already include the existing half speculative-band expansion.
bool bp_eligible(uint rank_a, uint rank_b)
{
    uint4 key_a = bp_load(broadphase.workspace.z + rank_a);
    uint4 key_b = bp_load(broadphase.workspace.z + rank_b);
    if (!bp_world_selected(key_a.x) || key_a.x != key_b.x) return false;
    ShapeModel a = shape_models[key_a.z], b = shape_models[key_b.z];
    if (a.meta.y == b.meta.y) return false;
    BodyModel ma = body_models[a.meta.y], mb = body_models[b.meta.y];
    if ((ma.meta.z & MotionNotResident) != 0u || (mb.meta.z & MotionNotResident) != 0u) return false;
    bool sensor = ((a.meta.w | b.meta.w) & ShapeSensor) != 0u;
    if (!sensor && (ma.meta.z & ~MotionNotResident) != MotionDynamic
        && (mb.meta.z & ~MotionNotResident) != MotionDynamic) return false;
    if ((a.surface.y & (1u << b.surface.x)) == 0u
        || (b.surface.y & (1u << a.surface.x)) == 0u) return false;
    if (bp_suppressed(key_a.x, a.meta.y, b.meta.y)) return false;
    Aabb ba = bounds[key_a.z], bb = bounds[key_b.z];
    return !any(ba.minimum.xyz > bb.maximum.xyz) && !any(bb.minimum.xyz > ba.maximum.xyz);
}

void bp_round(uint lane, bool fill)
{
    uint ordinal;
    uint active_count = bp_load(broadphase.activeState.x).x;
    if (!bp_item(lane, active_count, ordinal)) return;
    uint rank = bp_load(broadphase.activeQueues.x + ordinal).x;
    // Host proves active queue uniqueness and rank<N. Handle corrupt rank
    // locally without following it into a storage access.
    if (rank >= broadphase.counts.x)
    {
        bp_store(broadphase.scan.x + ordinal, uint4(0u, 0u, BP_BAD_CURSOR, 0u));
        return;
    }
    uint4 row = bp_load(broadphase.rankRows.x + rank);
    uint4 offset_status = bp_load(broadphase.rankRows.y + rank);
    uint status = offset_status.z;
    uint quantum = broadphase.round.x;
    if (quantum == 0u || quantum > BroadphaseNeighborLimit) status |= BP_BAD_QUANTUM;
    if (row.y < rank + 1u || row.y > row.x || row.x > broadphase.counts.x) status |= BP_BAD_CURSOR;
    uint end = row.y;
    if (status == 0u) end += min(quantum, row.x - row.y);
    uint4 key_a = bp_load(broadphase.workspace.z + rank);
    for (uint other = row.y; other < end; ++other)
    {
        uint4 key_b = bp_load(broadphase.workspace.z + other);
        if (key_b.w != 0u || key_b.x != key_a.x) { status |= BP_BAD_MODEL; continue; }
        if (!bp_eligible(rank, other)) continue;
        if (fill)
        {
            bool overflow;
            uint2 destination = bp_add64(offset_status.xy, uint2(row.w, 0u), overflow);
            if (overflow || destination.y != 0u || destination.x >= broadphase.candidates.z || row.w >= row.z)
                status |= BP_FILL_MISMATCH;
            else
                bp_store(broadphase.candidates.x + destination.x,
                    uint4(min(key_a.z, key_b.z), max(key_a.z, key_b.z), 0u, key_a.x));
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
    bp_store(broadphase.scan.x + ordinal, uint4(active, 0u, status, 0u));
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
    if (!bp_item(gid.x, block_count, block)) return; // uniform across group.
    uint first = block * BroadphaseBlockSize;
    bool valid = lane < length - first;
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
    if (lane == BroadphaseBlockSize - 1u) bp_store(broadphase.scan.z + block, bp_shared[BroadphaseBlockSize - 1u]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reduce_blocks(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint length = bp_scan_length();
    uint block_count = bp_blocks(length);
    uint block;
    if (!bp_item(gid.x, block_count, block)) return;
    uint first = block * BroadphaseBlockSize;
    bool valid = lane < length - first;
    uint4 input = uint4(0u);
    if (valid) input = bp_load(broadphase.scan.x + first + lane);
    bp_shared[lane] = input;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = BroadphaseBlockSize / 2u; stride != 0u; stride /= 2u)
    {
        if (lane < stride) bp_shared[lane] = bp_combine(bp_shared[lane], bp_shared[lane + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) bp_store(broadphase.scan.z + block, bp_shared[0u]);
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
    uint4 header = bp_load(bp_h1());
    header.z |= bp_load(broadphase.summary.x).z;
    bp_store(bp_h1(), header);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void publish_active_summary(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 input = bp_load(broadphase.activeState.x);
    uint4 header = bp_load(bp_h1());
    uint4 total = uint4(0u);
    // Zero indirect groups have not produced a root for this window. Never
    // load the stale root, even though its memory was initialized earlier.
    if (input.x != 0u) total = bp_load(broadphase.summary.x);
    uint status = input.z | header.z | total.z;
    if (total.y != 0u) status |= BP_INDEX_OVERFLOW;
    if (input.x > broadphase.counts.x || total.x > input.x || total.x > broadphase.counts.x)
        status |= BP_BAD_CURSOR;
    uint completed_rounds = input.y;
    if (input.x != 0u)
    {
        uint passes = broadphase.round.y;
        if (passes == 0u || passes > 4u) status |= BP_BAD_QUANTUM;
        else if (passes > 0xffffffffu - completed_rounds) status |= BP_SUM_OVERFLOW;
        else completed_rounds += passes;
    }
    header.z = status;
    bp_store(bp_h1(), header); // Preserve C64 and maxWorldC, including during fill.
    bp_store(broadphase.activeState.y, uint4(total.x, completed_rounds, status, 0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void materialize_counts(uint3 tid : SV_DispatchThreadID)
{
    uint rank;
    if (!bp_item(tid.x, broadphase.counts.x, rank)) return;
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
    if (!bp_item(tid.x, broadphase.counts.x, rank)) return;
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
    uint4 header = bp_load(bp_h1());
    // Keep the exact required C even when it cannot fit downstream indices.
    uint status = header.z | total.z;
    if (total.y != 0u || total.x > 0xffffffffu / MaxContactPointsPerPair) status |= BP_INDEX_OVERFLOW;
    bp_store(bp_h1(), uint4(total.x, total.y, status, 0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_world_ranges(uint3 tid : SV_DispatchThreadID)
{
    uint world;
    if (!bp_item(tid.x, broadphase.counts.z, world)) return;
    WorldRange runtime = author_worlds[world];
    if (!bp_world_selected(world))
    {
        runtime.bodies.zw = uint2(0u);
        runtime_worlds[world] = runtime;
        bp_store(broadphase.summary.y + world, uint4(0u));
        return;
    }
    uint first = bp_world_bound(world, false), end = bp_world_bound(world, true);
    uint4 total = bp_load(bp_h1());
    uint2 begin_offset = total.xy;
    uint2 end_offset = total.xy;
    if (first != broadphase.counts.x) begin_offset = bp_load(broadphase.rankRows.y + first).xy;
    if (end != broadphase.counts.x) end_offset = bp_load(broadphase.rankRows.y + end).xy;
    bool underflow;
    uint2 count = bp_sub64(end_offset, begin_offset, underflow);
    uint status = total.z;
    if (underflow || begin_offset.y != 0u || count.y != 0u) status |= BP_INDEX_OVERFLOW;
    runtime.bodies.z = begin_offset.x;
    runtime.bodies.w = count.x;
    runtime_worlds[world] = runtime;
    bp_store(broadphase.summary.y + world, uint4(count.x, count.y, status, count.x));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void publish_world_summary(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 total = bp_load(broadphase.summary.x);
    uint4 header = bp_load(bp_h1());
    header.z |= total.z;
    if (total.x != header.x || total.y != header.y) header.z |= BP_BAD_MODEL;
    header.w = total.w;
    bp_store(bp_h1(), header);
}

// After capacity acceptance, preserve completed rank rows/offsets and reset only
// cursors/emitted. Fill has exactly the same N-rank initial active prefix.
[shader("compute")]
[numthreads(64, 1, 1)]
void reset_fill_ranks(uint3 tid : SV_DispatchThreadID)
{
    uint rank;
    if (!bp_item(tid.x, broadphase.counts.x, rank)) return;
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
    if (!bp_item(tid.x, broadphase.counts.x, rank)) return;
    uint4 row = bp_load(broadphase.rankRows.x + rank);
    uint status = bp_load(broadphase.rankRows.y + rank).z;
    if (row.y != row.x || row.w != row.z) status |= BP_FILL_MISMATCH;
    bp_store(broadphase.summary.z + rank, uint4(row.w, 0u, status, 0u));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void publish_fill_summary(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 emitted = bp_load(broadphase.summary.x);
    uint4 header = bp_load(bp_h1());
    header.z |= emitted.z;
    if (emitted.x != header.x || emitted.y != header.y) header.z |= BP_FILL_MISMATCH;
    bp_store(bp_h1(), header);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void set_candidate_bank(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 header = bp_load(bp_h1());
    uint status = header.z;
    if (header.y != 0u || header.x != broadphase.candidates.z) status |= BP_FILL_MISMATCH;
    header.z = status;
    bp_store(bp_h1(), header);
    // Host calls only after every canonical-sort chunk has completed.
    bp_store(bp_h0(), uint4(broadphase.merge.x, broadphase.candidates.z, 0u, 0u));
}
