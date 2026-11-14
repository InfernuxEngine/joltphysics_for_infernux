// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT
#include "../ConstraintTypes.h"

ConstantBuffer<ConstraintParams> constraint_params;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<WorldRange> worlds;
StructuredBuffer<QueryRecordWord> joint_stream;
StructuredBuffer<ContactPatch> contact_patches;
StructuredBuffer<Contact> contacts;
// Cross-target scalar atomic resources. Atomic<uint> has uint's 4-byte layout;
// unlike HLSL Interlocked intrinsics on ordinary uint, its methods support WGSL.
// Every access uses the atomic wrapper, including dispatch-separated init/copy.
RWStructuredBuffer<Atomic<uint>> labels_old;
RWStructuredBuffer<Atomic<uint>> labels_next;
RWStructuredBuffer<ConstraintScheduleWord> constraint_schedule;

uint4 cs_load(uint p) { return constraint_schedule[p].value; }
void cs_store(uint p, uint4 v) { constraint_schedule[p].value = v; }
bool cs_item(uint lane, uint length, out uint item)
{
    item = 0u;
    if (lane >= stage_work.range.y || stage_work.range.x >= length) return false;
    if (lane >= length - stage_work.range.x) return false;
    item = stage_work.range.x + lane;
    return true;
}
bool cs_in_range(uint i, uint first, uint n) { return i >= first && i - first < n; }
bool cs_resident(BodyModel m) { return (m.meta.z & MotionNotResident) == 0u; }
bool cs_writer(BodyModel m)
{
    // Temporary-static effective model only becomes Kinematic, never Dynamic.
    return cs_resident(m) && (m.meta.z & ~MotionNotResident) == MotionDynamic;
}
bool cs_body(uint body, uint world)
{
    if (body >= constraint_params.counts.x || world >= constraint_params.counts.w) return false;
    BodyModel m = body_models[body];
    WorldRange w = worlds[world];
    return world == constraint_params.selection.x && m.meta.w == world
        && cs_in_range(body, constraint_params.selection.y, constraint_params.selectedCounts.x)
        && cs_in_range(body, w.bodies.x, w.bodies.y);
}
uint cs_ref_base(uint kind)
{
    return kind == ConstraintJoint ? constraint_params.references.x : constraint_params.references.y;
}
uint cs_ref_count(uint kind)
{
    return kind == ConstraintJoint ? constraint_params.counts.y : constraint_params.counts.z;
}
void cs_ref(uint kind, uint ordinal, uint world, uint a, uint b, bool active, uint status)
{
    uint p = cs_ref_base(kind) + 2u * ordinal;
    cs_store(p, uint4(world, a, b, ordinal));
    cs_store(p + 1u, uint4(kind, active && status == 0u ? 1u : 0u, status, 0u));
}
uint cs_blocks(uint n) { return n / ConstraintBlockSize + (n % ConstraintBlockSize != 0u ? 1u : 0u); }

// Exact chunks only. Host seeds every subsequently read empty-total / unused
// row with zero, including B range summaries, before writers populate prefixes.
[shader("compute")][numthreads(64, 1, 1)]
void constraint_init_words(uint3 tid : SV_DispatchThreadID)
{
    uint p;
    if (cs_item(tid.x, constraint_params.limits.x, p)) cs_store(p, uint4(0u));
}
[shader("compute")][numthreads(1, 1, 1)]
void constraint_seed(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint h = constraint_params.layout.x;
    cs_store(h, uint4(0u, 0u, 0u, ConstraintNotReady));
    cs_store(h + 1u, uint4(0u));
    cs_store(h + 2u, uint4(constraint_params.islands.x, constraint_params.sources.z,
        constraint_params.sources.w, constraint_params.sources.x));
    cs_store(h + 3u, uint4(constraint_params.sources.y, constraint_params.limits.x, 0u, 0u));
    cs_store(constraint_params.layout.y, uint4(constraint_params.counts.x != 0u ? 1u : 0u, 0u, 0u, 0u));
}

// Runs only after the complete SetupJoints dispatch sequence. It reads initialized
// author/runtime scalar metadata, never BodyState and never applies impulses.
[shader("compute")][numthreads(64, 1, 1)]
void constraint_build_joint_refs(uint3 tid : SV_DispatchThreadID)
{
    uint j;
    if (!cs_item(tid.x, constraint_params.counts.y, j)) return;
    if (!cs_in_range(j, constraint_params.selection.w, constraint_params.selectedCounts.z))
    {
        cs_ref(ConstraintJoint, j, ConstraintInvalid, ConstraintInvalid, ConstraintInvalid, false, 0u);
        return;
    }
    uint4 header = joint_stream[0].value;
    uint status = 0u;
    if (header.x != constraint_params.counts.y || header.y != 1u + JointAuthorWords * header.x)
    {
        cs_ref(ConstraintJoint, j, ConstraintInvalid, ConstraintInvalid, ConstraintInvalid, false, ConstraintBadJoint);
        return;
    }
    uint author = 1u + JointAuthorWords * j;
    uint4 e = joint_stream[author].value;
    uint world = joint_stream[author + 1u].value.z;
    uint4 state = joint_stream[header.y + JointStateWords * j].value;
    if (state.y != JointStatusSuccess || e.z > JointSwingTwist) status |= ConstraintBadJoint;
    if (world >= constraint_params.counts.w || !cs_body(e.y, world)
        || (e.x != InvalidQueryBodySlot && !cs_body(e.x, world))) status |= ConstraintBadModel;
    uint a = ConstraintInvalid, b = ConstraintInvalid;
    bool resident = false;
    if ((status & ConstraintBadModel) == 0u)
    {
        WorldRange w = worlds[world];
        if (!cs_in_range(j, w.leaves.z, w.leaves.w)) status |= ConstraintBadJoint;
        BodyModel mb = body_models[e.y];
        resident = cs_resident(mb);
        if (cs_writer(mb)) b = e.y;
        if (e.x != InvalidQueryBodySlot)
        {
            BodyModel ma = body_models[e.x];
            resident = resident && cs_resident(ma);
            if (cs_writer(ma)) a = e.x;
        }
    }
    bool active = resident && state.x != 0u && (e.w & JointEnabled) != 0u
        && (a != ConstraintInvalid || b != ConstraintInvalid);
    cs_ref(ConstraintJoint, j, world, a, b, active, status);
}

// Candidate count is an iteration extent ONLY. Actual responding patch flags
// and initialized Contact active prefix establish every live graph edge.
[shader("compute")][numthreads(64, 1, 1)]
void constraint_build_patch_refs(uint3 tid : SV_DispatchThreadID)
{
    uint p;
    if (!cs_item(tid.x, constraint_params.counts.z, p)) return;
    uint4 patch = contact_patches[contact_patches[0].leaves.x + p].leaves;
    uint status = 0u, world = ConstraintInvalid;
    uint a = ConstraintInvalid, b = ConstraintInvalid;
    bool responds = (patch.z & PatchActive) != 0u;
    bool patch_sensor = (patch.z & PatchSensor) != 0u;
    bool active = responds && !patch_sensor;
    if (responds && patch_sensor) status |= ConstraintBadManifold;
    if (patch.x >= constraint_params.limits.y || patch.y >= constraint_params.limits.y)
        status |= ConstraintBadModel;
    else
    {
        ShapeModel sa = shape_models[patch.x], sb = shape_models[patch.y];
        if (sa.meta.y >= constraint_params.counts.x || sb.meta.y >= constraint_params.counts.x)
            status |= ConstraintBadModel;
        else
        {
            BodyModel ma = body_models[sa.meta.y], mb = body_models[sb.meta.y];
            world = ma.meta.w;
            if (!cs_body(sa.meta.y, world) || !cs_body(sb.meta.y, world)) status |= ConstraintBadModel;
            else
            {
                WorldRange w = worlds[world];
                if (!cs_in_range(p, w.bodies.z, w.bodies.w)
                    || !cs_in_range(patch.x, w.leaves.x, w.leaves.y)
                    || !cs_in_range(patch.y, w.leaves.x, w.leaves.y)
                    || !cs_in_range(patch.x, ma.meta.x, ma.meta.y)
                    || !cs_in_range(patch.y, mb.meta.x, mb.meta.y)) status |= ConstraintBadModel;
            }
            if (cs_writer(ma)) a = sa.meta.y;
            if (cs_writer(mb)) b = sb.meta.y;
            bool sensor = ((sa.meta.w | sb.meta.w) & ShapeSensor) != 0u;
            if (sensor != patch_sensor || (responds && (sensor || !cs_resident(ma) || !cs_resident(mb))))
                status |= ConstraintBadManifold;
            bool saw_inactive = false;
            uint active_points = 0u;
            // Narrowphase initializes all four records, including inactive tail.
            for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
            {
                uint4 c = contacts[p * MaxContactPointsPerPair + point].meta;
                if (c.z != 0u)
                {
                    ++active_points;
                    if (!active || saw_inactive || c.x != sa.meta.y || c.y != sb.meta.y)
                        status |= ConstraintBadManifold;
                }
                else saw_inactive = true;
            }
            if (active && active_points == 0u) status |= ConstraintBadManifold;
        }
    }
    active = active && (a != ConstraintInvalid || b != ConstraintInvalid);
    cs_ref(ConstraintPatch, p, world, a, b, active, status);
}

[shader("compute")][numthreads(64, 1, 1)]
void constraint_init_labels(uint3 tid : SV_DispatchThreadID)
{
    uint v;
    if (!cs_item(tid.x, constraint_params.counts.x, v)) return;
    BodyModel m = body_models[v];
    labels_old[v].store(cs_writer(m) && cs_body(v, m.meta.w) ? v : ConstraintInvalid);
    labels_next[v].store(ConstraintInvalid);
}
bool cs_label(uint vertex, uint label)
{
    if (vertex >= constraint_params.counts.x || label >= constraint_params.counts.x) return false;
    BodyModel v = body_models[vertex], l = body_models[label];
    return cs_writer(v) && cs_writer(l) && v.meta.w == l.meta.w
        && cs_body(vertex, v.meta.w) && cs_body(label, l.meta.w) && label <= vertex;
}

[shader("compute")][numthreads(64, 1, 1)]
void constraint_shortcut_labels(uint3 tid : SV_DispatchThreadID)
{
    uint v;
    if (!cs_item(tid.x, constraint_params.counts.x, v)) return;
    uint l = labels_old[v].load(), next = ConstraintInvalid;
    if (cs_load(constraint_params.layout.y).x == 0u)
    {
        labels_next[v].store(l);
        return;
    }
    if (cs_label(v, l))
    {
        uint parent = labels_old[l].load();
        if (cs_label(l, parent)) next = parent;
    }
    // This dispatch has UNIQUE atomicStore writers to Next. It completes before
    // the next dispatch performs atomics, while Old remains immutable throughout.
    labels_next[v].store(next);
}

[shader("compute")][numthreads(64, 1, 1)]
void constraint_propagate_edges(uint3 tid : SV_DispatchThreadID)
{
    uint r, kind = constraint_params.select.x;
    if (!cs_item(tid.x, cs_ref_count(kind), r)) return;
    if (cs_load(constraint_params.layout.y).x == 0u) return;
    uint p = cs_ref_base(kind) + 2u * r;
    if (cs_load(p + 1u).y == 0u) return;
    uint4 e = cs_load(p);
    if (e.y == ConstraintInvalid || e.z == ConstraintInvalid) return;
    uint la = labels_old[e.y].load(), lb = labels_old[e.z].load();
    if (!cs_label(e.y, la) || !cs_label(e.z, lb)) return;
    uint m = min(la, lb);
    labels_next[e.y].min(m);
    labels_next[e.z].min(m);
}

[shader("compute")][numthreads(64, 1, 1)]
void constraint_check_round(uint3 tid : SV_DispatchThreadID)
{
    uint v;
    if (!cs_item(tid.x, constraint_params.counts.x, v)) return;
    uint old = labels_old[v].load(), next = labels_next[v].load(), status = 0u;
    BodyModel m = body_models[v];
    if (cs_writer(m) && cs_body(v, m.meta.w))
    {
        if (!cs_label(v, old) || !cs_label(v, next) || next > old) status |= ConstraintBadLabel;
    }
    else if (old != ConstraintInvalid || next != ConstraintInvalid) status |= ConstraintBadLabel;
    cs_store(constraint_params.layout.z + v, uint4(old != next ? 1u : 0u, 0u, 0u, status));
}

// Host uses hierarchical mode0 reduction on all B round-check rows, then this
// 1-lane commit; only after completion may it swap label Old/Next bindings.
[shader("compute")][numthreads(1, 1, 1)]
void constraint_commit_round(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 prev = cs_load(constraint_params.layout.y);
    uint4 check = cs_load(constraint_params.round.x);
    if (prev.x == 0u)
    {
        cs_store(constraint_params.layout.y, uint4(0u, prev.y, prev.z | check.w, 0u));
        return;
    }
    uint completed = prev.y + 1u;
    uint status = prev.z | check.w;
    if (prev.y == ConstraintInvalid || completed > constraint_params.counts.x
        || (completed == constraint_params.counts.x && check.x != 0u)) status |= ConstraintNotConverged;
    cs_store(constraint_params.layout.y, uint4(check.x, completed, status, 0u));
}

// Independent post-convergence verification reads the SWAPPED final Old.
// Host reduces these rows after the final swap, never in an edge-writing pass.
[shader("compute")][numthreads(64, 1, 1)]
void constraint_validate_labels(uint3 tid : SV_DispatchThreadID)
{
    uint v;
    if (!cs_item(tid.x, constraint_params.counts.x, v)) return;
    uint status = 0u, l = labels_old[v].load();
    if (cs_writer(body_models[v]) && cs_body(v, body_models[v].meta.w))
    {
        if (!cs_label(v, l)) status |= ConstraintBadLabel;
        else if (labels_old[l].load() != l) status |= ConstraintBadLabel;
    }
    else if (l != ConstraintInvalid) status |= ConstraintBadLabel;
    cs_store(constraint_params.layout.z + v, uint4(0u, 0u, 0u, status));
}
[shader("compute")][numthreads(64, 1, 1)]
void constraint_validate_edges(uint3 tid : SV_DispatchThreadID)
{
    uint r, kind = constraint_params.select.x;
    if (!cs_item(tid.x, cs_ref_count(kind), r)) return;
    uint p = cs_ref_base(kind) + 2u * r;
    uint4 e = cs_load(p), flags = cs_load(p + 1u);
    uint status = flags.z;
    if (flags.y != 0u)
    {
        uint root = ConstraintInvalid;
        if (e.y != ConstraintInvalid)
        {
            root = labels_old[e.y].load();
            if (!cs_label(e.y, root)) status |= ConstraintBadLabel;
        }
        if (e.z != ConstraintInvalid)
        {
            uint other = labels_old[e.z].load();
            if (!cs_label(e.z, other) || (root != ConstraintInvalid && root != other)) status |= ConstraintBadLabel;
            root = other;
        }
        if (root == ConstraintInvalid || e.x >= constraint_params.counts.w) status |= ConstraintBadLabel;
        else if (body_models[root].meta.w != e.x || labels_old[root].load() != root) status |= ConstraintBadLabel;
    }
    uint offset = kind == ConstraintJoint ? 0u : constraint_params.counts.y;
    cs_store(constraint_params.layout.w + offset + r, uint4(
        kind == ConstraintJoint ? flags.y : 0u, kind == ConstraintPatch ? flags.y : 0u, 0u, status));
}
[shader("compute")][numthreads(1, 1, 1)]
void constraint_accept_components(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 control = cs_load(constraint_params.layout.y);
    uint status = control.z | cs_load(constraint_params.summary.z).w | cs_load(constraint_params.summary.w).w;
    if (control.x != 0u || control.y > constraint_params.counts.x) status |= ConstraintNotConverged;
    control.z = status;
    control.w = status == 0u ? 1u : 0u;
    cs_store(constraint_params.layout.y, control);
}

// Mode0: three checked sums + status OR. Mode1: two sums + two maxima.
// Mode1 inputs are disjoint ranges with totals <= admitted J/P; never use it
// for untrusted arbitrary integer reductions (it has no fourth status lane).
uint4 cs_combine(uint4 a, uint4 b)
{
    if (constraint_params.scanParent.y == 1u)
        return uint4(a.x + b.x, a.y + b.y, max(a.z, b.z), max(a.w, b.w));
    uint3 sum = a.xyz + b.xyz;
    uint status = a.w | b.w | (any(sum < a.xyz) ? ConstraintOverflow : 0u);
    return uint4(sum, status);
}
groupshared uint4 cs_shared[64];
// StageWork first/count means BLOCKS for these two collective entries only.
// Every lane in a dispatched group participates in every group barrier.
[shader("compute")][numthreads(64, 1, 1)]
void constraint_scan_blocks(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint block;
    if (!cs_item(gid.x, cs_blocks(constraint_params.scan.w), block)) return;
    uint first = block * ConstraintBlockSize;
    bool valid = lane < constraint_params.scan.w - first;
    uint4 input = uint4(0u);
    if (valid) input = cs_load(constraint_params.scan.x + first + lane);
    cs_shared[lane] = input;
    GroupMemoryBarrierWithGroupSync();
    // Fixed six scan steps; no chain traversal or globally spinning group.
    for (uint offset = 1u; offset < 64u; offset *= 2u)
    {
        uint4 left = uint4(0u);
        if (lane >= offset) left = cs_shared[lane - offset];
        GroupMemoryBarrierWithGroupSync();
        uint4 sum = cs_combine(left, cs_shared[lane]);
        GroupMemoryBarrierWithGroupSync();
        cs_shared[lane] = sum;
        GroupMemoryBarrierWithGroupSync();
    }
    if (valid)
    {
        uint4 exclusive = uint4(0u);
        if (lane != 0u) exclusive = cs_shared[lane - 1u];
        cs_store(constraint_params.scan.y + first + lane, exclusive);
    }
    if (lane == 63u) cs_store(constraint_params.scan.z + block, cs_shared[63]);
}
[shader("compute")][numthreads(64, 1, 1)]
void constraint_reduce_blocks(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint block;
    if (!cs_item(gid.x, cs_blocks(constraint_params.scan.w), block)) return;
    uint first = block * ConstraintBlockSize;
    uint4 input = uint4(0u);
    if (lane < constraint_params.scan.w - first) input = cs_load(constraint_params.scan.x + first + lane);
    cs_shared[lane] = input;
    GroupMemoryBarrierWithGroupSync();
    for (uint width = 32u; width != 0u; width /= 2u)
    {
        if (lane < width) cs_shared[lane] = cs_combine(cs_shared[lane], cs_shared[lane + width]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) cs_store(constraint_params.scan.z + block, cs_shared[0]);
}
[shader("compute")][numthreads(64, 1, 1)]
void constraint_add_scan_offsets(uint3 tid : SV_DispatchThreadID)
{
    uint i;
    if (!cs_item(tid.x, constraint_params.scan.w, i)) return;
    cs_store(constraint_params.scan.y + i, cs_combine(cs_load(constraint_params.scanParent.x + i / 64u),
        cs_load(constraint_params.scan.y + i)));
}

bool cs_less(uint4 a, uint4 b)
{
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    if (a.z != b.z) return a.z < b.z;
    return a.w < b.w;
}
[shader("compute")][numthreads(64, 1, 1)]
void constraint_seed_sort(uint3 tid : SV_DispatchThreadID)
{
    uint r, kind = constraint_params.select.x;
    if (!cs_item(tid.x, cs_ref_count(kind), r)) return;
    uint p = cs_ref_base(kind) + 2u * r;
    uint4 key = uint4(ConstraintInvalid);
    if (cs_load(constraint_params.layout.y).w == 1u && cs_load(p + 1u).y != 0u)
    {
        uint4 e = cs_load(p);
        uint body = e.y != ConstraintInvalid ? e.y : e.z;
        key = uint4(e.x, labels_old[body].load(), e.w, kind);
    }
    cs_store(constraint_params.select.y + r, key);
}

// Stable merge-by-rank. One unique output store/item; no lane walks a run.
// Each binary search is bounded by the 32-bit index space. This small generic
// primitive is a candidate for sharing with broadphase, not a duplicated SAP
// shader or private BP arena contract.
[shader("compute")][numthreads(64, 1, 1)]
void constraint_merge_items(uint3 tid : SV_DispatchThreadID)
{
    uint i;
    if (!cs_item(tid.x, constraint_params.merge.z, i)) return;
    uint width = constraint_params.merge.w;
    uint run = i / width, begin = run * width;
    uint end = begin + min(width, constraint_params.merge.z - begin);
    bool right = (run & 1u) != 0u;
    uint other_begin = right ? begin - width : end;
    uint other_end = right ? begin : other_begin + min(width, constraint_params.merge.z - other_begin);
    uint4 key = cs_load(constraint_params.merge.x + i);
    uint lo = other_begin, hi = other_end;
    for (uint step = 0u; step < 32u; ++step)
    {
        if (lo >= hi) break;
        uint mid = lo + (hi - lo) / 2u;
        uint4 other = cs_load(constraint_params.merge.x + mid);
        // Left keys use lower_bound; right keys use upper_bound. Thus equal
        // initialized sentinel keys retain original canonical source order.
        bool precedes = right ? !cs_less(key, other) : cs_less(other, key);
        if (precedes) lo = mid + 1u; else hi = mid;
    }
    uint pair_begin = right ? other_begin : begin;
    uint destination = pair_begin + (i - begin) + (lo - other_begin);
    cs_store(constraint_params.merge.y + destination, key);
}

// Final CC Old is read-only. Next is now repurposed, so no further CC round
// may use it; cancellation/resume must remember this explicit phase boundary.
[shader("compute")][numthreads(64, 1, 1)]
void constraint_clear_root_flags(uint3 tid : SV_DispatchThreadID)
{
    uint v;
    if (cs_item(tid.x, constraint_params.counts.x, v)) labels_next[v].store(0u);
}
[shader("compute")][numthreads(64, 1, 1)]
void constraint_mark_roots(uint3 tid : SV_DispatchThreadID)
{
    uint r, kind = constraint_params.select.x;
    if (!cs_item(tid.x, cs_ref_count(kind), r)) return;
    if (cs_load(constraint_params.layout.y).w != 1u) return;
    uint p = cs_ref_base(kind) + 2u * r;
    if (cs_load(p + 1u).y == 0u) return;
    uint4 e = cs_load(p);
    uint body = e.y != ConstraintInvalid ? e.y : e.z;
    uint root = labels_old[body].load();
    labels_next[root].or(1u << kind);
}
[shader("compute")][numthreads(64, 1, 1)]
void constraint_build_root_rows(uint3 tid : SV_DispatchThreadID)
{
    uint v;
    if (!cs_item(tid.x, constraint_params.counts.x, v)) return;
    uint flags = labels_next[v].load();
    uint status = flags != 0u && labels_old[v].load() != v ? ConstraintBadLabel : 0u;
    cs_store(constraint_params.islands.y + v,
        uint4(flags != 0u ? 1u : 0u, flags & 1u, (flags >> 1u) & 1u, status));
}

// Count lower/upper bounds of (world,root) in an independently sorted kind.
uint cs_bound(uint base, uint count, uint world, uint root, bool upper)
{
    uint lo = 0u, hi = count;
    for (uint step = 0u; step < 32u; ++step)
    {
        if (lo >= hi) break;
        uint mid = lo + (hi - lo) / 2u;
        uint4 key = cs_load(base + mid);
        bool before = key.x < world || (key.x == world && (key.y < root || (upper && key.y == root)));
        if (before) lo = mid + 1u; else hi = mid;
    }
    return lo;
}
[shader("compute")][numthreads(64, 1, 1)]
void constraint_write_sources(uint3 tid : SV_DispatchThreadID)
{
    uint i, kind = constraint_params.select.x;
    if (!cs_item(tid.x, cs_ref_count(kind), i)) return;
    uint base = kind == ConstraintJoint ? constraint_params.references.z : constraint_params.references.w;
    uint destination = kind == ConstraintJoint ? constraint_params.sources.x : constraint_params.sources.y;
    uint4 key = cs_load(base + i);
    cs_store(destination + i, uint4(key.z, 0u, 0u, 0u));
}
[shader("compute")][numthreads(64, 1, 1)]
void constraint_write_islands(uint3 tid : SV_DispatchThreadID)
{
    uint root;
    if (!cs_item(tid.x, constraint_params.counts.x, root)) return;
    uint4 row = cs_load(constraint_params.islands.y + root);
    // Every B summary row gets an exact write. Isolated unused Dynamic bodies
    // do not create fake numerical islands.
    uint4 range_summary = uint4(0u);
    if (row.x != 0u && cs_load(constraint_params.layout.y).w == 1u)
    {
        uint4 prefix = cs_load(constraint_params.islands.z + root);
        uint id = prefix.x, world = body_models[root].meta.w;
        uint jb = cs_bound(constraint_params.references.z, constraint_params.counts.y, world, root, false);
        uint je = cs_bound(constraint_params.references.z, constraint_params.counts.y, world, root, true);
        uint pb = cs_bound(constraint_params.references.w, constraint_params.counts.z, world, root, false);
        uint pe = cs_bound(constraint_params.references.w, constraint_params.counts.z, world, root, true);
        uint jc = je - jb, pc = pe - pb;
        cs_store(constraint_params.islands.x + 2u * id, uint4(world, root, jb, jc));
        cs_store(constraint_params.islands.x + 2u * id + 1u, uint4(pb, pc, 0u, 0u));
        if (row.y != 0u) cs_store(constraint_params.sources.z + prefix.y, uint4(id, 0u, 0u, 0u));
        if (row.z != 0u) cs_store(constraint_params.sources.w + prefix.z, uint4(id, 0u, 0u, 0u));
        range_summary = uint4(jc, pc, jc, pc);
    }
    cs_store(constraint_params.islands.w + root, range_summary);
}

// Must run after all sorts, source writes, root scan and independent final CC
// root/edge validation. Construction guarantees list/range/table identities;
// output rereads belong only to the separate private diagnostic fragment.
[shader("compute")][numthreads(1, 1, 1)]
void constraint_finalize(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 control = cs_load(constraint_params.layout.y);
    uint4 roots = cs_load(constraint_params.summary.x);
    uint4 ranges = cs_load(constraint_params.summary.y);
    uint4 refs = cs_load(constraint_params.summary.z);
    uint status = control.z | roots.w | refs.w | cs_load(constraint_params.summary.w).w;
    if (control.w != 1u) status |= ConstraintNotConverged;
    if (roots.x > constraint_params.counts.x || roots.y > roots.x || roots.z > roots.x
        || ranges.x > constraint_params.counts.y || ranges.y > constraint_params.counts.z
        || ranges.z > ranges.x || ranges.w > ranges.y
        || ranges.x != refs.x || ranges.y != refs.y) status |= ConstraintBadSchedule;
    cs_store(constraint_params.layout.x + 1u, ranges);
    cs_store(constraint_params.layout.x, uint4(roots.xyz, status));
}
