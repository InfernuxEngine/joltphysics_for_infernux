// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT
#include "../ConstraintTypes.h"
#include "../SleepTypes.h"
#include "key_sort.slang"
#include "component_union.slang"

ConstantBuffer<ConstraintParams> constraint_params;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<QueryRecordWord> joint_stream;
StructuredBuffer<ContactPatch> contact_patches;
StructuredBuffer<Contact> contacts;
StructuredBuffer<BodySleepRecord> body_sleep;
// Cross-target scalar atomic resources. Atomic<uint> has uint's 4-byte layout;
// unlike HLSL Interlocked intrinsics on ordinary uint, its methods support WGSL.
// The parent forest and the vote/status bank have independent lifetimes.
RWStructuredBuffer<Atomic<uint>> component_parents;
RWStructuredBuffer<Atomic<uint>> root_votes;
RWStructuredBuffer<Atomic<uint>> step_status;
RWStructuredBuffer<ConstraintScheduleWord> constraint_schedule;
RWStructuredBuffer<ConstraintScheduleWord> topology_arguments;

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
    if (world != constraint_params.selection.x || !cs_in_range(body, constraint_params.selection.y, constraint_params.counts.x))
        return false;
    return body_models[body].meta.w == world;
}
uint cs_local_body(uint packed) { return packed - constraint_params.selection.y; }
uint cs_packed_body(uint local) { return constraint_params.selection.y + local; }
uint cs_initial_root(uint local)
{
    BodyModel model = body_models[cs_packed_body(local)];
    return cs_writer(model) && model.meta.w == constraint_params.selection.x ? local : ConstraintInvalid;
}

uint cs_ref_base(uint kind)
{
    return kind == ConstraintJoint ? constraint_params.references.x : constraint_params.references.y;
}
uint cs_ref_count(uint kind)
{
    return kind == ConstraintJoint ? constraint_params.counts.y : constraint_params.counts.z;
}
uint cs_ref_scratch(uint kind, uint ordinal)
{
    return constraint_params.layout.w + (kind == ConstraintJoint ? 0u : constraint_params.counts.y) + ordinal;
}
void cs_ref(uint kind, uint ordinal, uint world, uint a, uint b, bool active, uint status)
{
    uint p = cs_ref_base(kind) + 2u * ordinal;
    uint source = kind == ConstraintJoint ? constraint_params.selection.w + ordinal : ordinal;
    active = active && status == 0u;
    cs_store(p, uint4(world, a, b, source));
    cs_store(p + 1u, uint4(kind, active ? 1u : 0u, status, 0u));
    if (constraint_params.select.z == 0u)
    {
        bool links = active && a != ConstraintInvalid && b != ConstraintInvalid;
        uint2 cursors = links ? uint2(cs_local_body(a), cs_local_body(b)) : uint2(0u);
        cs_store(cs_ref_scratch(kind, ordinal), uint4(cursors, 0u, 0u));
    }
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
    cs_store(h + 3u, uint4(constraint_params.sources.y, constraint_params.limits.x, constraint_params.limits.w, constraint_params.counts.x));
}

// Runs only after the complete SetupJoints dispatch sequence. It reads initialized
// author/runtime scalar metadata, never BodyState and never applies impulses.
[shader("compute")][numthreads(64, 1, 1)]
void constraint_build_joint_refs(uint3 tid : SV_DispatchThreadID)
{
    uint item;
    if (!cs_item(tid.x, constraint_params.counts.y, item)) return;
    uint j = constraint_params.selection.w + item;
    uint4 header = joint_stream[0].value;
    uint status = 0u;
    if (header.x != constraint_params.counts.w || header.w != 0u ||
        header.y < 1u + JointDirectoryWords * header.x || header.z < header.y)
    {
        cs_ref(ConstraintJoint, item, ConstraintInvalid, ConstraintInvalid, ConstraintInvalid, false, ConstraintBadJoint);
        return;
    }
    uint author = 1u + JointDirectoryWords * j;
    uint4 e = joint_stream[author].value;
    uint world = joint_stream[author + 1u].value.z;
    uint4 layout = joint_stream[author + 2u].value;
    uint author_words = 0u, state_words = 0u;
    switch (e.z)
    {
    case JointFixed: author_words = JointFixedAuthorWords; state_words = JointFixedStateWords; break;
    case JointHinge: author_words = JointHingeAuthorWords; state_words = JointHingeStateWords; break;
    case JointSlider: author_words = JointSliderAuthorWords; state_words = JointSliderStateWords; break;
    case JointDistance: author_words = JointDistanceAuthorWords; state_words = JointDistanceStateWords; break;
    case JointSwingTwist: author_words = JointSwingAuthorWords; state_words = JointSwingStateWords; break;
    case JointSixDOF: author_words = JointSixAuthorWords; state_words = JointSixStateWords; break;
    }
    if (author_words == 0u || state_words == 0u || any(layout.zw != uint2(0u))
        || layout.x < 1u + JointDirectoryWords * header.x || layout.x > header.y
        || author_words > header.y - layout.x || layout.y < header.y || layout.y > header.z
        || state_words > header.z - layout.y)
    {
        cs_ref(ConstraintJoint, item, ConstraintInvalid, ConstraintInvalid, ConstraintInvalid, false, ConstraintBadJoint);
        return;
    }
    uint4 state = joint_stream[layout.y].value;
    if (state.y != JointStatusSuccess) status |= ConstraintBadJoint;
    if (!cs_body(e.y, world)
        || (e.x != InvalidQueryBodySlot && !cs_body(e.x, world))) status |= ConstraintBadModel;
    uint a = ConstraintInvalid, b = ConstraintInvalid;
    bool resident = false;
    if ((status & ConstraintBadModel) == 0u)
    {
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
    bool active = resident && (constraint_params.select.z == 0u || state.x != 0u) && (e.w & JointEnabled) != 0u
        && (a != ConstraintInvalid || b != ConstraintInvalid);
    if (constraint_params.select.z != 0u) {
        bool awake = false;
        if (b != ConstraintInvalid) awake = body_sleep[b].runtime.state.x != 0u;
        if (a != ConstraintInvalid) awake = awake || body_sleep[a].runtime.state.x != 0u;
        active = active && awake;
    }
    cs_ref(ConstraintJoint, item, world, a, b, active, status);
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
    if (!cs_in_range(patch.x, constraint_params.selection.z, constraint_params.limits.y)
        || !cs_in_range(patch.y, constraint_params.selection.z, constraint_params.limits.y))
        status |= ConstraintBadModel;
    else
    {
        ShapeModel sa = shape_models[patch.x], sb = shape_models[patch.y];
        world = constraint_params.selection.x;
        if (!cs_body(sa.meta.y, world) || !cs_body(sb.meta.y, world))
            status |= ConstraintBadModel;
        else
        {
            BodyModel ma = body_models[sa.meta.y], mb = body_models[sb.meta.y];
            if (!cs_in_range(patch.x, ma.meta.x, ma.meta.y)
                || !cs_in_range(patch.y, mb.meta.x, mb.meta.y)) status |= ConstraintBadModel;
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
    if (constraint_params.select.z != 0u) {
        bool awake = false;
        if (a != ConstraintInvalid) awake = body_sleep[a].runtime.state.x != 0u;
        if (b != ConstraintInvalid) awake = awake || body_sleep[b].runtime.state.x != 0u;
        active = active && awake;
    }
    cs_ref(ConstraintPatch, p, world, a, b, active, status);
}

// A complete bounded graph fits in one workgroup. It owns the parent forest
// until every edge is linked, so joining, flattening and publishing the roots
// need no device-wide barriers between them. Shared read-only bodies never link islands.
groupshared Atomic<uint> cs_local_parents[ConstraintLocalBodies];
groupshared Atomic<uint> cs_local_status;

[shader("compute")][numthreads(64, 1, 1)]
void constraint_build_local_topology(uint lane : SV_GroupIndex)
{
    const uint bodies = constraint_params.counts.x, joints = constraint_params.counts.y;
    const uint refs = joints + constraint_params.counts.z;
    for (uint body = lane; body < bodies; body += ConstraintBlockSize)
        cs_local_parents[body].store(cs_initial_root(body));
    if (lane == 0u) cs_local_status.store(0u);
    GroupMemoryBarrierWithGroupSync();

    uint status = 0u;
    for (uint i = lane; i < refs; i += ConstraintBlockSize)
    {
        const uint row = i < joints ? cs_ref_base(ConstraintJoint) + 2u * i
                                    : cs_ref_base(ConstraintPatch) + 2u * (i - joints);
        const uint4 flags = cs_load(row + 1u);
        status |= flags.z;
        if (flags.y == 0u) continue;
        uint2 edge = cs_load(row).yz;
        if (any(edge == uint2(ConstraintInvalid))) continue;
        edge -= uint2(constraint_params.selection.y);
        while (edge.x != edge.y)
        {
            const uint hi = max(edge.x, edge.y), lo = min(edge.x, edge.y);
            const uint parent = cs_local_parents[hi].compareExchange(hi, lo);
            if (parent == hi) break;
            // A failed exchange observes a strictly smaller parent. No lane waits for a peer.
            edge = uint2(parent, lo);
        }
    }
    if (status != 0u) cs_local_status.or(status);
    GroupMemoryBarrierWithGroupSync();

    const uint errors = cs_local_status.load();
    for (uint body = lane; body < bodies; body += ConstraintBlockSize)
    {
        uint root = cs_local_parents[body].load();
        if (root != ConstraintInvalid)
        {
            uint parent = cs_local_parents[root].load();
            while (parent != root)
            {
                root = parent;
                parent = cs_local_parents[root].load();
            }
        }
        component_parents[body].store(root);
        cs_store(constraint_params.limits.w + body, uint4(root, 0u, 0u, 0u));
        root_votes[body].store(0u);
    }
    // The wake kernels consume these roots on the same queue. An invalid input
    // dispatches no consumers and reports the error when the tick publishes.
    if (lane < cs_blocks(max(bodies, max(joints, constraint_params.counts.z))))
        topology_arguments[lane].value = uint4(errors == 0u ? lane + 1u : 0u, 1u, 1u, 0u);
    if (lane == 0u)
    {
        if (errors != 0u) step_status[StepTopologyStatus].or(errors);
        cs_store(constraint_params.layout.y, uint4(0u, 1u, errors, errors == 0u ? 1u : 0u));
    }
}

// The one validated CC labels every resident Dynamic, including an isolated
// vertex without a numeric island. Other slots have an explicit Invalid row.
[shader("compute")][numthreads(64,1,1)]
void constraint_write_body_roots(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!cs_item(tid.x,constraint_params.counts.x,body)) return;
    cs_store(constraint_params.limits.w+body,uint4(component_parents[body].load(),0u,0u,0u));
}
[shader("compute")][numthreads(64, 1, 1)]
void constraint_init_parents(uint3 tid : SV_DispatchThreadID)
{
    uint v;
    if (!cs_item(tid.x, constraint_params.counts.x, v)) return;
    component_parents[v].store(cs_initial_root(v));
}
bool cs_label(uint vertex, uint label)
{
    if (vertex >= constraint_params.counts.x || label > vertex) return false;
    return cs_initial_root(vertex) != ConstraintInvalid && cs_initial_root(label) != ConstraintInvalid;
}

[shader("compute")][numthreads(1, 1, 1)]
void constraint_reset_connectivity(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0u) root_votes[constraint_params.counts.x].store(0u);
}

[shader("compute")][numthreads(64, 1, 1)]
void constraint_join_edges(uint3 tid : SV_DispatchThreadID)
{
    uint r, kind = constraint_params.select.x;
    if (!cs_item(tid.x, cs_ref_count(kind), r)) return;
    uint p = cs_ref_scratch(kind, r);
    uint2 edge = cs_load(p).xy;
    if (edge.x == edge.y) return;
    edge = jph_join_components(component_parents, edge, ConstraintConnectivityQuantum);
    cs_store(p, uint4(edge, 0u, 0u));
    if (edge.x != edge.y) root_votes[constraint_params.counts.x].or(ConstraintLinksPending);
}

// With the links settled, each invocation shortens only its own parent.
// Concurrent shortcuts are safe to observe and can only reduce the work.
[shader("compute")][numthreads(64, 1, 1)]
void constraint_flatten_parents(uint3 tid : SV_DispatchThreadID)
{
    uint v;
    if (!cs_item(tid.x, constraint_params.counts.x, v)) return;
    uint status = constraint_params.counts.x;
    if ((root_votes[status].load() & ConstraintLinksPending) != 0u) return;
    uint ancestor = component_parents[v].load();
    if (ancestor == v || ancestor == ConstraintInvalid) return;
    for (uint n = 0u; n < ConstraintConnectivityQuantum; ++n)
    {
        uint parent = component_parents[ancestor].load();
        if (parent == ancestor)
        {
            component_parents[v].store(ancestor);
            return;
        }
        ancestor = parent;
    }
    component_parents[v].store(ancestor);
    root_votes[status].or(ConstraintRootsPending);
}

[shader("compute")][numthreads(64, 1, 1)]
void constraint_gate_validation(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= ConstraintBlockLimit) return;
    uint pending = root_votes[constraint_params.counts.x].load();
    topology_arguments[i].value = uint4(pending == 0u ? i + 1u : 0u, 1u, 1u, 0u);
    if (i == 0u)
    {
        uint address = constraint_params.layout.y;
        uint pass = cs_load(address).y + 1u;
        cs_store(address, uint4(pending, pass, 0u, 0u));
    }
}

// Verification consumes the settled forest in separate dispatches.
[shader("compute")][numthreads(64, 1, 1)]
void constraint_validate_labels(uint3 tid : SV_DispatchThreadID)
{
    uint v;
    if (!cs_item(tid.x, constraint_params.counts.x, v)) return;
    uint status = 0u, l = component_parents[v].load();
    if (cs_initial_root(v) != ConstraintInvalid)
    {
        if (!cs_label(v, l)) status |= ConstraintBadLabel;
        else if (component_parents[l].load() != l) status |= ConstraintBadLabel;
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
            uint a = cs_local_body(e.y);
            root = component_parents[a].load();
            if (!cs_label(a, root)) status |= ConstraintBadLabel;
        }
        if (e.z != ConstraintInvalid)
        {
            uint b = cs_local_body(e.z);
            uint other = component_parents[b].load();
            if (!cs_label(b, other) || (root != ConstraintInvalid && root != other)) status |= ConstraintBadLabel;
            root = other;
        }
        if (root >= constraint_params.counts.x || e.x != constraint_params.selection.x) status |= ConstraintBadLabel;
        else if (component_parents[root].load() != root) status |= ConstraintBadLabel;
    }
    cs_store(cs_ref_scratch(kind, r), uint4(
        kind == ConstraintJoint ? flags.y : 0u, kind == ConstraintPatch ? flags.y : 0u, 0u, status));
}
[shader("compute")][numthreads(1, 1, 1)]
void constraint_accept_components(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 control = cs_load(constraint_params.layout.y);
    uint status = control.z | cs_load(constraint_params.summary.z).w | cs_load(constraint_params.summary.w).w;
    if (control.x != 0u) status |= ConstraintNotConverged;
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

[shader("compute")][numthreads(64, 1, 1)]
void constraint_seed_sort(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    uint r, kind = constraint_params.select.x;
    bool valid = cs_item(tid.x, cs_ref_count(kind), r);
    uint4 key = uint4(ConstraintInvalid);
    if (valid)
    {
        uint p = cs_ref_base(kind) + 2u * r;
        if (cs_load(constraint_params.layout.y).w == 1u && cs_load(p + 1u).y != 0u)
        {
            uint4 e = cs_load(p);
            uint body = e.y != ConstraintInvalid ? e.y : e.z;
            key = uint4(e.x, component_parents[cs_local_body(body)].load(), e.w, kind);
        }
    }
    key = jph_sort_tile(lane, key);
    if (valid) cs_store(constraint_params.select.y + r, key);
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
        bool precedes = right ? !jph_key_less(key, other) : jph_key_less(other, key);
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
    if (cs_item(tid.x, constraint_params.counts.x, v)) root_votes[v].store(0u);
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
    uint root = component_parents[cs_local_body(body)].load();
    root_votes[root].or(1u << kind);
}
[shader("compute")][numthreads(64, 1, 1)]
void constraint_build_root_rows(uint3 tid : SV_DispatchThreadID)
{
    uint v;
    if (!cs_item(tid.x, constraint_params.counts.x, v)) return;
    uint flags = root_votes[v].load();
    uint status = flags != 0u && component_parents[v].load() != v ? ConstraintBadLabel : 0u;
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
    uint2 writers = uint2(0u);
    if (kind == ConstraintPatch && key.z != ConstraintInvalid)
        writers = cs_load(cs_ref_base(kind) + 2u * key.z).yz;
    cs_store(destination + i, uint4(key.z, writers, 0u));
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
        uint id = prefix.x, world = constraint_params.selection.x;
        uint jb = cs_bound(constraint_params.references.z, constraint_params.counts.y, world, root, false);
        uint je = cs_bound(constraint_params.references.z, constraint_params.counts.y, world, root, true);
        uint pb = cs_bound(constraint_params.references.w, constraint_params.counts.z, world, root, false);
        uint pe = cs_bound(constraint_params.references.w, constraint_params.counts.z, world, root, true);
        uint jc = je - jb, pc = pe - pb;
        cs_store(constraint_params.islands.x + 2u * id, uint4(world, cs_packed_body(root), jb, jc));
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
