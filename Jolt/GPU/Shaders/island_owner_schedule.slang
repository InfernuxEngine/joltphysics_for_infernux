// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT
#include "../ConstraintTypes.h"

StructuredBuffer<ConstraintScheduleWord> constraint_schedule;

// A rectangular unit addresses an accepted table span. A whole-table unit
// uses the header's device count. A workgroup walks the same bounded contact
// tiles as rectangular units, in source order for its island. Whole-table
// range.x specifies the round count for batched contact solving.
bool jph_island_unit(uint lane, bool joints, out uint4 joint_range, out uint4 patch_range)
{
    joint_range = uint4(0u);
    patch_range = uint4(0u);
    if (lane >= stage_work.range.w) return false;
    uint table_first = stage_work.range.z;
    if (table_first == ConstraintWholeIslandTable)
    {
        uint4 counts = constraint_schedule[0].value;
        if (lane >= (joints ? counts.y : counts.z)) return false;
        table_first = 0u;
    }
    uint4 bases = constraint_schedule[2].value;
    uint table_base = joints ? bases.y : bases.z;
    uint island = constraint_schedule[table_base + table_first + lane].value.x;
    joint_range = constraint_schedule[bases.x + 2u * island].value;
    patch_range = constraint_schedule[bases.x + 2u * island + 1u].value;
    return joint_range.x == step_params.selection.x;
}

bool jph_island_joint_range(uint lane, out uint begin, out uint first, out uint end)
{
    begin = first = end = 0u;
    uint4 jr, pr;
    if (!jph_island_unit(lane, true, jr, pr)) return false;
    first = stage_work.range.z == ConstraintWholeIslandTable ? 0u : stage_work.range.x;
    if (first >= jr.w) return false;
    begin = jr.z;
    end = first + min(stage_work.range.y, jr.w - first);
    return true;
}

bool jph_island_patch_range(uint lane, out uint begin, out uint first, out uint end)
{
    begin = first = end = 0u;
    uint4 jr, pr;
    if (!jph_island_unit(lane, false, jr, pr)) return false;
    first = stage_work.range.z == ConstraintWholeIslandTable ? 0u : stage_work.range.x;
    if (first >= pr.y) return false;
    begin = pr.x;
    end = first + min(stage_work.range.y, pr.y - first);
    return true;
}

uint jph_island_joint_source(uint begin, uint local)
{
    return constraint_schedule[constraint_schedule[2].value.w + begin + local].value.x;
}

uint4 jph_island_patch_record(uint index)
{
    return constraint_schedule[constraint_schedule[3].value.x + index].value;
}
