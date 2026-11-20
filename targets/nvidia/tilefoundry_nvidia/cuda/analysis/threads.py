"""Shared memory that two threads may write in one issue of an action.

An instruction without an execution mesh is executed by every thread of the mesh
it runs on, each on the operands as that thread sees them, and no step orders the
threads of one issue. The ``sync`` family refuses a Function in which two such
threads may write one shared memory element in the same issue, unless the
operand's layout gives each of its bytes to one of the threads that run it. See
[analysis §1](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/analysis.md#1-sync).
"""

from __future__ import annotations

from math import prod

from tilefoundry import analysis
from tilefoundry.ir import core, dim, expr, types, utils
from tilefoundry.ir.tir import memory
from tilefoundry.passes import transforms

from .ctas import Overlap, Reaches, overlap
from .emission import required_execution_mesh


def issued_op(planner, action):
    """The instruction every executing thread runs on its own for *action*, or None.

    None for a retained call, which orders its own accesses. None for an
    instruction with a required execution mesh, and for one that takes its
    execution mesh as an attribute, as ``CopyAsyncTensor`` does: each states for
    itself which threads execute it.
    """
    owner, tag = action.owner, action.tag
    if isinstance(owner, core.LoopRegion) or tag in ("base", "update"):
        op = memory.Copy()
    elif planner.locations[id(owner)].attachment(owner) is not None:
        return None
    elif planner.visit(owner) in (planner.visit_Zeros, planner.visit_FullLike):
        op = memory.Fill()
    else:
        op = transforms.selected_op(owner, planner.context.target)
    if op is None or required_execution_mesh(op) is not None:
        return None
    stated = getattr(type(op), "execution_mesh", None)
    return None if isinstance(stated, expr.ParamDef) and stated.kind == "attribute" else op


def copy_types(action, op) -> tuple:
    """The source and destination types of *action* when it issues a ``Copy``, else ()."""
    owner, tag = action.owner, action.tag
    if isinstance(owner, core.LoopRegion):
        kind, slot = tag
        source = owner.args[slot] if kind == "entry" else owner.yield_values[slot]
        return (source.type, owner.params[slot].type)
    if tag in ("base", "update"):
        return (owner.args[0 if tag == "base" else 1].type, owner.type)
    if isinstance(op, memory.Copy):
        return tuple(transforms.operand_types(owner, op))
    return ()


def projected(type_list) -> types.Mesh | None:
    """The mesh through whose split a ``Copy`` of these two operands moves each thread's slice.

    For a plain operand paired with a sharded one, the runtime's copy moves only
    the slice the unit owns, projected through the sharded side
    (``copy_impl::local_as``). The slices of different units are disjoint
    when the shard splits along every mesh axis of its thread topology. None otherwise.
    An operand whose mesh holds no thread topology, such as a tile Varying per
    CTA, is the same for every thread of a CTA, so it counts as plain here.
    """
    sharded = [
        type_
        for type_ in type_list
        if isinstance(getattr(type_, "layout", None), types.ShardLayout)
        and "thread" in utils.topology_names(type_.layout.mesh)
    ]
    if len(type_list) != 2 or len(sharded) != 1:
        return None
    layout = sharded[0].layout
    mesh_axes = zip(layout.attrs, utils.axis_keys(layout.mesh), strict=True)
    on_threads = [attr for attr, (topology, _) in mesh_axes if topology == "thread"]
    split = bool(on_threads) and all(isinstance(attr, types.Split) for attr in on_threads)
    return layout.mesh if split else None


def held_threads(mesh: types.Mesh) -> frozenset | None:
    """The threads of one CTA that *mesh*'s thread topology holds, None when they are not static."""
    names = utils.topology_names(mesh)
    if "thread" not in names or (
        isinstance(mesh.layout, types.ComposedLayout) and not isinstance(mesh.layout.offset, int)
    ):
        return None
    thread = names.index("thread")
    try:
        layout = types.plain_layout(utils.unit_layout(mesh, thread))
    except ValueError:
        return None
    first_unit_id = utils.first_unit_ids(mesh)[thread]
    if layout is None or not isinstance(first_unit_id, int):
        return None
    count = types.size(layout)
    if not isinstance(count, int):
        return None
    held = [first_unit_id + layout(index) for index in range(count)]
    return frozenset(held) if all(isinstance(thread, int) for thread in held) else None


def one_writer_each(mesh: types.Mesh | None, reach, threads: int | None) -> bool:
    """Whether a split over *mesh* gives each byte of *reach* one writer among its threads.

    A split gives each unit of *mesh* its own bytes. The threads that run
    the reach are those every thread slice around it holds, or every thread of
    the CTA. The split gives one writer each only when its units are exactly
    those threads: a mesh that holds other threads, or fewer, repeats over them,
    and threads that read one unit's bytes write the same bytes.
    """
    split = None if mesh is None else held_threads(mesh)
    runners = None if threads is None else frozenset(range(threads))
    for slice_ in reach.slices:
        kept = held_threads(slice_)
        if kept is None:
            return False
        runners = kept if runners is None else runners & kept
    return split is not None and split == runners


def thread_coordinate(value) -> bool:
    """Whether *value* is a coordinate of a mesh axis of the thread topology."""
    if not (isinstance(value, expr.Call) and isinstance(value.target, dim.MeshCoord)):
        return False
    mesh_axis = expr.static_dim_value(value.args[0])
    return mesh_axis is not None and utils.axis_keys(value.target.mesh)[mesh_axis][0] == "thread"


def refuse_thread_overlaps(planner) -> None:
    """Refuse the planner's Function when two threads may write one shared element in one issue.

    Each shared memory write of an action that every executing thread runs on its
    own pairs with itself, unless its layout, or a split operand a ``Copy``
    projects it through, gives each of its bytes one writer among the threads
    that run it. Both executions see the same values for everything but thread
    coordinates: the same CTA, the same loop iterations and the same parameters.
    """
    sizes = [t.size for t in planner.context.topologies or () if t.name == "thread"]
    threads = prod(sizes) if all(isinstance(size, int) for size in sizes) else None
    if threads == 1:
        return
    reaches = Reaches(planner, expr.StorageKind.SMEM, "thread")
    for action in planner.actions.values():
        op = issued_op(planner, action)
        if op is None:
            continue
        through = projected(copy_types(action, op))
        for reach in reaches.of(action):
            if not reach.write or any(
                one_writer_each(mesh, reach, threads) for mesh in (reach.split, through)
            ):
                continue
            verdict = overlap(
                reach, reach, lambda value: not thread_coordinate(value), threads, "thread"
            )
            if verdict is not None:
                what = "write" if verdict is Overlap.CONFLICT else "may write"
                raise analysis.Error(
                    f"sync: two threads that execute {reach.name} {what} one shared memory "
                    "element at once, which no step orders"
                )


__all__ = ["refuse_thread_overlaps"]
