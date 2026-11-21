"""Global memory that two CTAs may both reach while one of them writes.

No step orders two CTAs of one launch, so the ``sync`` family refuses an entry in
which two such accesses may reach one element, unless both are the same
tensor-map reduction. See
[analysis §1](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/analysis.md#1-sync).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from math import prod

import isl

from tilefoundry import analysis
from tilefoundry.interfaces import access, schedule
from tilefoundry.ir import core, dim, expr, types, utils
from tilefoundry.passes import transforms
from tilefoundry_nvidia.cuda.ir.tir import memory

from .completion import Proxy
from .emission import operand_roles


class Overlap(Enum):
    """Why two accesses from two different CTAs may reach one element."""

    CONFLICT = auto()
    UNPROVEN = auto()


@dataclass(frozen=True)
class Reach:
    """One read or write of an action on a global memory root or on shared memory.

    ``reached`` is None when the relation is unknown: a retained call, or a value
    with no device coordinates. ``slices`` are the sliced meshes of the checked
    topology around it, CTAs or threads, the only meshes that keep a unit of that
    topology from running it. ``split`` is the mesh whose units the layout of
    the operand gives its bytes to, one unit each, None when it gives none.
    """

    name: str
    write: bool
    async_proxy: bool
    reduction: object | None
    root: object
    reached: analysis.Access | None
    loops: tuple = ()
    slices: tuple = ()
    split: types.Mesh | None = None


def storage_of(value: core.SSAValue):
    leaves = utils.leaves_of(value.expr.type)
    return getattr(leaves[value.index], "storage", None) if value.index < len(leaves) else None


class Reaches:
    """The reads and writes of a planner's actions on *storage*, with their device relations.

    *storage* is global memory, whose reaches carry their allocation roots, or
    shared memory. *topology* names the units whose sliced meshes each reach keeps:
    CTAs, or the threads of one CTA.
    """

    def __init__(
        self, planner, storage: expr.StorageKind = expr.StorageKind.GMEM, topology: str = "cta"
    ):
        self.planner = planner
        self.storage = storage
        self.topology = topology
        self.scope = planner.root
        self.summary = planner.summary

    def physical(self, reached: analysis.Access) -> analysis.Access:
        """*reached* through the address its target's storage takes."""
        if reached.target not in self.planner.values:
            return reached
        alias = self.summary.address_alias(reached.target)
        if alias.target == reached.target:
            return reached
        relation = reached.relation.compose(alias.relation)
        return replace(reached, target=alias.target, relation=relation)

    def records(self, call: expr.Call):
        """The device reads and writes the scope records for *call*, with their paths."""
        found = []
        for side, write in ((self.scope.access.accesses, False), (self.scope.access.outputs, True)):
            for path, (primitive, accesses) in side.get("device", {}).items():
                if primitive is call:
                    found.extend((path, write, reached) for reached in accesses)
        return found

    def whole(self, node, value, *, input_index=None, output_index=None) -> analysis.Access | None:
        """Every element of *value*'s first leaf, through its own device coordinates."""
        leaf = core.SSAValue(value, 0)
        leaves = utils.leaves_of(value.type)
        if not leaves or getattr(leaves[0], "storage", None) is not self.storage:
            return None
        entry = node.provenance.lookup(leaf)
        if entry.device is None:
            return None
        shape = tuple(leaves[0].shape)
        values = {}
        held = dim.shape_to_isl_set(shape, values)
        every = access.identity(len(shape)).relation.intersect_range(held)
        relation = access.Relation(every, values)
        return analysis.Access(
            input_index, relation.compose(entry.device), entry.root, output_index
        )

    def of(self, action) -> list[Reach]:
        """Each read and write of *action* on the storage, as the planner dispatched it."""
        owner, tag = action.owner, action.tag
        if not self.touched(action):
            return []
        if isinstance(owner, core.LoopRegion):
            return self.carry(action, owner, tag)
        if tag in ("base", "update"):
            return self.insert_slice(action, owner, tag)
        if self.planner.locations[id(owner)].attachment(owner) is not None:
            return self.opaque(action)
        if self.planner.visit(owner) in (self.planner.visit_Zeros, self.planner.visit_FullLike):
            return self.fill(action, owner)
        return self.instruction(
            action, owner, transforms.selected_op(owner, self.planner.context.target)
        )

    def touched(self, action):
        """The uses of *action* on the storage: global memory roots, or shared memory bytes."""
        if self.storage is expr.StorageKind.GMEM:
            return [use for use in action.uses if use.global_roots]
        return [use for use in action.uses if use.spans]

    def roots_of(self, use) -> tuple:
        """The global memory roots of *use*; shared memory has one unknown root."""
        return use.global_roots if self.storage is expr.StorageKind.GMEM else (None,)

    def root(self, target: core.SSAValue):
        """The allocation root the planner gives *target*, None when it has none."""
        if target not in self.planner.values:
            return None
        return self.summary.root(target).value

    def opaque(self, action) -> list[Reach]:
        """A retained call's reaches, whose relations stay unknown."""
        return [
            Reach(action.name, use.effect.write, use.effect.proxy is Proxy.ASYNC, None, root, None)
            for use in self.touched(action)
            for root in self.roots_of(use)
        ]

    def slices(self, node) -> tuple:
        """The sliced meshes of the topology's regions around *node*; only these guard who runs it.

        Lowering guards a region by its mesh only when that mesh is a slice
        (``mesh_scope.emit_mesh_scope``); any other mesh repeats over the topology.
        """
        return tuple(
            scope.owner.mesh
            for scope in node.ancestors()
            if isinstance(scope.owner, core.MeshRegion)
            and isinstance(scope.owner.mesh.layout, types.ComposedLayout)
            and any(topology.name == self.topology for topology in scope.owner.mesh.topologies)
        )

    def added(self, action) -> tuple:
        """The sliced mesh of the topology that lowering guards a copy or fill it adds with.

        Lowering runs each copy and fill it adds inside the regions
        ``transfer_meshes`` gives, the innermost being the action's mesh; it
        opens no region for an instruction's mesh.
        """
        mesh = action.mesh
        sliced = mesh is not None and isinstance(mesh.layout, types.ComposedLayout)
        if sliced and any(topology.name == self.topology for topology in mesh.topologies):
            return (mesh,)
        return ()

    def bound(self, action, call: expr.Call, recorded, reads, writes, slices=()) -> list[Reach]:
        """Reaches for *action*'s uses, from *call*'s ``(path, write, reached)`` records.

        *slices* are sliced meshes around the action besides those of the regions around *call*.
        """
        node = self.planner.locations[id(call)]
        loops, slices = node.loop_scopes, self.slices(node) + slices
        found = []
        for use in self.touched(action):
            param = use.effect.param
            async_proxy = use.effect.proxy is Proxy.ASYNC
            split = None if use.threads is None else use.threads.mesh
            chosen = [
                item for item in recorded if not item[1] and item[2].input_index == reads.get(param)
            ]
            if not chosen and param in writes:
                chosen = [item for item in recorded if item[1]]
            for *_, reached in chosen:
                reached = self.physical(reached)
                if storage_of(reached.target) is not self.storage:
                    continue
                found.append(
                    Reach(
                        f"{action.name} {param}",
                        use.effect.write,
                        async_proxy,
                        None,
                        self.root(reached.target),
                        reached,
                        loops,
                        slices,
                        split,
                    )
                )
            if not chosen:
                name, write = f"{action.name} {param}", use.effect.write
                found.extend(
                    Reach(name, write, async_proxy, None, root, None, split=split)
                    for root in self.roots_of(use)
                )
        return found

    def instruction(self, action, call: expr.Call, op) -> list[Reach]:
        params = schedule.call_params(op, len(transforms.operand_types(call, op)))
        supplied, produced = operand_roles(call, params)
        reads = {param.name: index for index, param in enumerate(supplied)}
        writes = {param.name for param in produced}
        found = self.bound(action, call, self.records(call), reads, writes)
        kind = op.kind if isinstance(op, memory.CopyReduceAsyncTensor) else None
        return [replace(reach, reduction=kind) if reach.async_proxy else reach for reach in found]

    def fill(self, action, call: expr.Call) -> list[Reach]:
        recorded = [item for item in self.records(call) if item[1]]
        writes = {use.effect.param for use in self.touched(action) if use.effect.write}
        return self.bound(action, call, recorded, {}, writes, self.added(action))

    def copied(self, action, node, sides) -> list[Reach]:
        """Reaches of a Copy lowering adds at *node*, from its ``(write, reached)`` pairs."""
        loops, slices = node.loop_scopes, self.slices(node) + self.added(action)
        split = {
            use.effect.write: None if use.threads is None else use.threads.mesh
            for use in self.touched(action)
        }
        found = []
        for write, reached in sides:
            if reached is None:
                found.extend(
                    Reach(action.name, use.effect.write, False, None, root, None)
                    for use in self.touched(action)
                    if use.effect.write == write
                    for root in self.roots_of(use)
                )
                continue
            reached = self.physical(reached)
            if storage_of(reached.target) is self.storage:
                root = self.root(reached.target)
                found.append(
                    Reach(
                        action.name,
                        write,
                        False,
                        None,
                        root,
                        reached,
                        loops,
                        slices,
                        split.get(write),
                    )
                )
        return found

    def insert_slice(self, action, call: expr.Call, tag) -> list[Reach]:
        node = self.planner.locations[id(call)]
        if tag == "base":
            sides = [
                (False, self.whole(node, call.args[0], input_index=0)),
                (True, self.whole(node, call, output_index=0)),
            ]
        else:
            sides = [(False, self.whole(node, call.args[1], input_index=1))]
            windows = [reached for *_, write, reached in self.records(call) if write]
            sides += [(True, reached) for reached in windows] or [(True, None)]
        return self.copied(action, node, sides)

    def carry(self, action, loop: core.LoopRegion, tag) -> list[Reach]:
        kind, slot = tag
        inner = self.planner.nodes.get(id(loop))
        node = self.planner.locations[id(loop)]
        source = loop.args[slot] if kind == "entry" else loop.yield_values[slot]
        at = node if kind == "entry" or inner is None else inner
        sides = [
            (False, self.whole(at, source, input_index=0)),
            (True, self.whole(at, loop.params[slot], output_index=0)),
        ]
        return self.copied(action, at, sides)


def mesh_axis_numbering(mesh: types.Mesh, mesh_axis: int):
    """The unit ids of the topology that holds *mesh_axis* of *mesh*, or None when not static.

    Returns the topology's first unit id, the extents and strides of its
    ``unit_layout``, and the mesh axis's index among them. A unit reads
    ``((unit - first_unit_id) // stride) % extent`` on that mesh axis, as the runtime's
    ``mesh_coords`` does.
    """
    keys = utils.axis_keys(mesh)
    if mesh_axis >= len(keys):
        return None
    numbering = topology_numbering(mesh, keys[mesh_axis][0])
    if numbering is None:
        return None
    return (*numbering, mesh_axis - [name for name, _ in keys].index(keys[mesh_axis][0]))


def topology_numbering(mesh: types.Mesh, name: str):
    """Topology *name*'s first unit id, and the extents and strides of its ``unit_layout``.

    None when *mesh* does not name it or they are not static.
    """
    names = utils.topology_names(mesh)
    if name not in names:
        return None
    t = names.index(name)
    try:
        layout = types.plain_layout(utils.unit_layout(mesh, t))
    except ValueError:
        return None
    if layout is None:
        return None
    layout = types.flatten(layout)
    first_unit_id = utils.first_unit_ids(mesh)[t]
    extents, strides = layout.shape, layout.strides
    if not all(isinstance(value, int) for value in (first_unit_id, *extents, *strides)):
        return None
    return first_unit_id, extents, strides


def unit_read(name: str, coordinate: expr.Call, unit: str) -> str | None:
    """The condition that *unit* reads *name* as *coordinate*; None when it has no statement.

    *unit* numbers a CTA of the launch or a thread of a CTA, on the topology that
    holds the coordinate's mesh axis. Every unit reads it: a mesh smaller than its
    topology repeats.
    """
    mesh_axis = expr.static_dim_value(coordinate.args[0])
    numbering = None if mesh_axis is None else mesh_axis_numbering(coordinate.target.mesh, mesh_axis)
    if numbering is None:
        return None
    first_unit_id, extents, strides, index = numbering
    if extents[index] == 1:
        return f"{name} = 0"
    if strides[index] <= 0:
        return None
    return f"{name} = floor(({unit} - {first_unit_id}) / {strides[index]}) mod {extents[index]}"


def unit_guard(mesh: types.Mesh, unit: str, name: str = "cta") -> str | None:
    """The condition that *unit* runs inside topology *name* of *mesh*; None without a statement.

    It is the runtime's ``contains``: a unit of that topology the mesh's layout reaches.
    """
    numbering = topology_numbering(mesh, name)
    if numbering is None:
        return None
    first_unit_id, extents, strides = numbering
    steps = [f"p{mode}" for mode in range(len(extents))]
    held = [f"{unit} = {first_unit_id}" + "".join(f" + {s} * {p}" for s, p in zip(strides, steps))]
    held += [f"0 <= {p} < {e}" for p, e in zip(steps, extents)]
    return f"exists ({', '.join(steps)} : {' and '.join(held)})"


def unit_reach(reach: Reach, suffix: str, shared, topology: str = "cta"):
    """The elements one execution by unit ``__<topology>`` + *suffix* reaches, names renamed.

    *topology* is ``"cta"`` or ``"thread"``. Returns the set, the conditions on that
    unit (None when a coordinate cannot be bound to it), and the names whose
    source the renaming loses. A unit runs the reach only inside the sliced meshes
    of its topology around it; a slice whose units cannot be stated adds no
    condition, which only widens the units.
    """
    relation = reach.reached.relation
    if reach.loops:
        relation = relation.constrain(analysis.iteration_domain(None, None, loops=reach.loops))
    values = dict(relation.values)
    rel = relation.relation
    bindings, bounds, free = [], [], []
    for index in range(rel.dim(isl.dim_type.PARAM)):
        name = rel.get_dim_name(isl.dim_type.PARAM, index)
        value = values.get(name)
        if value is not None and shared(value):
            continue
        if isinstance(value, expr.Call) and isinstance(value.target, dim.MeshCoord):
            mesh_axis = expr.static_dim_value(value.args[0])
            extent = value.target.mesh.shape[mesh_axis]
            bounds.append((f"{name}{suffix}", extent))
            if utils.axis_keys(value.target.mesh)[mesh_axis][0] == topology:
                renamed = f"{name}{suffix}"
                bindings.append((renamed, unit_read(renamed, value, f"__{topology}{suffix}")))
        elif not (
            isinstance(value, expr.Var) and any(s.owner.induction_var is value for s in reach.loops)
        ):
            free.append(name)
        rel = rel.set_dim_name(isl.dim_type.PARAM, index, f"{name}{suffix}")
    for name, extent in bounds:
        rel = rel.intersect_params(isl.set(f"[{name}] -> {{ : 0 <= {name} < {extent} }}"))
    if any(condition is None for _, condition in bindings):
        return rel.range(), None, free
    guards = (unit_guard(mesh, f"__{topology}{suffix}", topology) for mesh in reach.slices)
    bindings += [(None, guard) for guard in guards if guard is not None]
    return rel.range(), bindings, free


def overlap(
    first: Reach, second: Reach, shared, launch: int | None, topology: str = "cta"
) -> Overlap | None:
    """Why executions of *first* and *second* by two different units may reach one element.

    None when they are the same reduction or provably reach no common element.
    Each execution runs in one unit of *topology*, ``__<topology>`` and ``__<topology>_y``,
    below *launch* and inside the runs it keeps to; every coordinate of that topology
    a relation names is the coordinate that unit reads, and the two units differ.
    """
    if first.reduction is not None and first.reduction == second.reduction:
        return None
    if first.reached is None or second.reached is None:
        return Overlap.UNPROVEN
    if first.reached.target != second.reached.target:
        return Overlap.UNPROVEN
    if first.reached.relation.may_access or second.reached.relation.may_access:
        return Overlap.UNPROVEN
    left, bound_x, free_x = unit_reach(first, "", shared, topology)
    right, bound_y, free_y = unit_reach(second, "_y", shared, topology)
    if bound_x is None or bound_y is None:
        return Overlap.UNPROVEN
    left = left.align_params(right.get_space())
    right = right.align_params(left.get_space())
    both = left.intersect(right)
    limit = "" if launch is None else f" < {launch}"
    unit, other = f"__{topology}", f"__{topology}_y"
    conditions = [
        f"0 <= {unit}{limit}",
        f"0 <= {other}{limit}",
        f"({unit} < {other} or {unit} > {other})",
        *(condition for _, condition in (*bound_x, *bound_y)),
    ]
    named = (name for name, _ in (*bound_x, *bound_y) if name is not None)
    names = ", ".join([unit, other, *named])
    apart = isl.set(f"[{names}] -> {{ : {' and '.join(conditions)} }}")
    hit = both.align_params(apart.get_space()).intersect_params(
        apart.align_params(both.get_space())
    )
    if hit.is_empty():
        return None
    return Overlap.UNPROVEN if free_x or free_y else Overlap.CONFLICT


def refuse_cta_overlaps(planner) -> None:
    """Refuse the planner's Function when two CTAs may reach one global element unordered.

    It pairs the global reaches of which one writes; a reach pairs with itself
    too. A launch of one CTA has no pairs.
    """
    sizes = [t.size for t in planner.context.topologies or () if t.name == "cta"]
    launch = prod(sizes) if all(isinstance(size, int) for size in sizes) else None
    if launch == 1:
        return
    reaches = Reaches(planner)
    found = [reach for action in planner.actions.values() for reach in reaches.of(action)]
    params = {id(param) for param in planner.function.params}

    def shared(value) -> bool:
        return (isinstance(value, expr.Var) and id(value) in params) or isinstance(
            getattr(value, "target", value), expr.DimVar
        )

    for index, first in enumerate(found):
        for second in found[index:]:
            if not (first.write or second.write):
                continue
            if None not in (first.root, second.root) and first.root != second.root:
                continue
            verdict = overlap(first, second, shared, launch)
            if verdict is not None:
                what = "reach" if verdict is Overlap.CONFLICT else "may reach"
                raise analysis.Error(
                    f"sync: {first.name} and {second.name} {what} one global element "
                    "from two CTAs, which no step orders"
                )


__all__ = ["Overlap", "Reaches", "mesh_axis_numbering", "overlap", "refuse_cta_overlaps"]
