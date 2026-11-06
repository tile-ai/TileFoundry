"""Global memory that two CTAs may both reach while one of them goes through the async proxy.

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

from tilefoundry.analysis.access import Access
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.loop_domain import iteration_domain
from tilefoundry.ir.core import Call, SSAValue, Var
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.mesh_scope import device_layout
from tilefoundry.ir.types import Mesh, StorageKind
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.layout import ComposedLayout, flatten
from tilefoundry.ir.types.mesh import axis_keys, separate
from tilefoundry.ir.types.utils import leaves_of, static_dim_value
from tilefoundry.passes.transforms.convert_hir_to_tir import operand_types, selected_op
from tilefoundry.visitor_registry.access_relation import AccessRelation, identity_access
from tilefoundry.visitor_registry.instruction import call_params
from tilefoundry_nvidia.cuda.analysis.completion import Proxy
from tilefoundry_nvidia.cuda.analysis.emission import operand_roles
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_reduce_async_tensor import CopyReduceAsyncTensor


class Overlap(Enum):
    """Why two accesses from two different CTAs may reach one element."""

    CONFLICT = auto()
    UNPROVEN = auto()


@dataclass(frozen=True)
class _Reach:
    """One read or write of an action on a global memory root.

    ``access`` is None when the relation is unknown: a retained call, or a value
    with no device coordinates. ``slices`` are the sliced CTA meshes around it,
    the only meshes that keep a CTA from running it.
    """

    name: str
    write: bool
    async_proxy: bool
    reduction: object | None
    root: object
    access: Access | None
    loops: tuple = ()
    slices: tuple = ()


def _storage(value: SSAValue):
    leaves = leaves_of(value.expr.type)
    return getattr(leaves[value.index], "storage", None) if value.index < len(leaves) else None


class _Reaches:
    """The global memory reads and writes of a planner's actions, with their device relations."""

    def __init__(self, planner):
        self.planner = planner
        self.scope = planner.root
        self.summary = planner.summary

    def physical(self, access: Access) -> Access:
        """*access* through the address its target's storage takes."""
        if access.target not in self.planner.values:
            return access
        alias = self.summary.address_alias(access.target)
        if alias.target == access.target:
            return access
        relation = access.relation.compose(alias.relation)
        return replace(access, target=alias.target, relation=relation)

    def records(self, call: Call):
        """The device reads and writes the scope records for *call*, with their paths."""
        found = []
        for side, write in ((self.scope.access.accesses, False), (self.scope.access.outputs, True)):
            for path, (primitive, accesses) in side.get("device", {}).items():
                if primitive is call:
                    found.extend((path, write, access) for access in accesses)
        return found

    def whole(self, node, value, *, input_index=None, output_index=None) -> Access | None:
        """Every element of *value*'s first leaf, through its own device coordinates."""
        leaf = SSAValue(value, 0)
        leaves = leaves_of(value.type)
        if not leaves or getattr(leaves[0], "storage", None) is not StorageKind.GMEM:
            return None
        entry = node.provenance.lookup(leaf)
        if entry.device is None:
            return None
        shape = tuple(leaves[0].shape)
        values = {}
        held = shape_to_isl_set(shape, values)
        every = identity_access(len(shape)).relation.intersect_range(held)
        relation = AccessRelation(every, values)
        return Access(input_index, relation.compose(entry.device), entry.root, output_index)

    def of(self, action) -> list[_Reach]:
        """Each global memory read and write of *action*, as the planner dispatched it."""
        owner, tag = action.owner, action.tag
        if not self.gmem(action):
            return []
        if isinstance(owner, LoopRegion):
            return self.carry(action, owner, tag)
        if tag in ("base", "update"):
            return self.insert_slice(action, owner, tag)
        if self.planner.locations[id(owner)].attachment(owner) is not None:
            return self.opaque(action)
        if self.planner.visit(owner) in (self.planner.visit_Zeros, self.planner.visit_FullLike):
            return self.fill(action, owner)
        return self.instruction(action, owner, selected_op(owner, self.planner.context.target))

    def gmem(self, action):
        """The uses of *action* on global memory roots."""
        return [use for use in action.uses if use.global_roots]

    def root(self, target: SSAValue):
        """The allocation root the planner gives *target*, None when it has none."""
        if target not in self.planner.values:
            return None
        return self.summary.root(target).value

    def opaque(self, action) -> list[_Reach]:
        """A retained call's reaches, whose relations stay unknown."""
        return [
            _Reach(
                action.name, use.effect.write, use.effect.proxy is Proxy.ASYNC, None, root, None
            )
            for use in self.gmem(action)
            for root in use.global_roots
        ]

    def slices(self, node) -> tuple:
        """The sliced CTA meshes of the regions around *node*; only these guard who runs it."""
        return tuple(
            scope.owner.mesh
            for scope in node.ancestors()
            if isinstance(scope.owner, MeshRegion)
            and isinstance(scope.owner.mesh.layout, ComposedLayout)
            and any(topology.name == "cta" for topology in scope.owner.mesh.topologies)
        )

    def bound(self, action, call: Call, recorded, reads, writes) -> list[_Reach]:
        """Reaches for *action*'s global uses, from *call*'s ``(path, write, access)`` records."""
        node = self.planner.locations[id(call)]
        loops, slices = node.loop_scopes, self.slices(node)
        found = []
        for use in self.gmem(action):
            param = use.effect.param
            async_proxy = use.effect.proxy is Proxy.ASYNC
            chosen = [
                item
                for item in recorded
                if not item[1] and item[2].input_index == reads.get(param)
            ]
            if not chosen and param in writes:
                chosen = [item for item in recorded if item[1]]
            for _path, _write, access in chosen:
                access = self.physical(access)
                if _storage(access.target) is not StorageKind.GMEM:
                    continue
                found.append(
                    _Reach(
                        f"{action.name} {param}",
                        use.effect.write,
                        async_proxy,
                        None,
                        self.root(access.target),
                        access,
                        loops,
                        slices,
                    )
                )
            if not chosen:
                name, write = f"{action.name} {param}", use.effect.write
                found.extend(
                    _Reach(name, write, async_proxy, None, root, None)
                    for root in use.global_roots
                )
        return found

    def instruction(self, action, call: Call, op) -> list[_Reach]:
        params = call_params(op, len(operand_types(call, op)))
        supplied, produced = operand_roles(call, params)
        reads = {param.name: index for index, param in enumerate(supplied)}
        writes = {param.name for param in produced}
        found = self.bound(action, call, self.records(call), reads, writes)
        kind = op.kind if isinstance(op, CopyReduceAsyncTensor) else None
        return [replace(reach, reduction=kind) if reach.async_proxy else reach for reach in found]

    def fill(self, action, call: Call) -> list[_Reach]:
        recorded = [item for item in self.records(call) if item[1]]
        writes = {use.effect.param for use in self.gmem(action) if use.effect.write}
        return self.bound(action, call, recorded, {}, writes)

    def copied(self, action, node, sides) -> list[_Reach]:
        """Reaches of a Copy lowering adds at *node*, from its ``(write, access)`` pairs."""
        loops, slices = node.loop_scopes, self.slices(node)
        found = []
        for write, access in sides:
            if access is None:
                found.extend(
                    _Reach(action.name, use.effect.write, False, None, root, None)
                    for use in self.gmem(action)
                    if use.effect.write == write
                    for root in use.global_roots
                )
                continue
            access = self.physical(access)
            if _storage(access.target) is StorageKind.GMEM:
                root = self.root(access.target)
                found.append(
                    _Reach(action.name, write, False, None, root, access, loops, slices)
                )
        return found

    def insert_slice(self, action, call: Call, tag) -> list[_Reach]:
        node = self.planner.locations[id(call)]
        if tag == "base":
            sides = [
                (False, self.whole(node, call.args[0], input_index=0)),
                (True, self.whole(node, call, output_index=0)),
            ]
        else:
            sides = [(False, self.whole(node, call.args[1], input_index=1))]
            windows = [access for _path, write, access in self.records(call) if write]
            sides += [(True, access) for access in windows] or [(True, None)]
        return self.copied(action, node, sides)

    def carry(self, action, loop: LoopRegion, tag) -> list[_Reach]:
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


def axis_numbering(mesh: Mesh, axis: int):
    """The numbering of the level of *mesh* that holds *axis*, or None when it is not static.

    Returns the level's start, extents and strides, and the axis's index among
    them. A unit reads ``((unit - start) // stride) % extent`` on that axis, as
    the runtime's ``mesh_coords`` does.
    """
    first = 0
    for level in separate(mesh):
        layout = device_layout(level)
        if axis < first + len(layout.shape):
            break
        first += len(layout.shape)
    else:
        return None
    start = level.layout.offset if isinstance(level.layout, ComposedLayout) else 0
    extents, strides = layout.shape, layout.strides
    if not all(isinstance(value, int) for value in (start, *extents, *strides)):
        return None
    return start, extents, strides, axis - first


def _cta_read(name: str, coordinate: Call, unit: str) -> str | None:
    """The condition that CTA *unit* reads *name* as *coordinate*; None when it has no statement.

    Every CTA of the launch reads it: a mesh smaller than its topology repeats.
    """
    axis = static_dim_value(coordinate.args[0])
    numbering = None if axis is None else axis_numbering(coordinate.target.mesh, axis)
    if numbering is None:
        return None
    start, extents, strides, index = numbering
    if extents[index] == 1:
        return f"{name} = 0"
    if strides[index] <= 0:
        return None
    return f"{name} = floor(({unit} - {start}) / {strides[index]}) mod {extents[index]}"


def _cta_guard(mesh: Mesh, unit: str) -> str | None:
    """The condition that CTA *unit* runs inside the sliced *mesh*; None when it has no statement.

    It is the runtime's ``contains``: a CTA position the slice's layout reaches.
    """
    level = next((level for level in separate(mesh) if level.topologies[0].name == "cta"), None)
    if level is None:
        return None
    layout = device_layout(level)
    start = level.layout.offset if isinstance(level.layout, ComposedLayout) else 0
    extents, strides = layout.shape, layout.strides
    if not all(isinstance(value, int) for value in (start, *extents, *strides)):
        return None
    steps = [f"p{position}" for position in range(len(extents))]
    held = [f"{unit} = {start}" + "".join(f" + {s} * {p}" for s, p in zip(strides, steps))]
    held += [f"0 <= {p} < {e}" for p, e in zip(steps, extents)]
    return f"exists ({', '.join(steps)} : {' and '.join(held)})"


def _instance(reach: _Reach, suffix: str, shared):
    """The elements one execution in CTA ``__cta`` + *suffix* reaches, its own names renamed.

    Returns the set, the conditions on that CTA (None when a coordinate cannot be
    bound to it), and the names whose source the renaming loses. A slice whose
    positions cannot be stated adds no condition, which only widens the CTAs.
    """
    relation = reach.access.relation
    if reach.loops:
        relation = relation.constrain(iteration_domain(None, None, loops=reach.loops))
    values = dict(relation.values)
    rel = relation.relation
    bindings, bounds, free = [], [], []
    for index in range(rel.dim(isl.dim_type.PARAM)):
        name = rel.get_dim_name(isl.dim_type.PARAM, index)
        value = values.get(name)
        if value is not None and shared(value):
            continue
        if isinstance(value, Call) and isinstance(value.target, MeshCoord):
            axis = static_dim_value(value.args[0])
            extent = flatten(value.target.mesh.layout).shape[axis]
            bounds.append((f"{name}{suffix}", extent))
            if axis_keys(value.target.mesh)[axis][0] == "cta":
                renamed = f"{name}{suffix}"
                bindings.append((renamed, _cta_read(renamed, value, f"__cta{suffix}")))
        elif not (
            isinstance(value, Var) and any(s.owner.induction_var is value for s in reach.loops)
        ):
            free.append(name)
        rel = rel.set_dim_name(isl.dim_type.PARAM, index, f"{name}{suffix}")
    for name, extent in bounds:
        rel = rel.intersect_params(isl.set(f"[{name}] -> {{ : 0 <= {name} < {extent} }}"))
    if any(condition is None for _name, condition in bindings):
        return rel.range(), None, free
    guards = (_cta_guard(mesh, f"__cta{suffix}") for mesh in reach.slices)
    bindings += [(None, guard) for guard in guards if guard is not None]
    return rel.range(), bindings, free


def _overlap(first: _Reach, second: _Reach, shared, launch: int | None) -> Overlap | None:
    """Why executions of *first* and *second* in two different CTAs may reach one element.

    None when they are the same reduction or provably reach no common element.
    Each execution runs in one physical CTA, ``__cta`` and ``__cta_y``, below
    *launch* and inside the slices around it; every CTA coordinate a relation
    names is the coordinate that CTA reads, and the two CTAs differ.
    """
    if first.reduction is not None and first.reduction == second.reduction:
        return None
    if first.access is None or second.access is None:
        return Overlap.UNPROVEN
    if first.access.target != second.access.target:
        return Overlap.UNPROVEN
    if first.access.relation.may_access or second.access.relation.may_access:
        return Overlap.UNPROVEN
    left, bound_x, free_x = _instance(first, "", shared)
    right, bound_y, free_y = _instance(second, "_y", shared)
    if bound_x is None or bound_y is None:
        return Overlap.UNPROVEN
    left = left.align_params(right.get_space())
    right = right.align_params(left.get_space())
    both = left.intersect(right)
    limit = "" if launch is None else f" < {launch}"
    conditions = [
        f"0 <= __cta{limit}",
        f"0 <= __cta_y{limit}",
        "(__cta < __cta_y or __cta > __cta_y)",
        *(condition for _name, condition in (*bound_x, *bound_y)),
    ]
    named = (name for name, _condition in (*bound_x, *bound_y) if name is not None)
    names = ", ".join(["__cta", "__cta_y", *named])
    apart = isl.set(f"[{names}] -> {{ : {' and '.join(conditions)} }}")
    hit = both.align_params(apart.get_space()).intersect_params(
        apart.align_params(both.get_space())
    )
    if hit.is_empty():
        return None
    return Overlap.UNPROVEN if free_x or free_y else Overlap.CONFLICT


def refuse_cta_overlaps(planner) -> None:
    """Refuse the planner's Function when two CTAs may reach one global element unordered.

    It pairs the global reaches of which one goes through the async proxy and one
    writes; a reach pairs with itself too. A launch of one CTA has no pairs.
    """
    sizes = [t.size for t in planner.context.topologies or () if t.name == "cta"]
    launch = prod(sizes) if all(isinstance(size, int) for size in sizes) else None
    if launch == 1:
        return
    reaches = _Reaches(planner)
    found = [reach for action in planner.actions.values() for reach in reaches.of(action)]
    params = {id(param) for param in planner.function.params}

    def shared(value) -> bool:
        return (isinstance(value, Var) and id(value) in params) or isinstance(
            getattr(value, "target", value), DimVar
        )

    for index, first in enumerate(found):
        for second in found[index:]:
            if not (first.async_proxy or second.async_proxy) or not (first.write or second.write):
                continue
            if None not in (first.root, second.root) and first.root != second.root:
                continue
            verdict = _overlap(first, second, shared, launch)
            if verdict is not None:
                what = "reach" if verdict is Overlap.CONFLICT else "may reach"
                raise AnalysisError(
                    f"sync: {first.name} and {second.name} {what} one global element "
                    "from two CTAs, which no step orders"
                )


__all__ = ["refuse_cta_overlaps"]
