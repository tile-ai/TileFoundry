"""Measure authored-loop buffer access without requiring a schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import isl

from tilefoundry.ir.core import Call, Constant, Expr, Var, binding_name
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import shard_layout_of
from tilefoundry.ir.types.shard.shard_layout import split_target_axes
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    access_relation_registry,
    index_set,
    relation_of,
    relations_of,
    renaming_relation,
)
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext

from .poly.affine import LoopAffineTerm, loop_affine_term
from .visitor import StructuralMemo
from .walk import loop_trip_count


class _Unavailable(Exception):
    """An access that cannot be represented by this authored-loop model."""


@dataclass(frozen=True)
class _Access:
    """One access from a loop-prefixed statement domain to its source buffer."""

    buffer_id: int
    buffer: str
    level: str
    bit_width: int
    call: Call = field(repr=False)
    path: tuple[GridRegionExpr, ...]
    per_call_elements: int
    relation: isl.map = field(repr=False)
    domain: isl.set = field(repr=False)


@dataclass(frozen=True)
class _BufferReading:
    buffer: str
    level: str
    bytes: int
    device_bytes: int
    repeated_bytes: int


@dataclass(frozen=True)
class _LoopReading:
    buffers: tuple[_BufferReading, ...]
    known: bool


@dataclass(frozen=True)
class _RelationCase:
    """One domain of a Call and the original operands modeled on that domain."""

    domain: isl.set
    relations: AccessRelations
    local_types: tuple[object, ...]


def _local_type(type_: object) -> object:
    """Narrow every Split axis while preserving the tensor's logical rank."""
    if not isinstance(type_, TensorType):
        return type_
    layout = shard_layout_of(type_.layout)
    if layout is None:
        return type_
    local = list(type_.shape)
    for mesh_axis, tensor_axis in enumerate(split_target_axes(layout, type_.shape)):
        if tensor_axis is None:
            continue
        extent = layout.mesh.layout.shape[mesh_axis]
        if extent is None:
            local[tensor_axis] = 1
            continue
        size = local[tensor_axis]
        if not isinstance(size, int) or isinstance(size, bool) or size % extent != 0:
            raise _Unavailable
        local[tensor_axis] = size // extent
    return TensorType(shape=tuple(local), dtype=type_.dtype, layout=None, storage=type_.storage)


def _static_loop_bound(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise _Unavailable


def _loop_domain(inner: isl.set, loops: tuple[GridRegionExpr, ...]) -> isl.set:
    """Prefix a Call domain with its outermost-first authored loop axes."""
    if not loops:
        return inner
    params: dict[str, DimVar] = {}
    bounds: list[str] = []
    for index, loop in enumerate(loops):
        start = _static_loop_bound(loop.start)
        step = _static_loop_bound(loop.step)
        if isinstance(loop.extent, DimVar):
            params[loop.extent.name] = loop.extent
            stop = loop.extent.name
        else:
            stop = str(_static_loop_bound(loop.extent))
        bounds.append(f"{start} <= p{index} < {stop}")
        if step != 1:
            bounds.append(f"(p{index} - {start}) mod {step} = 0")
    for name, dim in params.items():
        bounds.append(f"{dim.lo} <= {name} < {dim.hi}")
    own_rank = inner.dim(isl.dim_type.SET)
    dims = [f"p{i}" for i in range(len(loops))]
    dims.extend(f"f{i}" for i in range(own_rank))
    prefix = f"[{', '.join(params)}] -> " if params else ""
    box = isl.set(f"{prefix}{{ [{', '.join(dims)}] : {' and '.join(bounds)} }}")
    return inner.insert_dims(isl.dim_type.SET, 0, len(loops)).intersect(box)


def _source_buffer(
    expr: Expr,
    access: isl.map,
    loops: tuple[GridRegionExpr, ...],
    ctx: TypeInferContext,
    *,
    narrow: bool,
) -> tuple[Expr, isl.map]:
    """Fold each view's coordinates until *access* reaches an allocation.

    The coordinates come from the view's own registered relation, read backwards
    through what it writes and forwards through what it reads, so a window's
    stride and a reshape's arithmetic are stated once for every reader of them.
    """
    while isinstance(expr, Call) and isinstance(expr.target, (Slice, Reshape)):
        folded = renaming_relation(expr, ctx)
        source = expr.args[0]
        held = _local_type(source.type) if narrow else source.type
        if not isinstance(held, TensorType):
            raise _Unavailable
        access = access.apply_range(relation_of(folded))
        access = _bind_parameters(
            access, folded, loops, held, access.domain(), narrow=narrow
        )
        expr = source
    return expr, access


@dataclass
class _RankPreserving(TypeInferContext):
    """A context whose local view narrows a Split axis and keeps the rank.

    A footprint is a relation over a tensor's logical axes, so a local view that
    factored them into layout positions would lose the very flow this model is
    for. Handlers ask their context this question, which is what lets one
    registered relation answer both the whole program and one participant.
    """

    def local_type_of(self, expr: Expr) -> object:
        return _local_type(expr.type)


def _placed_parameter(
    value: object, loops: tuple[GridRegionExpr, ...], *, narrow: bool
) -> LoopAffineTerm | None:
    """Where in this loop nest a parameter's value sits, when it sits anywhere."""
    if not isinstance(value, Expr):
        return None
    return loop_affine_term(value, loops, narrow=narrow)


def _widest_allowed(
    access: isl.map, name: str, held: object
) -> LoopAffineTerm | None:
    """The value a parameter may take that reaches the most of its operand.

    A footprint is an upper bound, so a parameter nobody here can place takes
    whichever end of its legal range touches more: where a window sits does not
    change how much of it there is, but how long it is does. Both ends are the
    Op's own contract, read off the relation rather than guessed.
    """
    names = [
        access.get_dim_name(isl.dim_type.PARAM, index)
        for index in range(access.dim(isl.dim_type.PARAM))
    ]
    space = f"[{', '.join(names)}] -> "
    probe = isl.set(f"{space}{{ [x] : x = {name} }}").intersect_params(access.params())
    ends = (probe.dim_min_val(0), probe.dim_max_val(0))
    if not all(end.is_int() for end in ends):
        return None
    box = index_set(tuple(held.shape)) if isinstance(held, TensorType) else None
    if box is None or box.dim(isl.dim_type.SET) != access.dim(isl.dim_type.OUT):
        least = ends[0].get_num_si()
        return LoopAffineTerm(None, 0, least, least)
    best: tuple[int, int] | None = None
    for value in sorted({end.get_num_si() for end in ends}):
        reach = access.intersect_params(
            isl.set(f"{space}{{ : {name} = {value} }}")
        ).range().intersect(box)
        amount = reach.coalesce().count_val()
        if not amount.is_int():
            return None
        if best is None or amount.get_num_si() > best[0]:
            best = (amount.get_num_si(), value)
    return None if best is None else LoopAffineTerm(None, 0, best[1], best[1])


def _bind_parameters(
    access: isl.map,
    pattern: object,
    loops: tuple[GridRegionExpr, ...],
    held: object,
    domain: isl.set,
    *,
    narrow: bool,
) -> isl.map:
    """Say what each of a relation's parameters is, in this loop nest's terms.

    A window's offset is bound to the operand that states it, and that operand
    may be an induction variable: then the parameter is the loop coordinate, and
    projecting it out unions the window over the trip. The relation is held to
    the domain it runs over first, because what a parameter may be follows from
    where the Call walks. One nobody here can place takes whichever end of its
    legal range reaches the most, so a window of unknown position still counts
    as the window it is rather than as everywhere it could have gone.
    """
    bound = dict(getattr(pattern, "parameters", ()) or ())
    access = access.intersect_domain(domain)
    while access.dim(isl.dim_type.PARAM):
        name = access.get_dim_name(isl.dim_type.PARAM, 0)
        value = bound.get(name)
        number = static_dim_value(value)
        if number is not None:
            term = LoopAffineTerm(None, 0, number, number)
        else:
            term = _placed_parameter(value, loops, narrow=narrow)
            if term is None:
                term = _widest_allowed(access, name, held)
            if term is None:
                access = access.project_out(isl.dim_type.PARAM, 0, 1)
                continue
        local = isl.local_space.from_space(access.get_space())

        def placed(constraint: isl.constraint, sign: int) -> isl.constraint:
            constraint = constraint.set_coefficient_si(isl.dim_type.PARAM, 0, sign)
            if term.loop_axis is not None:
                constraint = constraint.set_coefficient_si(
                    isl.dim_type.IN, term.loop_axis, -sign * term.stride
                )
            return constraint

        if term.low == term.high:
            equality = placed(isl.constraint.alloc_equality(local), 1)
            access = access.add_constraint(equality.set_constant_si(-term.low))
        else:
            floor = placed(isl.constraint.alloc_inequality(local), 1)
            ceiling = placed(isl.constraint.alloc_inequality(local), -1)
            access = access.add_constraint(floor.set_constant_si(-term.low))
            access = access.add_constraint(ceiling.set_constant_si(term.high))
        access = access.project_out(isl.dim_type.PARAM, 0, 1)
    return access


def _within(access: isl.map, held: TensorType) -> isl.map:
    """The part of a relation that lands on coordinates the operand has.

    A relation stated over a container reaches past a participant that holds part
    of it, and a boundary answers for what it was given: reaching a coordinate
    nobody holds is not reaching.
    """
    box = index_set(tuple(held.shape))
    if box is None or box.dim(isl.dim_type.SET) != access.dim(isl.dim_type.OUT):
        return access
    return access.intersect_range(box)


def _touched(call: Call, relations: AccessRelations):
    """Each operand this Call reaches into, paired with the boundary saying where.

    One boundary per operand, which is what the operand's own relation says it
    reaches. A result is somewhere of its own here, so an Op that leaves part of
    a container alone reaches only the part it read: the rest it kept is bytes it
    did not touch on this side.
    """
    for index, boundary in enumerate(relations.inputs):
        if index < len(call.args):
            yield index, boundary


def _relation_cases(
    call: Call, ctx: TypeInferContext, *, narrow: bool
) -> tuple[_RelationCase, ...]:
    """The registered relations of a Call, on the space that Call iterates.

    An access map's domain is the Op's iteration space, so the space to walk is
    read off the relations rather than rebuilt from a result's extents: a
    contraction walks its contracted axis and a reduction walks what it reads,
    and neither has the shape of what it produces.
    """
    if access_relation_registry.lookup(type(call.target)) is None:
        raise _Unavailable
    try:
        relations = relations_of(call, ctx)
        domain = _iterated(relations)
        if domain is None or not domain.is_bounded():
            raise _Unavailable
        local_types = tuple(
            ctx.local_type_of(arg) if narrow else arg.type for arg in call.args
        )
    except (NotImplementedError, TypeError, ValueError, isl.Error) as error:
        raise _Unavailable from error
    return (
        _RelationCase(domain=domain, relations=relations, local_types=local_types),
    )


def _iterated(relations: AccessRelations) -> "isl.set | None":
    """The space a Call's own relations say it walks.

    Every boundary of one Op is asked by the same coordinates, so any of them
    names the space; a boundary that is partial in it names less, and the union
    is what the Call actually ran.
    """
    walked = None
    for boundary in (*relations.outputs, *relations.inputs):
        try:
            own = relation_of(boundary.pattern).domain()
        except (TypeError, ValueError, isl.Error):
            return None
        walked = own if walked is None else walked.union(own)
    if walked is None:
        return None
    return _without_parameters(walked).coalesce()


def _without_parameters(walked: "isl.set") -> "isl.set":
    """One space with its parameters taken out, so it stands for coordinates only.

    An offset a boundary is waiting for narrows that boundary, not the space the
    Call walks; leaving it in makes a set nobody can count, and isl answers zero
    for one of those rather than refusing.
    """
    count = walked.dim(isl.dim_type.PARAM)
    return walked if not count else walked.project_out(isl.dim_type.PARAM, 0, count)


def _labels(fn: Function, structural_memo: StructuralMemo) -> dict[int, str]:
    values = structural_memo.definition_order(fn)
    order: list[Expr] = [
        *fn.params,
        *(expr for expr in values if isinstance(expr, (Call, Constant))),
    ]
    seen = {id(expr) for expr in order}
    order.extend(expr for expr in values if id(expr) not in seen)
    taken: set[str] = set()
    labels: dict[int, str] = {}
    for position, expr in enumerate(order):
        base = expr.name if isinstance(expr, Var) else binding_name(expr) or f"<value {position}>"
        name, suffix = base, 2
        while name in taken:
            name = f"{base}_{suffix}"
            suffix += 1
        taken.add(name)
        labels[id(expr)] = name
    return labels


def _enclosing_loops(
    call: Call,
    loops: dict[int, GridRegionExpr],
    structural_memo: StructuralMemo,
) -> tuple[GridRegionExpr, ...]:
    return tuple(
        sorted(
            (
                loop
                for loop in loops.values()
                if structural_memo.scope(loop).is_variant(call)
            ),
            key=lambda loop: -structural_memo.node(loop).definition_index,
        )
    )


def _collect(
    module: Module, fn: Function, structural_memo: StructuralMemo, *, narrow: bool = True
) -> tuple[
    dict[int, GridRegionExpr],
    dict[int, list[_Access]],
    dict[int, list[Call]],
]:
    """Collect each directly owned Call's accesses and unavailable relation scopes."""
    values = structural_memo.definition_order(fn)
    loops = {id(expr): expr for expr in values if isinstance(expr, GridRegionExpr)}
    labels = _labels(fn, structural_memo)
    scope = FunctionScope(module, fn)
    ctx = _RankPreserving(scope=scope) if narrow else TypeInferContext(scope=scope)
    accesses: dict[int, list[_Access]] = {key: [] for key in loops}
    unavailable: dict[int, list[Call]] = {key: [] for key in loops}
    for call in (expr for expr in values if isinstance(expr, Call)):
        path = _enclosing_loops(call, loops, structural_memo)
        if not path or isinstance(call.target, (Slice, Reshape, TupleGetItem)):
            continue
        scope = id(path[-1])
        try:
            cases = _relation_cases(call, ctx, narrow=narrow)
        except _Unavailable:
            for loop in path:
                unavailable[id(loop)].append(call)
            continue
        call_unavailable = False
        for case in cases:
            try:
                domain = _loop_domain(case.domain, path)
            except _Unavailable:
                call_unavailable = True
                break
            depth = len(path)
            for index, boundary in _touched(call, case.relations):
                operand = call.args[index]
                raw = relation_of(boundary.pattern)
                local_type = case.local_types[index]
                if raw.dim(isl.dim_type.OUT) == 0 or not isinstance(local_type, TensorType):
                    continue
                access = raw.insert_dims(isl.dim_type.IN, 0, depth)
                try:
                    access = _bind_parameters(
                        access, boundary.pattern, path, local_type, domain, narrow=narrow
                    )
                    access = _within(access, local_type)
                    elements = _one_pass(access, domain, depth)
                    buffer, access = _source_buffer(
                        operand, access, path, ctx, narrow=narrow
                    )
                    if not isinstance(buffer.type, TensorType):
                        continue
                except (NotImplementedError, TypeError, ValueError, isl.Error, _Unavailable):
                    call_unavailable = True
                    break
                accesses[scope].append(
                    _Access(
                        buffer_id=id(buffer),
                        buffer=labels[id(buffer)],
                        level=str(buffer.type.storage),
                        bit_width=buffer.type.dtype.bit_width,
                        call=call,
                        path=path,
                        per_call_elements=elements,
                        relation=access.intersect_domain(domain),
                        domain=domain,
                    )
                )
            if call_unavailable:
                break
        if call_unavailable:
            for loop in path:
                unavailable[id(loop)].append(call)
    return loops, accesses, unavailable


def _one_pass(access: isl.map, domain: isl.set, depth: int) -> int:
    """How many elements one iteration reaches, with every loop standing still.

    This is what would move if nothing were reused, so it is asked of the same
    relation as the union rather than of the operand's size: charging a whole
    cache for the rows one pass replaced is how a reuse figure stops meaning
    anything.
    """
    standing = domain
    for axis in range(depth):
        standing = standing.fix_si(isl.dim_type.SET, axis, standing.dim_min_val(axis).get_num_si())
    reached = access.intersect_domain(standing).range()
    if reached.dim(isl.dim_type.PARAM):
        raise _Unavailable
    amount = reached.coalesce().count_val()
    if not amount.is_int():
        raise _Unavailable
    return amount.get_num_si()


def _reached(access: _Access, loop: GridRegionExpr) -> isl.set:
    """The source elements reached while *loop* and its descendants vary."""
    current = next(index for index, candidate in enumerate(access.path) if candidate is loop)
    domain = access.domain
    for axis, outer in enumerate(access.path[:current]):
        domain = domain.fix_si(isl.dim_type.SET, axis, _static_loop_bound(outer.start))
    return access.relation.intersect_domain(domain).range()


def _reached_elements(sets: list[isl.set]) -> int:
    """Exact integer-point count of each rank's unioned access image."""
    merged: dict[int, isl.set] = {}
    for set_ in sets:
        rank = set_.dim(isl.dim_type.SET)
        merged[rank] = set_ if rank not in merged else merged[rank].union(set_)
    total = 0
    for set_ in merged.values():
        if set_.is_empty():
            continue
        if set_.dim(isl.dim_type.PARAM):
            raise _Unavailable
        amount = set_.coalesce().count_val()
        if not amount.is_int():
            raise _Unavailable
        total += amount.get_num_si()
    return total


def _packed_bytes(elements: int, bit_width: int) -> int:
    return math.ceil(elements * bit_width / 8)


def _reading(
    loop: GridRegionExpr,
    accesses_by_scope: dict[int, list[_Access]],
    *,
    known: bool,
    validate_repeated: bool = True,
) -> _LoopReading:
    grouped: dict[tuple[int, str, int], list[_Access]] = {}
    for accesses in accesses_by_scope.values():
        for access in accesses:
            if any(candidate is loop for candidate in access.path):
                grouped.setdefault((access.buffer_id, access.level, access.bit_width), []).append(
                    access
                )
    buffers: list[_BufferReading] = []
    complete = known
    for (_, level, bit_width), group in sorted(
        grouped.items(), key=lambda item: (item[1][0].buffer, item[0][1])
    ):
        try:
            repeated = sum(
                access.per_call_elements
                * math.prod(
                    loop_trip_count(nested)
                    for nested in access.path[
                        next(
                            index
                            for index, candidate in enumerate(access.path)
                            if candidate is loop
                        ) :
                    ]
                )
                for access in group
            )
            reached = _reached_elements([_reached(access, loop) for access in group])
            bytes_ = _packed_bytes(reached, bit_width)
            repeated_bytes = _packed_bytes(repeated, bit_width)
            if validate_repeated and bytes_ > repeated_bytes:
                raise _Unavailable
            buffers.append(
                _BufferReading(
                    buffer=group[0].buffer,
                    level=level,
                    bytes=bytes_,
                    device_bytes=0,
                    repeated_bytes=repeated_bytes,
                )
            )
        except (ValueError, isl.Error, _Unavailable):
            complete = False
    return _LoopReading(tuple(buffers), complete)


def loop_footprints(
    module: Module, fn: Function, structural_memo: StructuralMemo
) -> dict[int, _LoopReading]:
    """Return each authored loop's known reading or best available lower bound."""
    loops, accesses, unavailable = _collect(module, fn, structural_memo)
    _, device_accesses, device_unavailable = _collect(
        module, fn, structural_memo, narrow=False
    )

    readings: dict[int, _LoopReading] = {}
    for loop_id, loop in loops.items():
        local = _reading(
            loop,
            accesses,
            known=not unavailable[loop_id],
        )
        device = _reading(
            loop,
            device_accesses,
            known=not device_unavailable[loop_id],
            validate_repeated=False,
        )
        local_rows = {(item.buffer, item.level): item for item in local.buffers}
        device_rows = {(item.buffer, item.level): item for item in device.buffers}
        keys = sorted(local_rows.keys() | device_rows.keys())
        readings[loop_id] = _LoopReading(
            buffers=tuple(
                _BufferReading(
                    buffer=buffer,
                    level=level,
                    bytes=(local_rows[key].bytes if key in local_rows else 0),
                    device_bytes=(device_rows[key].bytes if key in device_rows else 0),
                    repeated_bytes=(local_rows[key].repeated_bytes if key in local_rows else 0),
                )
                for key in keys
                for buffer, level in (key,)
            ),
            known=local.known and device.known,
        )
    return readings


__all__ = ["loop_footprints"]
