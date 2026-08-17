"""Measure authored-loop buffer access without requiring a schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import isl

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var, binding_name
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.nn.rope import RoPE
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.reshape import Reshape, flat_reshape_map
from tilefoundry.ir.hir.tensor.slice import Slice, window_base
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import TensorType, numel
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import shard_layout_of
from tilefoundry.ir.types.shard.shard_layout import split_target_axes
from tilefoundry.visitor_registry.access_relation import AccessRelationResult, build_relation
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext
from tilefoundry.visitor_registry.isl_utility import to_domain
from tilefoundry.visitor_registry.relation_build import identity_map

from .walk import loop_repeated_values, loop_trip_count, postorder


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

    result: AccessRelationResult
    operand_maps: tuple[tuple[int, int], ...]
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


def _constant_int(expr: Expr) -> int | None:
    if (
        isinstance(expr, Constant)
        and isinstance(expr.value, int)
        and not isinstance(expr.value, bool)
    ):
        return int(expr.value)
    return None


def _static_range(expr: Expr, *, narrow: bool) -> tuple[int, int] | None:
    """Bound one mesh-position expression, or erase its fixed local translation."""
    value = _constant_int(expr)
    if value is not None:
        return value, value
    if not isinstance(expr, Call):
        return None
    if isinstance(expr.target, Local):
        return (0, 0) if narrow else _static_range(expr.args[0], narrow=narrow)
    if isinstance(expr.target, (Reshape, Reshard)):
        return _static_range(expr.args[0], narrow=narrow)
    if isinstance(expr.target, Arange):
        start, step = expr.target.start, expr.target.step
        (length,) = expr.target.type.shape
        if not all(
            isinstance(item, int) and not isinstance(item, bool) for item in (start, step, length)
        ):
            return None
        if length <= 0:
            return 0, 0
        return start, start + (length - 1) * step
    if isinstance(expr.target, Binary):
        left = _static_range(expr.args[0], narrow=narrow)
        right = _static_range(expr.args[1], narrow=narrow)
        if left is None or right is None:
            return None
        if expr.target.kind is BinaryKind.ADD:
            return left[0] + right[0], left[1] + right[1]
        if expr.target.kind is BinaryKind.MUL:
            products = tuple(a * b for a in left for b in right)
            return min(products), max(products)
    return None


def _slice_start(
    start: Expr, loops: tuple[GridRegionExpr, ...], *, narrow: bool
) -> tuple[int | None, int, int]:
    base, offset = window_base(start)
    if base is None:
        return None, offset, offset
    for index, loop in enumerate(loops):
        if loop.induction_var is base:
            return index, offset, offset
    if isinstance(start, Call) and isinstance(start.target, Binary):
        if start.target.kind is BinaryKind.ADD:
            for loop_expr, invariant in (
                (start.args[0], start.args[1]),
                (start.args[1], start.args[0]),
            ):
                for index, loop in enumerate(loops):
                    if loop.induction_var is loop_expr:
                        bounds = _static_range(invariant, narrow=narrow)
                        if bounds is not None:
                            return index, bounds[0], bounds[1]
    raise _Unavailable


def _fold_slice(
    access: isl.map,
    call: Call,
    loops: tuple[GridRegionExpr, ...],
    *,
    narrow: bool,
) -> isl.map:
    return _fold_window(access, call.args[1], call.target.strides, loops, narrow=narrow)


def _fold_window(
    access: isl.map,
    starts: Expr,
    strides: tuple[int, ...],
    loops: tuple[GridRegionExpr, ...],
    *,
    narrow: bool,
) -> isl.map:
    """Move local window coordinates to their source-buffer coordinates."""
    rank = access.dim(isl.dim_type.OUT)
    elements = starts.elements if isinstance(starts, Tuple) else (starts,)
    if len(elements) != rank or len(strides) != rank:
        raise _Unavailable
    transformed = access.insert_dims(isl.dim_type.OUT, rank, rank)
    local = isl.local_space.from_space(transformed.get_space())
    for axis, (start, stride) in enumerate(zip(elements, strides)):
        if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
            raise _Unavailable
        loop_axis, lower, upper = _slice_start(start, loops, narrow=narrow)

        def positioned(constraint: isl.constraint, sign: int) -> isl.constraint:
            constraint = constraint.set_coefficient_si(
                isl.dim_type.OUT, rank + axis, sign
            ).set_coefficient_si(isl.dim_type.OUT, axis, -sign * stride)
            if loop_axis is not None:
                constraint = constraint.set_coefficient_si(isl.dim_type.IN, loop_axis, -sign)
            return constraint

        if lower == upper:
            constraint = positioned(isl.constraint.alloc_equality(local), 1)
            transformed = transformed.add_constraint(constraint.set_constant_si(-lower))
        else:
            low = positioned(isl.constraint.alloc_inequality(local), 1)
            high = positioned(isl.constraint.alloc_inequality(local), -1)
            transformed = transformed.add_constraint(low.set_constant_si(-lower))
            transformed = transformed.add_constraint(high.set_constant_si(upper))
    return transformed.project_out(isl.dim_type.OUT, 0, rank)


def _source_buffer(
    expr: Expr,
    access: isl.map,
    loops: tuple[GridRegionExpr, ...],
    *,
    narrow: bool,
) -> tuple[Expr, isl.map]:
    """Fold Slice and Reshape coordinates until *access* reaches an allocation."""
    while isinstance(expr, Call) and isinstance(expr.target, (Slice, Reshape)):
        if isinstance(expr.target, Slice):
            access = _fold_slice(access, expr, loops, narrow=narrow)
        else:
            old_type = _local_type(expr.args[0].type) if narrow else expr.args[0].type
            new_type = _local_type(expr.type) if narrow else expr.type
            if not isinstance(old_type, TensorType) or not isinstance(new_type, TensorType):
                raise _Unavailable
            access = access.apply_range(flat_reshape_map(old_type.shape, new_type.shape))
        expr = expr.args[0]
    return expr, access


def _narrow_reshard_window(call: Call) -> TensorType | None:
    if not (
        isinstance(call.target, Reshard)
        and isinstance(call.args[0], Call)
        and isinstance(call.args[0].target, Slice)
    ):
        return None
    source = _local_type(call.args[0].type)
    target = _local_type(call.type)
    if not isinstance(source, TensorType) or not isinstance(target, TensorType):
        return None
    if source.shape == target.shape or len(source.shape) != len(target.shape):
        return None
    if not all(
        isinstance(source_size, int) and isinstance(target_size, int) and target_size <= source_size
        for source_size, target_size in zip(source.shape, target.shape)
    ):
        return None
    return TensorType(
        shape=target.shape,
        dtype=source.dtype,
        layout=None,
        storage=source.storage,
    )


def _relation(call: Call, ctx: TypeInferContext, *, narrow: bool):
    input_types = tuple(_local_type(arg.type) if narrow else arg.type for arg in call.args)
    if isinstance(call.target, InsertSlice):
        update = input_types[1]
        if not isinstance(update, TensorType):
            raise _Unavailable
        domain, param_map = to_domain(update.shape)
        ident = identity_map(len(update.shape))
        none = isl.map(f"{{ [{', '.join(f'd{i}' for i in range(len(update.shape)))}] -> [] }}")
        return AccessRelationResult(
            domain=domain,
            maps=(ident, ident, none, ident),
            param_map=param_map,
        )
    if isinstance(call.target, CacheUpdate):
        new = input_types[3]
        if not isinstance(new, TensorType):
            raise _Unavailable
        domain, param_map = to_domain(new.shape)
        ident = identity_map(len(new.shape))
        none = isl.map(f"{{ [{', '.join(f'd{i}' for i in range(len(new.shape)))}] -> [] }}")
        return AccessRelationResult(
            domain=domain,
            maps=(ident, none, none, ident, ident),
            param_map=param_map,
        )
    if isinstance(call.target, Reshard) and not narrow:
        source = input_types[0]
        if not isinstance(source, TensorType):
            raise _Unavailable
        domain, param_map = to_domain(source.shape)
        ident = identity_map(len(source.shape))
        return AccessRelationResult(domain=domain, maps=(ident, ident), param_map=param_map)
    if narrow and (window := _narrow_reshard_window(call)) is not None:
        domain, param_map = to_domain(window.shape)
        ident = identity_map(len(window.shape))
        return AccessRelationResult(domain=domain, maps=(ident, ident), param_map=param_map)
    try:
        return build_relation(call, input_types, ctx)
    except (NotImplementedError, TypeError, ValueError, isl.Error) as error:
        raise _Unavailable from error


def _relation_cases(
    call: Call, ctx: TypeInferContext, *, narrow: bool
) -> tuple[_RelationCase, ...]:
    """Build one normal domain, or RoPE's separate Q and K domains."""
    local_types = tuple(_local_type(arg.type) if narrow else arg.type for arg in call.args)
    if isinstance(call.target, RoPE):
        cases: list[_RelationCase] = []
        for value_index in (0, 1):
            branch_types = (
                local_types[value_index],
                local_types[value_index],
                local_types[2],
                local_types[3],
                local_types[4],
            )
            try:
                result = build_relation(call, branch_types, ctx)
            except (NotImplementedError, TypeError, ValueError, isl.Error) as error:
                raise _Unavailable from error
            if result is None:
                raise _Unavailable
            cases.append(
                _RelationCase(
                    result=result,
                    operand_maps=((value_index, 0), (2, 2), (3, 3), (4, 4)),
                    local_types=local_types,
                )
            )
        return tuple(cases)
    result = _relation(call, ctx, narrow=narrow)
    if result is None or len(result.maps) < len(call.args):
        raise _Unavailable
    if narrow and (window := _narrow_reshard_window(call)) is not None:
        local_types = (window,)
    return (
        _RelationCase(
            result=result,
            operand_maps=tuple((index, index) for index in range(len(call.args))),
            local_types=local_types,
        ),
    )


def _labels(fn: Function) -> dict[int, str]:
    order: list[Expr] = [
        *fn.params,
        *(expr for expr in postorder(fn.body) if isinstance(expr, (Call, Constant))),
    ]
    seen = {id(expr) for expr in order}
    order.extend(expr for expr in postorder(fn.body) if id(expr) not in seen)
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
    repeated: dict[int, set[int]],
    order: dict[int, int],
) -> tuple[GridRegionExpr, ...]:
    return tuple(
        sorted(
            (loop for loop_id, loop in loops.items() if id(call) in repeated[loop_id]),
            key=lambda loop: -order[id(loop)],
        )
    )


def _collect(
    module: Module, fn: Function, *, narrow: bool = True
) -> tuple[
    dict[int, GridRegionExpr],
    dict[int, list[_Access]],
    dict[int, list[Call]],
]:
    """Collect each directly owned Call's accesses and unavailable relation scopes."""
    values = postorder(fn.body)
    loops = {id(expr): expr for expr in values if isinstance(expr, GridRegionExpr)}
    order = {id(expr): index for index, expr in enumerate(values)}
    repeated = {loop_id: loop_repeated_values(loop) for loop_id, loop in loops.items()}
    labels = _labels(fn)
    ctx = TypeInferContext(scope=FunctionScope(module, fn))
    accesses: dict[int, list[_Access]] = {key: [] for key in loops}
    unavailable: dict[int, list[Call]] = {key: [] for key in loops}
    for call in (expr for expr in values if isinstance(expr, Call)):
        path = _enclosing_loops(call, loops, repeated, order)
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
                domain = _loop_domain(case.result.domain, path)
            except _Unavailable:
                call_unavailable = True
                break
            depth = len(path)
            for index, map_index in case.operand_maps:
                operand = call.args[index]
                raw = case.result.maps[map_index]
                local_type = case.local_types[index]
                if raw.dim(isl.dim_type.OUT) == 0 or not isinstance(local_type, TensorType):
                    continue
                access = raw.insert_dims(isl.dim_type.IN, 0, depth)
                try:
                    if isinstance(call.target, InsertSlice) and index == 0:
                        update = case.local_types[1]
                        if not isinstance(update, TensorType):
                            raise _Unavailable
                        access = _fold_window(
                            access,
                            call.args[2],
                            (1,) * len(update.shape),
                            path,
                            narrow=narrow,
                        )
                    buffer, access = _source_buffer(operand, access, path, narrow=narrow)
                    if not isinstance(buffer.type, TensorType):
                        continue
                    if isinstance(call.target, (InsertSlice, CacheUpdate)) and index == 0:
                        window_type = case.local_types[
                            1 if isinstance(call.target, InsertSlice) else 3
                        ]
                        if not isinstance(window_type, TensorType):
                            raise _Unavailable
                        elements = numel(window_type)
                    else:
                        elements = numel(local_type)
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


def loop_footprints(module: Module, fn: Function) -> dict[int, _LoopReading]:
    """Return each authored loop's known reading or best available lower bound."""
    loops, accesses, unavailable = _collect(module, fn)
    _, device_accesses, device_unavailable = _collect(module, fn, narrow=False)

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
