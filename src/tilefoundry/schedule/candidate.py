from __future__ import annotations

import math
from dataclasses import dataclass

from tilefoundry.ir.core import Tuple
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.shard.shard_layout import Partial, ShardLayout, Split

from .graph import GraphOp, ProgramScheduleGraph


class DistributionError(ValueError):
    """Base error for illegal or unsupported distribution candidates."""


class UnsupportedDistributionError(DistributionError):
    pass


@dataclass(frozen=True, slots=True)
class Submesh:
    offsets: tuple[int, ...]
    extents: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.extents):
            raise ValueError("Submesh offsets and extents must have the same rank")
        if any(offset < 0 or extent <= 0 for offset, extent in zip(self.offsets, self.extents)):
            raise ValueError("Submesh offsets must be non-negative and extents positive")

    @property
    def rank(self) -> int:
        return len(self.extents)

    @property
    def size(self) -> int:
        return math.prod(self.extents)


@dataclass(frozen=True, slots=True)
class LayoutState:
    rank: int
    split_axis: int | None = None

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("layout rank must be non-negative")
        if self.split_axis is not None and not 0 <= self.split_axis < self.rank:
            raise ValueError("layout split axis is outside the tensor rank")

    @property
    def is_broadcast(self) -> bool:
        return self.split_axis is None


@dataclass(frozen=True, slots=True)
class PartialState:
    reduction: str = "sum"
    topology: str = "cta"


@dataclass(frozen=True, slots=True)
class DistributionState:
    layout: LayoutState
    cta_count: int = 1
    partial: PartialState | None = None

    def __post_init__(self) -> None:
        if self.cta_count <= 0:
            raise ValueError("DistributionState cta_count must be positive")
        if self.layout.is_broadcast and self.cta_count != 1 and self.partial is None:
            raise ValueError("a broadcast DistributionState must use one CTA")


@dataclass(frozen=True, slots=True)
class WorkEstimate:
    flops: float
    traffic_bytes: float
    dtype: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.flops) or not math.isfinite(self.traffic_bytes):
            raise ValueError("candidate work estimates must be finite")
        if self.flops < 0 or self.traffic_bytes < 0:
            raise ValueError("candidate work estimates must be non-negative")


@dataclass(frozen=True, slots=True)
class OpCandidate:
    id: int
    op_id: int
    input_states: tuple[DistributionState, ...]
    output_states: tuple[DistributionState, ...]
    cta_count: int
    estimated_work: WorkEstimate
    implementation_key: str


@dataclass(frozen=True, slots=True)
class CandidateTable:
    options: tuple[tuple[int, tuple[OpCandidate, ...]], ...]

    def for_op(self, op_id: int) -> tuple[OpCandidate, ...]:
        for key, candidates in self.options:
            if key == op_id:
                return candidates
        raise KeyError(op_id)

    def all_candidates(self) -> tuple[OpCandidate, ...]:
        return tuple(candidate for _, options in self.options for candidate in options)


_ELEMENTWISE_OPS = {
    "Binary",
    "Unary",
    "Cast",
    "RMSNorm",
    "ReLU",
    "Sigmoid",
    "Softplus",
    "Tanh",
    "Clamp",
    "FullLike",
    "Quant",
    "Log",
    "Exp",
    "Rsqrt",
    "Maximum",
    "Minimum",
}


def _numel(ty: object) -> int:
    if isinstance(ty, TensorType):
        if all(isinstance(dim, int) and dim >= 0 for dim in ty.shape):
            return math.prod(ty.shape)
        return 1
    if isinstance(ty, TupleType):
        return sum(_numel(field) for field in ty.fields)
    return 1


def _dtype_name(ty: object) -> str:
    return getattr(getattr(ty, "dtype", None), "value", "f32")


def _static_dim(ty: object, axis: int) -> int | None:
    if not isinstance(ty, TensorType) or axis < 0 or axis >= len(ty.shape):
        return None
    dim = ty.shape[axis]
    return dim if isinstance(dim, int) else None


def _base_state(expr) -> DistributionState:
    ty = getattr(expr, "type", None)
    if not isinstance(ty, TensorType):
        return DistributionState(LayoutState(0), 1)
    if isinstance(ty.layout, ShardLayout):
        split_axis = None
        partial = None
        for attr in ty.layout.attrs:
            if isinstance(attr, Split):
                split_axis = attr.axis
            elif isinstance(attr, Partial):
                partial = PartialState(attr.reduction or "sum")
        cta_count = 1
        try:
            cta_count = math.prod(int(size) for size in ty.layout.mesh.shape)
        except (AttributeError, TypeError, ValueError):
            cta_count = 1
        if split_axis is None and partial is None:
            cta_count = 1
        return DistributionState(LayoutState(len(ty.shape), split_axis), cta_count, partial)
    return DistributionState(LayoutState(len(ty.shape)), 1)


def _split_states(expr, max_ctas: int) -> tuple[DistributionState, ...]:
    if isinstance(expr, Tuple):
        if expr.elements:
            return _split_states(expr.elements[0], max_ctas)
        return (DistributionState(LayoutState(0), 1),)
    base = _base_state(expr)
    ty = getattr(expr, "type", None)
    if not isinstance(ty, TensorType):
        return (base,)
    states = [base]
    for axis in range(len(ty.shape)):
        dim = _static_dim(ty, axis)
        if dim is None or dim <= 1:
            continue
        for count in (2, 4, 8, 16, 32, 64, 128):
            if count > max_ctas or count > dim or dim % count:
                continue
            states.append(DistributionState(LayoutState(len(ty.shape), axis), count))
    return tuple(states)


def _work(op: GraphOp) -> WorkEstimate:
    output_type = op.ir_expr.type
    output_elements = _numel(output_type)
    dtype = _dtype_name(output_type)
    target_name = type(op.target).__name__
    if target_name == "MatMul" and len(op.ir_expr.args) >= 2:
        lhs = getattr(op.ir_expr.args[0], "type", None)
        rhs = getattr(op.ir_expr.args[1], "type", None)
        if isinstance(lhs, TensorType) and isinstance(rhs, TensorType) and len(lhs.shape) >= 2:
            m, k, n = lhs.shape[-2], lhs.shape[-1], rhs.shape[-1]
            if all(isinstance(dim, int) for dim in (m, k, n)):
                batch = math.prod(lhs.shape[:-2]) if all(isinstance(dim, int) for dim in lhs.shape[:-2]) else 1
                return WorkEstimate(2.0 * batch * m * k * n, sum(_numel(arg.type) for arg in op.ir_expr.args) * 2.0, dtype)
    if target_name == "Reduce":
        return WorkEstimate(2.0 * output_elements, output_elements * 4.0, dtype)
    return WorkEstimate(float(output_elements), float(output_elements * 2), dtype)


def _state_for_output(op: GraphOp, state: DistributionState) -> DistributionState:
    output_type = op.ir_expr.type
    if isinstance(output_type, TensorType) and state.layout.rank != len(output_type.shape):
        return DistributionState(
            LayoutState(len(output_type.shape)),
            state.cta_count if state.partial is not None else 1,
            state.partial,
        )
    return state


def _input_states_for(op: GraphOp, state: DistributionState) -> tuple[DistributionState, ...]:
    target_name = type(op.target).__name__
    if target_name == "MatMul" and len(op.ir_expr.args) >= 2:
        lhs_ty = getattr(op.ir_expr.args[0], "type", None)
        rhs_ty = getattr(op.ir_expr.args[1], "type", None)
        lhs_rank = len(getattr(lhs_ty, "shape", ()))
        rhs_rank = len(getattr(rhs_ty, "shape", ()))
        output_rank = len(getattr(op.ir_expr.type, "shape", ()))
        if state.partial is not None:
            return (
                DistributionState(LayoutState(lhs_rank, lhs_rank - 1), state.cta_count),
                DistributionState(LayoutState(rhs_rank, rhs_rank - 2), state.cta_count),
            )
        if state.layout.split_axis == output_rank - 2:
            return (
                DistributionState(LayoutState(lhs_rank, lhs_rank - 2), state.cta_count),
                DistributionState(LayoutState(rhs_rank), 1),
            )
        if state.layout.split_axis == output_rank - 1:
            return (
                DistributionState(LayoutState(lhs_rank), 1),
                DistributionState(LayoutState(rhs_rank, rhs_rank - 1), state.cta_count),
            )
    if target_name == "Transpose" and op.ir_expr.args:
        permutation = tuple(getattr(op.target, "perm", ()))
        if state.layout.split_axis is not None and permutation:
            input_axis = permutation[state.layout.split_axis]
            input_rank = len(getattr(op.ir_expr.args[0].type, "shape", ()))
            return (DistributionState(LayoutState(input_rank, input_axis), state.cta_count),)
    return tuple(state for _ in op.inputs)


def _candidate_states(op: GraphOp, max_ctas: int) -> tuple[DistributionState, ...]:
    target_name = type(op.target).__name__
    if isinstance(op.target, Function):
        return (_base_state(op.ir_expr),)
    if target_name == "MatMul" and len(op.ir_expr.args) >= 2:
        lhs = op.ir_expr.args[0]
        rhs = op.ir_expr.args[1]
        lhs_ty = getattr(lhs, "type", None)
        rhs_ty = getattr(rhs, "type", None)
        out_ty = getattr(op.ir_expr, "type", None)
        if not all(isinstance(ty, TensorType) and len(ty.shape) >= 2 for ty in (lhs_ty, rhs_ty, out_ty)):
            raise UnsupportedDistributionError("MatMul distribution requires rank-2 tensor operands")
        states = [DistributionState(LayoutState(len(out_ty.shape)), 1)]
        m, k, n = lhs_ty.shape[-2], lhs_ty.shape[-1], rhs_ty.shape[-1]
        for count in (2, 4, 8, 16, 32, 64, 128):
            if count > max_ctas:
                continue
            if isinstance(m, int) and m % count == 0:
                states.append(DistributionState(LayoutState(len(out_ty.shape), len(out_ty.shape) - 2), count))
            if isinstance(n, int) and n % count == 0:
                states.append(DistributionState(LayoutState(len(out_ty.shape), len(out_ty.shape) - 1), count))
            if isinstance(k, int) and k % count == 0:
                states.append(DistributionState(LayoutState(len(out_ty.shape), len(out_ty.shape) - 1), count, PartialState()))
        return tuple(dict.fromkeys(states))
    if target_name == "Reduce":
        input_expr = op.ir_expr.args[0]
        input_states = _split_states(input_expr, max_ctas)
        axes = tuple(getattr(op.target, "axes", ()))
        rank = len(getattr(input_expr.type, "shape", ()))
        normalized = {axis + rank if axis < 0 else axis for axis in axes}
        output_states = []
        for state in input_states:
            if state.layout.split_axis in normalized:
                output_states.append(
                    DistributionState(LayoutState(len(getattr(op.ir_expr.type, "shape", ())), None), state.cta_count, PartialState())
                )
            else:
                output_states.append(_state_for_output(op, state))
        return tuple(dict.fromkeys(output_states))
    if target_name == "TopK":
        input_states = _split_states(op.ir_expr.args[0], max_ctas)
        axis = getattr(op.target, "axis", -1)
        rank = len(getattr(op.ir_expr.args[0].type, "shape", ()))
        axis = axis + rank if axis < 0 else axis
        return tuple(state for state in input_states if state.layout.split_axis != axis)
    if target_name == "TupleGetItem":
        source = op.ir_expr.args[0]
        index = getattr(op.target, "index", 0)
        if isinstance(source, Tuple) and 0 <= index < len(source.elements):
            return tuple(
                _state_for_output(op, state)
                for state in _split_states(source.elements[index], max_ctas)
            )
    if target_name == "Gather":
        input_states = _split_states(op.ir_expr.args[0], max_ctas)
        axis = getattr(op.target, "axis", 0)
        rank = len(getattr(op.ir_expr.args[0].type, "shape", ()))
        axis = axis + rank if axis < 0 else axis
        return tuple(
            DistributionState(_state_for_output(op, state).layout, state.cta_count, PartialState())
            if state.layout.split_axis == axis
            else _state_for_output(op, state)
            for state in input_states
        )
    if target_name not in _ELEMENTWISE_OPS and target_name not in {"Reshape", "Transpose", "TupleGetItem"}:
        raise UnsupportedDistributionError(
            f"no common distribution rule for HIR op {target_name!r}"
        )
    source = op.ir_expr.args[0] if op.ir_expr.args else op.ir_expr
    return tuple(_state_for_output(op, state) for state in _split_states(source, max_ctas))


def generate_distribution_candidates(
    graph: ProgramScheduleGraph, *, max_ctas: int = 132
) -> CandidateTable:
    """Generate finite common candidates for every logical graph operation."""
    options: list[tuple[int, tuple[OpCandidate, ...]]] = []
    next_id = 0
    for op in graph.ops:
        states = _candidate_states(op, max_ctas)
        if not states:
            raise DistributionError(f"no legal distribution candidates for op {op.id}")
        work = _work(op)
        candidates = []
        for state in states:
            input_states = _input_states_for(op, state)
            candidates.append(
                OpCandidate(
                    id=next_id,
                    op_id=op.id,
                    input_states=input_states,
                    output_states=(state,),
                    cta_count=state.cta_count,
                    estimated_work=work,
                    implementation_key=type(op.target).__name__,
                )
            )
            next_id += 1
        options.append((op.id, tuple(candidates)))
    return CandidateTable(tuple(options))


__all__ = [
    "CandidateTable",
    "DistributionError",
    "DistributionState",
    "LayoutState",
    "OpCandidate",
    "PartialState",
    "Submesh",
    "UnsupportedDistributionError",
    "WorkEstimate",
    "generate_distribution_candidates",
]
