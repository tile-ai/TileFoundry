"""How much work each operation asks for, per instance.

Flops and bytes follow from an operation's own semantics and its operand types,
not from the machine that will run it: the same MatMul asks for the same
multiply-accumulates on every backend. These evaluators are therefore owned here
rather than by any target package, and both the analysis layer and the scheduling
algorithms read the one registry they fill.
"""

from __future__ import annotations

import math

from tilefoundry.ir.core import Call
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.clamp import Clamp
from tilefoundry.ir.hir.math.softplus import Softplus
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.nn.conv2d import Conv2D
from tilefoundry.ir.hir.nn.gelu import Gelu
from tilefoundry.ir.hir.nn.layer_norm import LayerNorm
from tilefoundry.ir.hir.nn.matmul import MatMul, matmul_axes
from tilefoundry.ir.hir.nn.relu import ReLU
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.hir.nn.rope import RoPE
from tilefoundry.ir.hir.nn.sigmoid import Sigmoid
from tilefoundry.ir.hir.nn.silu import Silu
from tilefoundry.ir.hir.nn.softmax import SoftMax
from tilefoundry.ir.hir.nn.tanh import Tanh
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.hir.tensor.full_like import FullLike
from tilefoundry.ir.hir.tensor.index_add import IndexAdd
from tilefoundry.ir.hir.tensor.index_copy import IndexCopy
from tilefoundry.ir.hir.tensor.index_select import IndexSelect
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.ir.hir.tensor.rank import Rank
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.hir.tensor.repeat_interleave import RepeatInterleave
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.shape_of import ShapeOf
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.split import Split
from tilefoundry.ir.hir.tensor.stack import Stack
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.where import Where
from tilefoundry.ir.hir.tensor.zeros import Zeros
from tilefoundry.ir.types import DType, IntegerDType, TensorType, Type, numel, tensor_bytes
from tilefoundry.ir.types.shard import ShardLayout
from tilefoundry.ir.types.shard.shard_layout import layout_axis_to_tensor_axis
from tilefoundry.visitor_registry.access_relation import logical_axes_of

from .contexts import Cost, CostContext, TrafficBytes
from .registries import register_cost_evaluator


def _input_types(call: Call, ctx: CostContext) -> tuple[Type, ...]:
    return tuple(ctx.local_type_of(arg) for arg in call.args)


def _output_type(call: Call, ctx: CostContext) -> Type:
    return ctx.local_output_type(call)


def _traffic(inputs: tuple[Type, ...], output: Type) -> tuple[TrafficBytes, ...]:
    """One entry per operand: every input read whole, the result written whole.

    The default an Op gets by saying nothing about which part it touches.
    """
    return (
        *(TrafficBytes(read=tensor_bytes(type)) for type in inputs),
        TrafficBytes(write=tensor_bytes(output)),
    )


def _row(table: Type) -> int:
    """One position's worth of a cache whose leading axis is the position."""
    return tensor_bytes(table) // table.shape[0]


def _idle(call: Call) -> tuple[TrafficBytes, ...]:
    """No operand moves, but every operand still has a slot."""
    return tuple(TrafficBytes() for _ in range(len(call.args) + 1))


def _elementwise(call: Call, ctx: CostContext, *, dtype: DType | None = None) -> Cost:
    inputs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    result_dtype = dtype
    if result_dtype is None:
        if isinstance(output, TensorType):
            result_dtype = output.dtype
        else:
            result_dtype = next(
                type.dtype for type in inputs if isinstance(type, TensorType)
            )
    return Cost({result_dtype: numel(output)}, _traffic(inputs, output))


def _serviced(call: Call, ctx: CostContext, kind: str) -> Cost:
    """One result of *kind* per element, and no floating-point work at all."""
    inputs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    return Cost({}, _traffic(inputs, output), {kind: numel(output)})


@register_cost_evaluator(MatMul)
def _matmul(call: Call, ctx: CostContext) -> Cost:
    """One multiply and one add per multiply-accumulate: 2 * batch * m * k * n."""
    lhs, rhs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    if not all(isinstance(type, TensorType) for type in (lhs, rhs, output)):
        raise ValueError("MatMul cost requires tensor inputs and output")
    logical_lhs = ctx.type_of(call.args[0])
    if not isinstance(logical_lhs, TensorType):
        raise ValueError("MatMul cost requires a tensor lhs")
    _a_m, a_k, _b_n, _b_k = matmul_axes(call.target)
    k_axis = a_k % len(logical_lhs.shape)
    k = math.prod(
        extent
        for extent, logical_axis in zip(
            lhs.shape, logical_axes_of(lhs, logical_lhs)
        )
        if logical_axis == k_axis
    )
    flops = 2 * numel(output) * k
    return Cost({lhs.dtype: flops}, _traffic((lhs, rhs), output))


@register_cost_evaluator(Conv2D)
def _conv2d(call: Call, ctx: CostContext) -> Cost:
    input_, weight, bias = _input_types(call, ctx)
    output = _output_type(call, ctx)
    if not all(
        isinstance(type_, TensorType)
        for type_ in (input_, weight, bias, output)
    ):
        raise ValueError("Conv2D cost requires tensor inputs and output")
    global_weight = ctx.type_of(call.args[1])
    logical_weight_shape = weight.shape
    if isinstance(global_weight.layout, ShardLayout):
        logical_weight_shape = [1] * len(global_weight.shape)
        axes = layout_axis_to_tensor_axis(
            global_weight.layout.layout.shape, global_weight.shape
        )
        for extent, logical_axis in zip(weight.shape, axes):
            logical_weight_shape[logical_axis] *= extent
    flops = (
        2
        * numel(output)
        * logical_weight_shape[1]
        * logical_weight_shape[2]
        * logical_weight_shape[3]
    )
    return Cost(
        {input_.dtype: flops}, _traffic((input_, weight, bias), output)
    )


@register_cost_evaluator(Reduce)
def _reduce(call: Call, ctx: CostContext) -> Cost:
    (source,) = _input_types(call, ctx)
    output = _output_type(call, ctx)
    if not isinstance(source, TensorType):
        raise ValueError("Reduce cost requires a tensor input")
    return Cost({source.dtype: numel(source)}, _traffic((source,), output))


@register_cost_evaluator(RMSNorm)
def _rms_norm(call: Call, ctx: CostContext) -> Cost:
    inputs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    source = inputs[0]
    if not isinstance(source, TensorType):
        raise ValueError("RMSNorm cost requires a tensor input")
    return Cost({DType.f32: 8 * numel(source)}, _traffic(inputs, output))


_PREDICATES = frozenset(
    {
        BinaryKind.EQ,
        BinaryKind.NE,
        BinaryKind.LT,
        BinaryKind.LE,
        BinaryKind.GT,
        BinaryKind.GE,
        BinaryKind.AND,
        BinaryKind.OR,
    }
)


@register_cost_evaluator(Binary)
def _binary(call: Call, ctx: CostContext) -> Cost:
    """Arithmetic is floating-point work; comparing and combining truths is not.

    A comparison over floats produces booleans, and calling that a bool FLOP
    asks the target for a rate no machine publishes. What it really asks for is
    a predicate, which is a service a machine does state a throughput for.
    ``_PREDICATES`` holds the kinds whose result is one, whatever the operands
    were.
    """
    kind = call.target.kind
    if kind in _PREDICATES:
        return _serviced(call, ctx, "predicate")
    if _integral(call, ctx):
        return _serviced(call, ctx, "integer")
    return _elementwise(call, ctx)


def _integral(call: Call, ctx: CostContext) -> bool:
    """Whether this operation's result is whole numbers rather than reals."""
    output = _output_type(call, ctx)
    return isinstance(output, TensorType) and isinstance(output.dtype, IntegerDType)


_SPECIAL = frozenset(
    {UnaryKind.RSQRT, UnaryKind.EXP, UnaryKind.LOG, UnaryKind.EXP2, UnaryKind.LOG2}
)


@register_cost_evaluator(Unary)
def _unary(call: Call, ctx: CostContext) -> Cost:
    """An exponential is not a multiply, a negated truth is not arithmetic.

    ``_SPECIAL`` holds the kinds the machine answers on its
    special-function unit, at a rate of its own -- a quarter of the scalar one
    here. Counting one of those as a single FLOP would put it on the float pipe
    at four times the throughput the unit has, so it is a service rather than
    arithmetic. What is left for ``flops`` is the arithmetic that really is a
    multiply or an add.
    """
    kind = call.target.kind
    if kind in _SPECIAL:
        return _serviced(call, ctx, "special")
    if kind is UnaryKind.NOT:
        return _serviced(call, ctx, "predicate")
    if _integral(call, ctx):
        return _serviced(call, ctx, "integer")
    return _elementwise(call, ctx)


@register_cost_evaluator(Clamp)
def _clamp(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(Sigmoid)
def _sigmoid(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(Softplus)
def _softplus(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(Silu)
def _silu(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(Tanh)
def _tanh(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(ReLU)
def _relu(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(Gelu)
def _gelu(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(Concat)
def _concat(call: Call, ctx: CostContext) -> Cost:
    """A concatenation places its inputs side by side and computes nothing.

    Every input element is read once and written once, so the traffic is the
    inputs plus the output -- unlike a slice, which reads only what it keeps.
    """
    return Cost({}, _traffic(_input_types(call, ctx), _output_type(call, ctx)))


@register_cost_evaluator(Slice)
def _slice(call: Call, ctx: CostContext) -> Cost:
    """A slice is a view; reading its coordinates still has a cost."""
    _, starts = _input_types(call, ctx)
    return Cost(
        {},
        (
            TrafficBytes(),
            TrafficBytes(read=tensor_bytes(starts)),
            TrafficBytes(),
        ),
    )


@register_cost_evaluator(InsertSlice)
def _insert_slice(call: Call, ctx: CostContext) -> Cost:
    """Read and write the update window without charging the untouched dst."""
    _, update, offsets = _input_types(call, ctx)
    window = tensor_bytes(update)
    return Cost(
        {},
        (
            TrafficBytes(),
            TrafficBytes(read=window),
            TrafficBytes(read=tensor_bytes(offsets)),
            TrafficBytes(write=window),
        ),
    )


@register_cost_evaluator(CacheUpdate)
def _cache_update(call: Call, ctx: CostContext) -> Cost:
    """Charge the statically bounded new window, never the whole cache."""
    _, cur_pos, s, new = _input_types(call, ctx)
    window = tensor_bytes(new)
    return Cost(
        {},
        (
            TrafficBytes(),
            TrafficBytes(read=tensor_bytes(cur_pos)),
            TrafficBytes(read=tensor_bytes(s)),
            TrafficBytes(read=window),
            TrafficBytes(write=window),
        ),
    )


@register_cost_evaluator(SoftMax)
def _softmax(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(Where)
def _where(call: Call, ctx: CostContext) -> Cost:
    """Choosing between two values it already has is a select, not arithmetic."""
    return _serviced(call, ctx, "select")


@register_cost_evaluator(LayerNorm)
def _layer_norm(call: Call, ctx: CostContext) -> Cost:
    source = _input_types(call, ctx)[0]
    output = _output_type(call, ctx)
    if not isinstance(source, TensorType):
        raise ValueError("LayerNorm cost requires a tensor input")
    return Cost({DType.f32: 8 * numel(source)}, _traffic(_input_types(call, ctx), output))


@register_cost_evaluator(RoPE)
def _rope(call: Call, ctx: CostContext) -> Cost:
    """Rotate q and k in place against the caches.

    Each rotated element costs one multiply against the cosine, one against
    the sine of its partner, and the add that combines them. The output is the
    pair, so its element count already covers both q and k.

    The position-indexed caches contribute one row per call.
    """
    query, key, cos_cache, sin_cache, pos_ids = _input_types(call, ctx)
    output = _output_type(call, ctx)
    if not isinstance(query, TensorType):
        raise ValueError("RoPE cost requires a tensor query input")
    return Cost(
        {query.dtype: 3 * numel(output)},
        (
            TrafficBytes(read=tensor_bytes(query)),
            TrafficBytes(read=tensor_bytes(key)),
            TrafficBytes(read=_row(cos_cache)),
            TrafficBytes(read=_row(sin_cache)),
            TrafficBytes(read=tensor_bytes(pos_ids)),
            TrafficBytes(write=tensor_bytes(output)),
        ),
    )


@register_cost_evaluator(TopK)
def _topk(call: Call, ctx: CostContext) -> Cost:
    inputs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    source = inputs[0]
    dtype = source.dtype if isinstance(source, TensorType) else DType.f32
    return Cost({dtype: numel(source)}, _traffic(inputs, output))


@register_cost_evaluator(IndexSelect)
def _index_select(call: Call, ctx: CostContext) -> Cost:
    index = _input_types(call, ctx)[1]
    rows = tensor_bytes(_output_type(call, ctx))
    return Cost(
        {},
        (
            TrafficBytes(read=rows),
            TrafficBytes(read=tensor_bytes(index)),
            TrafficBytes(write=rows),
        ),
    )


@register_cost_evaluator(IndexAdd)
def _index_add(call: Call, ctx: CostContext) -> Cost:
    _, index, src = _input_types(call, ctx)
    touched = tensor_bytes(src)
    return Cost(
        {src.dtype: numel(src)},
        (
            TrafficBytes(read=touched),
            TrafficBytes(read=tensor_bytes(index)),
            TrafficBytes(read=touched),
            TrafficBytes(write=touched),
        ),
    )


@register_cost_evaluator(IndexCopy)
def _index_copy(call: Call, ctx: CostContext) -> Cost:
    _, index, src = _input_types(call, ctx)
    touched = tensor_bytes(src)
    return Cost(
        {},
        (
            TrafficBytes(),
            TrafficBytes(read=tensor_bytes(index)),
            TrafficBytes(read=touched),
            TrafficBytes(write=touched),
        ),
    )


@register_cost_evaluator(ArgMax)
def _argmax(call: Call, ctx: CostContext) -> Cost:
    inputs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    return Cost({}, _traffic(inputs, output))


@register_cost_evaluator(Cast)
def _cast(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(Quant)
def _quant(call: Call, ctx: CostContext) -> Cost:
    inputs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    source = inputs[0]
    if not isinstance(source, TensorType):
        raise ValueError("Quant cost requires a tensor input")
    return Cost({DType.f32: 4 * numel(source)}, _traffic(inputs, output))


@register_cost_evaluator(TupleGetItem)
def _tuple_get_item(call: Call, ctx: CostContext) -> Cost:
    return Cost({}, _idle(call))


@register_cost_evaluator(Rank)
@register_cost_evaluator(ShapeOf)
@register_cost_evaluator(Local)
def _metadata_view(call: Call, ctx: CostContext) -> Cost:
    return Cost({}, _idle(call))


@register_cost_evaluator(Split)
@register_cost_evaluator(Stack)
def _structure(call: Call, ctx: CostContext) -> Cost:
    return Cost({}, _traffic(_input_types(call, ctx), _output_type(call, ctx)))


@register_cost_evaluator(FullLike)
def _full_like(call: Call, ctx: CostContext) -> Cost:
    return Cost({}, _traffic(_input_types(call, ctx), _output_type(call, ctx)))


@register_cost_evaluator(Arange)
def _arange(call: Call, ctx: CostContext) -> Cost:
    """Coordinates are synthesized metadata until a consumer materializes them."""
    return Cost({}, _idle(call))


@register_cost_evaluator(Zeros)
def _zeros(call: Call, ctx: CostContext) -> Cost:
    """Materialise a tensor of zeros: no arithmetic, one full write."""
    output = _output_type(call, ctx)
    return Cost({}, (TrafficBytes(write=tensor_bytes(output)),))


@register_cost_evaluator(RepeatInterleave)
def _repeat_interleave(call: Call, ctx: CostContext) -> Cost:
    inputs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    return Cost({}, _traffic(inputs, output))


@register_cost_evaluator(Reshape)
def _reshape(call: Call, ctx: CostContext) -> Cost:
    return Cost({}, _idle(call))


@register_cost_evaluator(Transpose)
def _transpose(call: Call, ctx: CostContext) -> Cost:
    moved = tensor_bytes(_output_type(call, ctx))
    return Cost({}, (TrafficBytes(read=moved), TrafficBytes(write=moved)))


@register_cost_evaluator(Reshard)
def _reshard(call: Call, ctx: CostContext) -> Cost:
    source = _input_types(call, ctx)[0]
    destination = _output_type(call, ctx)
    if source.storage == destination.storage:
        return Cost({}, _idle(call))
    return Cost({}, (
        TrafficBytes(read=tensor_bytes(source)),
        TrafficBytes(write=tensor_bytes(destination)),
    ))


__all__ = ["tensor_bytes"]
