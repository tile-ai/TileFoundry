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
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.clamp import Clamp
from tilefoundry.ir.hir.math.softplus import Softplus
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.nn.gelu import Gelu
from tilefoundry.ir.hir.nn.layer_norm import LayerNorm
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.nn.relu import ReLU
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.hir.nn.rope import RoPE
from tilefoundry.ir.hir.nn.sigmoid import Sigmoid
from tilefoundry.ir.hir.nn.silu import Silu
from tilefoundry.ir.hir.nn.softmax import SoftMax
from tilefoundry.ir.hir.nn.tanh import Tanh
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.hir.tensor.full_like import FullLike
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.hir.tensor.repeat_interleave import RepeatInterleave
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.zeros import Zeros
from tilefoundry.ir.types import DType, TensorType, Type, numel, tensor_bytes

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


@register_cost_evaluator(MatMul)
def _matmul(call: Call, ctx: CostContext) -> Cost:
    """One multiply and one add per multiply-accumulate, over every batch.

    The batch comes from the output rather than from the left operand. Either side
    may be the one that is broadcast: a block of a weight matrix multiplied by one
    token has its batch on the right, and reading the left gave a batch of one --
    the whole block loop's arithmetic charged as a single tile's. The output's batch
    is what the call produced, and every batch of it was computed.
    """
    lhs, rhs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    if not all(isinstance(type, TensorType) for type in (lhs, rhs, output)):
        raise ValueError("MatMul cost requires tensor inputs and output")
    m, k, n = lhs.shape[-2], lhs.shape[-1], rhs.shape[-1]
    batch = math.prod(output.shape[:-2])
    flops = 2 * batch * m * k * n
    return Cost({lhs.dtype: flops}, _traffic((lhs, rhs), output))


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


@register_cost_evaluator(Binary)
def _binary(call: Call, ctx: CostContext) -> Cost:
    kind = call.target.kind
    dtype = DType.bool if kind in {
        BinaryKind.EQ, BinaryKind.NE, BinaryKind.LT, BinaryKind.LE,
        BinaryKind.GT, BinaryKind.GE, BinaryKind.AND, BinaryKind.OR,
    } else None
    return _elementwise(call, ctx, dtype=dtype)


@register_cost_evaluator(Unary)
def _unary(call: Call, ctx: CostContext) -> Cost:
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
    """A slice selects elements and computes none.

    Its traffic is what it actually moves -- the selected region, not the tensor
    it came from. A model that splits a wide projection into unequal parts does
    that once per part, and charging each one for the whole projection would
    report a kernel reading its input as many times as it has pieces.
    """
    kept = tensor_bytes(_output_type(call, ctx))
    return Cost({}, (TrafficBytes(read=kept), TrafficBytes(write=kept)))


@register_cost_evaluator(SoftMax)
def _softmax(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


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


@register_cost_evaluator(Gather)
def _gather(call: Call, ctx: CostContext) -> Cost:
    indices = _input_types(call, ctx)[1]
    rows = tensor_bytes(_output_type(call, ctx))
    return Cost(
        {},
        (
            TrafficBytes(read=rows),
            TrafficBytes(read=tensor_bytes(indices)),
            TrafficBytes(write=rows),
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


@register_cost_evaluator(FullLike)
def _full_like(call: Call, ctx: CostContext) -> Cost:
    return Cost({}, _traffic(_input_types(call, ctx), _output_type(call, ctx)))


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
    return Cost({}, (
        TrafficBytes(read=tensor_bytes(source)),
        TrafficBytes(write=tensor_bytes(destination)),
    ))


__all__ = ["tensor_bytes"]
