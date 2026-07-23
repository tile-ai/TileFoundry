"""Private CUDA planning cost evaluators."""

from __future__ import annotations

import math

from tilefoundry.ir.core import Call
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.clamp import Clamp
from tilefoundry.ir.hir.math.softplus import Softplus
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.nn.layer_norm import LayerNorm
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.nn.relu import ReLU
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.hir.nn.sigmoid import Sigmoid
from tilefoundry.ir.hir.nn.softmax import SoftMax
from tilefoundry.ir.hir.nn.tanh import Tanh
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.full_like import FullLike
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.hir.tensor.repeat_interleave import RepeatInterleave
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import DType, TensorType, TupleType, Type
from tilefoundry.visitor_registry import register_cost_evaluator
from tilefoundry.visitor_registry.contexts import Cost, CostContext


def _numel(type: Type) -> int:
    if isinstance(type, TensorType):
        values = []
        for dim in type.shape:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                raise ValueError(
                    f"cost: tensor extent {dim!r} is not a concrete positive integer"
                )
            values.append(dim)
        return math.prod(values)
    if isinstance(type, TupleType):
        return sum(_numel(field) for field in type.fields)
    return 0


def tensor_bytes(type: Type) -> int:
    if isinstance(type, TensorType):
        return math.ceil(_numel(type) * type.dtype.bit_width / 8)
    if isinstance(type, TupleType):
        return sum(tensor_bytes(field) for field in type.fields)
    return 0


def _input_types(call: Call, ctx: CostContext) -> tuple[Type, ...]:
    return tuple(ctx.local_type_of(arg) for arg in call.args)


def _output_type(call: Call, ctx: CostContext) -> Type:
    return ctx.local_output_type(call)


def _traffic(inputs: tuple[Type, ...], output: Type) -> int:
    return sum(tensor_bytes(type) for type in inputs) + tensor_bytes(output)


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
    return Cost({result_dtype: _numel(output)}, _traffic(inputs, output))


@register_cost_evaluator(MatMul)
def _matmul(call: Call, ctx: CostContext) -> Cost:
    lhs, rhs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    if not all(isinstance(type, TensorType) for type in (lhs, rhs, output)):
        raise ValueError("MatMul cost requires tensor inputs and output")
    m, k, n = lhs.shape[-2], lhs.shape[-1], rhs.shape[-1]
    batch = math.prod(lhs.shape[:-2])
    flops = 2 * batch * m * k * n
    return Cost({lhs.dtype: flops}, _traffic((lhs, rhs), output))


@register_cost_evaluator(Reduce)
def _reduce(call: Call, ctx: CostContext) -> Cost:
    (source,) = _input_types(call, ctx)
    output = _output_type(call, ctx)
    if not isinstance(source, TensorType):
        raise ValueError("Reduce cost requires a tensor input")
    return Cost({source.dtype: _numel(source)}, _traffic((source,), output))


@register_cost_evaluator(RMSNorm)
def _rms_norm(call: Call, ctx: CostContext) -> Cost:
    source = _input_types(call, ctx)[0]
    output = _output_type(call, ctx)
    if not isinstance(source, TensorType):
        raise ValueError("RMSNorm cost requires a tensor input")
    return Cost({DType.f32: 8 * _numel(source)}, _traffic((source,), output))


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


@register_cost_evaluator(Tanh)
def _tanh(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(ReLU)
def _relu(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(SoftMax)
def _softmax(call: Call, ctx: CostContext) -> Cost:
    return _elementwise(call, ctx)


@register_cost_evaluator(LayerNorm)
def _layer_norm(call: Call, ctx: CostContext) -> Cost:
    source = _input_types(call, ctx)[0]
    output = _output_type(call, ctx)
    if not isinstance(source, TensorType):
        raise ValueError("LayerNorm cost requires a tensor input")
    return Cost({DType.f32: 8 * _numel(source)}, _traffic(_input_types(call, ctx), output))


@register_cost_evaluator(TopK)
def _topk(call: Call, ctx: CostContext) -> Cost:
    inputs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    source = inputs[0]
    dtype = source.dtype if isinstance(source, TensorType) else DType.f32
    return Cost({dtype: _numel(source)}, _traffic(inputs, output))


@register_cost_evaluator(Gather)
def _gather(call: Call, ctx: CostContext) -> Cost:
    inputs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    return Cost({}, _traffic(inputs, output))


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
    return Cost({DType.f32: 4 * _numel(source)}, _traffic(inputs, output))


@register_cost_evaluator(TupleGetItem)
def _tuple_get_item(call: Call, ctx: CostContext) -> Cost:
    return Cost({}, 0)


@register_cost_evaluator(FullLike)
def _full_like(call: Call, ctx: CostContext) -> Cost:
    output = _output_type(call, ctx)
    return Cost({}, tensor_bytes(output))


@register_cost_evaluator(RepeatInterleave)
def _repeat_interleave(call: Call, ctx: CostContext) -> Cost:
    inputs = _input_types(call, ctx)
    output = _output_type(call, ctx)
    return Cost({}, _traffic(inputs, output))


@register_cost_evaluator(Reshape)
def _reshape(call: Call, ctx: CostContext) -> Cost:
    return Cost({}, 0)


@register_cost_evaluator(Transpose)
def _transpose(call: Call, ctx: CostContext) -> Cost:
    return Cost({}, 0)


@register_cost_evaluator(Reshard)
def _reshard(call: Call, ctx: CostContext) -> Cost:
    source = _input_types(call, ctx)[0]
    destination = _output_type(call, ctx)
    return Cost({}, tensor_bytes(source) + tensor_bytes(destination))


__all__ = ["tensor_bytes"]
