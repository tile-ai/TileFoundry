"""Quant's exact per-token-group FP8 value, shape, and ownership contract."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    raw_shard_tensor_type,
    run_typeinfer_case,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.ir.types import (
    DType,
    TupleType,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.dim import DimFloorDiv, DimVar
from tilefoundry.ir.types.shard import Layout, ShardLayout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    Split,
    split_target_axes,
)
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_BF = DType.bf16
_FP8 = DType.fp8e4m3

CASES = [
    TypeInferCase(
        "rank2_per_token_group_128",
        Quant(),
        (make_tensor_type((1, 2048), _BF),),
        TupleType(
            fields=(
                make_tensor_type((1, 2048), _FP8),
                make_tensor_type((1, 16), DType.f32),
            )
        ),
    ),
    TypeInferCase(
        "rank_zero_rejected",
        Quant(group=1),
        (make_tensor_type((), _BF),),
        ExpectedError(match="at least rank-1"),
    ),
    TypeInferCase(
        "indivisible_last_dim",
        Quant(),
        (make_tensor_type((1, 100), _BF),),
        ExpectedError(match="not divisible by group"),
    ),
    TypeInferCase(
        "unknown_scheme",
        Quant(scheme="block"),
        (make_tensor_type((1, 128), _BF),),
        ExpectedError(match="scheme must be 'per_token_group'.*block"),
    ),
    TypeInferCase(
        "zero_group",
        Quant(group=0),
        (make_tensor_type((1, 128), _BF),),
        ExpectedError(match="positive non-boolean integer.*0"),
    ),
    TypeInferCase(
        "negative_group",
        Quant(group=-4),
        (make_tensor_type((1, 128), _BF),),
        ExpectedError(match="positive non-boolean integer.*-4"),
    ),
    TypeInferCase(
        "boolean_group",
        Quant(group=True),
        (make_tensor_type((1, 128), _BF),),
        ExpectedError(match="positive non-boolean integer.*True"),
    ),
    TypeInferCase(
        "unsupported_target_dtype",
        Quant(target_dtype=DType.f8e8m0),
        (make_tensor_type((1, 128), _BF),),
        ExpectedError(match="target_dtype must be fp8e4m3.*f8e8m0"),
    ),
    TypeInferCase(
        "partial_input_rejected",
        Quant(),
        (
            make_shard_tensor_type(
                (1, 2048), mesh=make_mesh((4,)), attrs=(Partial("max"),)
            ),
        ),
        ExpectedError(match="x carries Partial"),
    ),
    TypeInferCase(
        "last_split_through_group_rejected",
        Quant(group=128),
        (
            make_shard_tensor_type(
                (2, 256), mesh=make_mesh((4,)), attrs=(Split(1),)
            ),
        ),
        ExpectedError(match=r"last axis 1 Split cuts through group=128.*Reshard"),
    ),
    TypeInferCase(
        "strided_last_split_rejected",
        Quant(group=128),
        (
            raw_shard_tensor_type(
                (2, 1024),
                (2, 4, 256),
                (1024, 256, 2),
                (Split(1),),
                make_mesh((4,)),
            ),
        ),
        ExpectedError(match=r"last axis 1 Split cuts through group=128.*Reshard"),
    ),
    TypeInferCase(
        "overlapping_last_split_rejected",
        Quant(group=128),
        (
            raw_shard_tensor_type(
                (2, 1024),
                (2, 4, 256),
                (1024, 128, 1),
                (Split(1),),
                make_mesh((4,)),
            ),
        ),
        ExpectedError(match=r"last axis 1 Split cuts through group=128.*Reshard"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_quant_typeinfer(case) -> None:
    run_typeinfer_case(case)


def _reference(x: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // group, group)
    absmax = grouped.abs().amax(dim=-1)
    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 448.0)
    quantized = (
        (grouped / scale.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .reshape(x.shape)
        .to(torch.float8_e4m3fn)
    )
    return quantized, scale


def _quant_function(shape, group: int) -> tuple[Function, TupleType]:
    x = Var(type=make_tensor_type(shape, DType.f32), name="x")
    call = Call(type=x.type, target=Quant(group=group), args=(x,))
    result_type = TypeInferVisitor(TypeInferContext()).visit(call)
    call = replace(call, type=result_type)
    return (
        Function.build(
            name="quant_eval",
            params=(x,),
            body=call,
            return_type=result_type,
        ),
        result_type,
    )


@pytest.mark.parametrize(
    "source",
    (
        torch.tensor([[224.0, -112.0, 7.0, -1.0, 3.0, -12.0, 96.0, -48.0]]),
        torch.zeros((2, 8), dtype=torch.float32),
    ),
    ids=("signed", "zero_groups"),
)
def test_quant_matches_exact_fp8e4m3_reference(source) -> None:
    function, _ = _quant_function(tuple(source.shape), 4)
    quantized, scale = evaluate(function, source, device="cpu")
    expected_q, expected_scale = _reference(source, 4)

    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    torch.testing.assert_close(quantized.float(), expected_q.float(), atol=0, rtol=0)
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
    assert not torch.isnan(quantized.float()).any()
    assert not torch.isnan(scale).any()
    if not torch.count_nonzero(source):
        torch.testing.assert_close(scale, torch.ones_like(scale), atol=0, rtol=0)


def test_quant_plain_layouts_describe_each_result() -> None:
    source = make_tensor_type(
        (2, 256),
        _BF,
        layout=Layout(shape=(2, 256), strides=(256, 1)),
    )
    quantized, scale = infer_call(Quant(group=128), source).fields

    assert quantized.layout == Layout(shape=(2, 256), strides=(256, 1))
    assert quantized.layout is not source.layout
    assert scale.layout == Layout(shape=(2, 2), strides=(2, 1))


@pytest.mark.parametrize(
    ("shape", "split_axis", "expected_scale_shape"),
    (
        ((4, 256), 0, (4, 2)),
        ((2, 1024), 1, (2, 8)),
    ),
    ids=("outer_axis", "whole_group_last_axis"),
)
def test_quant_propagates_representable_sharding(
    shape, split_axis, expected_scale_shape
) -> None:
    source = make_shard_tensor_type(
        shape, _BF, mesh=make_mesh((4,)), attrs=(Split(split_axis),)
    )
    quantized, scale = infer_call(Quant(group=128), source).fields

    assert quantized.shape == shape
    assert scale.shape == expected_scale_shape
    for field in (quantized, scale):
        assert isinstance(field.layout, ShardLayout)
        assert field.layout is not source.layout
        assert split_target_axes(field.layout, field.shape) == (split_axis,)
        assert math.prod(field.layout.layout.shape) == math.prod(field.shape)


def test_quant_accepts_factorized_contiguous_whole_groups() -> None:
    source = raw_shard_tensor_type(
        (2, 1024),
        (2, 256, 4),
        (1024, 4, 1),
        (Split(1),),
        make_mesh((4,)),
        dtype=_BF,
    )
    quantized, scale = infer_call(Quant(group=128), source).fields

    assert scale.shape == (2, 8)
    assert split_target_axes(quantized.layout, quantized.shape) == (1,)
    assert split_target_axes(scale.layout, scale.shape) == (1,)


def test_quant_drops_fully_broadcast_mesh_ownership() -> None:
    source = make_shard_tensor_type(
        (2, 256), _BF, mesh=make_mesh((4,)), attrs=(Broadcast(),)
    )
    quantized, scale = infer_call(Quant(group=128), source).fields

    assert quantized.layout is None
    assert scale.layout is None


_SYMBOLIC_LAST = DimVar("quant_symbolic_last", 1, 1025)


def test_quant_symbolic_scale_extent_and_runtime_divisibility() -> None:
    function, result_type = _quant_function((2, _SYMBOLIC_LAST), 4)
    scale_last = result_type.fields[1].shape[-1]
    assert isinstance(scale_last, Call)
    assert isinstance(scale_last.target, DimFloorDiv)
    assert scale_last.args[0] is _SYMBOLIC_LAST
    assert scale_last.args[1].value == 4

    source = torch.arange(-12, 12, dtype=torch.float32).reshape(2, 12)
    quantized, scale = evaluate(function, source, device="cpu")
    expected_q, expected_scale = _reference(source, 4)
    torch.testing.assert_close(quantized.float(), expected_q.float(), atol=0, rtol=0)
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)

    with pytest.raises(EvalError, match="runtime last dim 10.*group=4"):
        evaluate(function, torch.zeros((2, 10)), device="cpu")
