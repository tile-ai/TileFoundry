"""MatMul's sharding invariants.

MatMul derives its output shape and ``ShardLayout`` from a forward access
relation (iteration domain ``[batch..., M, N, K]``; lhs ``[batch.., M, K]``, rhs
``[batch.., K, N]``, output ``[batch.., M, N]`` with K reduced). These cases pin
where each mesh axis lands: an rhs N-split becomes an output ``Split`` on N; a
K-split on both operands becomes an output ``Partial``; an lhs M-split passes
through; and a K-split with nothing to contract against, two splits competing for
one mesh axis, or a non-commuting ``Partial`` are errors rather than a pick.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.core.ops.ir.op_case import (
    Cost,
    ExpectedError,
    OpCase,
    check_op_case,
    infer_call,
    op_case_test,
)
from tilefoundry.interfaces import cost
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import nn
from tilefoundry.target import cpu

_MM = nn.MatMul()
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
_CTA = types.Topology("cta", 5)
_CTA_MESH = types.Mesh((_CTA,), types.Layout((5,), (1,)), ("g",))
LAYOUT_CASES = [
    pytest.param(nn.MatMul(), (2, 3), (3, 4), id="mk_kn_default"),
    pytest.param(nn.MatMul(b_layout="NK"), (2, 3), (4, 3), id="mk_nk"),
    pytest.param(nn.MatMul(a_layout="KM"), (3, 2), (3, 4), id="km_kn"),
    pytest.param(nn.MatMul(a_layout="KM", b_layout="NK"), (3, 2), (4, 3), id="km_nk"),
]


@pytest.mark.parametrize(("op", "lhs_shape", "rhs_shape"), LAYOUT_CASES)
def test_matmul_layouts_share_cost(op, lhs_shape, rhs_shape):
    lhs = types.TensorType.from_shape(lhs_shape, expr.DType.f32)
    rhs = types.TensorType.from_shape(rhs_shape, expr.DType.f32)
    assert infer_call(op, lhs, rhs) == types.TensorType.from_shape((2, 4), expr.DType.f32)
    check_op_case(
        OpCase(
            f"{op.a_layout}_{op.b_layout}",
            op,
            (lhs, rhs),
            cost=Cost(
                flops={expr.DType.f32: 2 * 2 * 3 * 4},
                traffic=(
                    cost.TrafficBytes(read=2 * 3 * 4),
                    cost.TrafficBytes(read=4 * 3 * 4),
                    cost.TrafficBytes(write=2 * 4 * 4),
                ),
            ),
        )
    )


def _layout_eval_case(param) -> OpCase:
    op, lhs_shape, rhs_shape = param.values
    lhs = torch.arange(6, dtype=torch.float32).reshape(lhs_shape)
    rhs = torch.arange(12, dtype=torch.float32).reshape(rhs_shape)
    logical_lhs = lhs.transpose(-1, -2) if op.a_layout == "KM" else lhs
    logical_rhs = rhs.transpose(-1, -2) if op.b_layout == "NK" else rhs
    return OpCase(param.id, op, (lhs, rhs), value=logical_lhs @ logical_rhs, atol=0, rtol=0)


_SIXTEENS = (torch.full((2, 3), 16.0), torch.full((3, 4), 16.0))
_BF16_LHS = torch.tensor([[1.0, 2.0 ** (-8)]], dtype=torch.bfloat16)
_BF16_RHS = torch.tensor([[1.0], [1.0]], dtype=torch.bfloat16)


def _sharded(shape, attrs):
    return utils.make_shard_tensor_type(shape, mesh=_M, attrs=attrs, dtype=expr.DType.bf16)


@pytest.mark.parametrize(
    ("attrs", "match"),
    (
        ({"a_layout": "bad"}, "layout must be"),
        ({"b_layout": "bad"}, "layout must be"),
        ({"out_dtype": "f99"}, "DType: unknown value 'f99'"),
    ),
    ids=("invalid_a_layout", "invalid_b_layout", "unknown_out_dtype"),
)
def test_matmul_rejects_invalid_attributes(attrs, match):
    """A bad layout fails in typeinfer; an unknown dtype name fails at construction."""
    with pytest.raises(ValueError, match=match):
        infer_call(
            nn.MatMul(**attrs),
            types.TensorType.from_shape((2, 3), expr.DType.f32),
            types.TensorType.from_shape((4, 3), expr.DType.f32),
        )


def _make_lhs_splits_k_rhs_unsplit_is_invalid():
    lhs = _sharded((16, 8), (types.Split(axis=1),))
    rhs = types.TensorType.from_shape((8, 32), expr.DType.bf16)
    bad = OpCase(
        "lhs_k_split_rhs_unsplit", _MM, (lhs, rhs), error=ExpectedError(match="contraction")
    )
    return bad


def _make_incompatible_shard_errors():
    lhs = _sharded((16, 8), (types.Split(axis=0),))
    rhs = _sharded((8, 32), (types.Split(axis=1),))
    bad = OpCase(
        "incompatible_shard",
        _MM,
        (lhs, rhs),
        error=ExpectedError(match="incompatible|more than one"),
    )
    return bad


CARRIES = [
    pytest.param(
        types.TensorType.from_shape((16, 8), expr.DType.bf16),
        _sharded((8, 32), (types.Split(axis=1),)),
        (16, 32),
        (types.Split(axis=1),),
        id="rhs_n_split_becomes_output_split",
    ),
    pytest.param(
        _sharded((16, 8), (types.Split(axis=0),)),
        types.TensorType.from_shape((8, 32), expr.DType.bf16),
        (16, 32),
        (types.Split(axis=0),),
        id="lhs_m_split_becomes_output_split",
    ),
    pytest.param(
        _sharded((16, 8), (types.Split(axis=1),)),
        _sharded((8, 32), (types.Split(axis=0),)),
        (16, 32),
        (types.Partial(reduction="sum"),),
        id="k_split_both_operands_becomes_partial",
    ),
    pytest.param(
        types.TensorType.from_shape((16, 8), expr.DType.bf16),
        _sharded((4, 8, 32), (types.Split(axis=2),)),
        (4, 16, 32),
        (types.Split(axis=2),),
        id="lower_rank_batched_rhs_split_maps_to_output",
    ),
    pytest.param(
        _sharded((16, 8), (types.Broadcast(),)),
        _sharded((8, 32), (types.Broadcast(),)),
        (16, 32),
        (types.Broadcast(),),
        id="replicated_operands_give_a_replicated_result_of_its_own_shape",
    ),
]


@pytest.mark.parametrize(("lhs", "rhs", "shape", "attrs"), CARRIES)
def test_matmul_carries_operand_shard(lhs, rhs, shape, attrs):
    out = infer_call(_MM, lhs, rhs)
    assert out.shape == shape
    assert out.layout.attrs == attrs
    assert math.prod(types.flatten(out.layout.shape)) == math.prod(shape), (
        "the result's layout domain holds the result, not an operand"
    )


@pytest.mark.parametrize(
    ("op", "lhs", "rhs", "attrs"),
    (
        pytest.param(
            nn.MatMul(b_layout="NK"),
            types.TensorType.from_shape((16, 8), expr.DType.bf16),
            _sharded((32, 8), (types.Split(axis=0),)),
            (types.Split(axis=1),),
            id="nk_n_split",
        ),
        pytest.param(
            nn.MatMul(a_layout="KM", b_layout="NK"),
            _sharded((8, 16), (types.Split(axis=1),)),
            types.TensorType.from_shape((32, 8), expr.DType.bf16),
            (types.Split(axis=0),),
            id="km_m_split",
        ),
        pytest.param(
            nn.MatMul(a_layout="KM", b_layout="NK"),
            _sharded((8, 16), (types.Split(axis=0),)),
            _sharded((32, 8), (types.Split(axis=1),)),
            (types.Partial(reduction="sum"),),
            id="km_nk_k_split",
        ),
    ),
)
def test_matmul_layout_literals_map_sharding(op, lhs, rhs, attrs):
    out = infer_call(op, lhs, rhs)
    assert out.shape == (16, 32)
    assert out.layout.attrs == attrs


def _make_double_partial_same_mesh_axis_errors():
    lhs = _sharded((16, 8), (types.Partial("sum"),))
    rhs = _sharded((8, 32), (types.Partial("sum"),))
    return OpCase(
        "double_partial_same_mesh_axis", _MM, (lhs, rhs), error=ExpectedError(match="mesh axis 0")
    )


def _make_partial_max_lhs_errors():
    lhs = _sharded((16, 8), (types.Partial("max"),))
    rhs = types.TensorType.from_shape((8, 32), expr.DType.bf16)
    return OpCase("partial_max_lhs", _MM, (lhs, rhs), error=ExpectedError(match="MatMul"))


CASES = (
    _make_lhs_splits_k_rhs_unsplit_is_invalid(),
    _make_incompatible_shard_errors(),
    _make_double_partial_same_mesh_axis_errors(),
    _make_partial_max_lhs_errors(),
    OpCase(
        "batch_from_broadcast_rhs",
        _MM,
        (types.TensorType.from_shape((1, 1, 4), expr.DType.f32), types.TensorType.from_shape((5, 4, 3), expr.DType.f32)),
        cost=Cost(
            flops={expr.DType.f32: 2 * 5 * 1 * 4 * 3},
            traffic=(
                cost.TrafficBytes(read=1 * 1 * 4 * 4),
                cost.TrafficBytes(read=5 * 4 * 3 * 4),
                cost.TrafficBytes(write=5 * 1 * 3 * 4),
            ),
        ),
    ),
    OpCase(
        "cta_projection_uses_split_rhs_batch",
        _MM,
        (
            types.TensorType.from_shape((1, 1, 4), expr.DType.f32),
            utils.make_shard_tensor_type(
                (5, 4, 3), mesh=_CTA_MESH, attrs=(types.Split(axis=0),), dtype=expr.DType.f32
            ),
        ),
        cost=Cost(
            flops={expr.DType.f32: 2 * 1 * 1 * 4 * 3},
            traffic=(
                cost.TrafficBytes(read=1 * 1 * 4 * 4),
                cost.TrafficBytes(read=1 * 4 * 3 * 4),
                cost.TrafficBytes(write=1 * 1 * 3 * 4),
            ),
            topology_level="cta",
            topologies=(_CTA,),
        ),
    ),
    OpCase(
        "km_nk_projection_reads_the_mapped_k_axis",
        nn.MatMul(a_layout="KM", b_layout="NK"),
        (
            utils.make_shard_tensor_type(
                (10, 4), mesh=_CTA_MESH, attrs=(types.Split(axis=0),), dtype=expr.DType.f32
            ),
            utils.make_shard_tensor_type(
                (3, 10), mesh=_CTA_MESH, attrs=(types.Split(axis=1),), dtype=expr.DType.f32
            ),
        ),
        cost=Cost(
            flops={expr.DType.f32: 2 * 4 * 2 * 3},
            traffic=(
                cost.TrafficBytes(read=4 * 2 * 4),
                cost.TrafficBytes(read=3 * 2 * 4),
                cost.TrafficBytes(write=4 * 3 * 4),
            ),
            topology_level="cta",
            topologies=(_CTA,),
        ),
    ),
    OpCase(
        "replicated_rectangular_projection_counts_the_whole_result",
        _MM,
        (
            utils.make_shard_tensor_type((4, 2), mesh=_CTA_MESH, attrs=(types.Broadcast(),), dtype=expr.DType.f32),
            utils.make_shard_tensor_type((2, 6), mesh=_CTA_MESH, attrs=(types.Broadcast(),), dtype=expr.DType.f32),
        ),
        cost=Cost(
            flops={expr.DType.f32: 2 * 4 * 6 * 2},
            traffic=(
                cost.TrafficBytes(read=4 * 2 * 4),
                cost.TrafficBytes(read=2 * 6 * 4),
                cost.TrafficBytes(write=4 * 6 * 4),
            ),
            topology_level="cta",
            topologies=(_CTA,),
        ),
    ),
    OpCase(
        "fp8_operands_count_their_flops_and_write_the_wider_result",
        nn.MatMul(out_dtype="f32"),
        (types.TensorType.from_shape((2, 3), expr.DType.fp8e4m3), types.TensorType.from_shape((3, 4), expr.DType.fp8e4m3)),
        cost=Cost(
            flops={expr.DType.fp8e4m3: 2 * 2 * 3 * 4},
            traffic=(
                cost.TrafficBytes(read=2 * 3 * 1),
                cost.TrafficBytes(read=3 * 4 * 1),
                cost.TrafficBytes(write=2 * 4 * 4),
            ),
        ),
    ),
    *(_layout_eval_case(param) for param in LAYOUT_CASES),
    OpCase(
        "fp8_into_f32",
        nn.MatMul(out_dtype="f32"),
        tuple((t.to(torch.float8_e4m3fn) for t in _SIXTEENS)),
        value=_SIXTEENS[0] @ _SIXTEENS[1],
        atol=0,
        rtol=0,
    ),
    OpCase(
        "bf16_into_f32_keeps_bits_bf16_would_drop",
        nn.MatMul(out_dtype=expr.DType.f32),
        (_BF16_LHS, _BF16_RHS),
        value=torch.tensor([[1.0 + 2.0 ** (-8)]]),
        atol=0,
        rtol=0,
    ),
    OpCase(
        "bf16_without_out_dtype_is_unchanged",
        nn.MatMul(out_dtype=None),
        (_BF16_LHS, _BF16_RHS),
        value=_BF16_LHS @ _BF16_RHS,
        atol=0,
        rtol=0,
    ),
    OpCase(
        "dtype_mismatch",
        _MM,
        (types.TensorType.from_shape((16, 8), expr.DType.bf16), types.TensorType.from_shape((8, 32), expr.DType.f32)),
        error=ExpectedError(match="dtype mismatch"),
    ),
    OpCase(
        "empty_m",
        _MM,
        (types.TensorType.from_shape((0, 16), expr.DType.bf16), types.TensorType.from_shape((16, 8), expr.DType.bf16)),
        type=types.TensorType.from_shape((0, 8), expr.DType.bf16),
    ),
    OpCase(
        "on_chip_without_a_target_is_undecided",
        nn.MatMul(out_dtype="f32"),
        (
            types.TensorType.from_shape((64, 16), expr.DType.bf16, "smem"),
            types.TensorType.from_shape((16, 32), expr.DType.bf16, "smem"),
        ),
        type=types.TensorType.from_shape((64, 32), expr.DType.f32, "umat"),
    ),
    OpCase(
        "cpu_on_chip_keeps_the_neutral_rule",
        _MM,
        (
            types.TensorType.from_shape((64, 16), expr.DType.f32, "smem"),
            types.TensorType.from_shape((16, 32), expr.DType.f32, "smem"),
        ),
        type=types.TensorType.from_shape((64, 32), expr.DType.f32, "smem"),
        target=cpu.Target(),
    ),
    OpCase(
        "fp8_out_dtype_names_the_result",
        nn.MatMul(out_dtype="f32"),
        (types.TensorType.from_shape((16, 8), expr.DType.fp8e4m3), types.TensorType.from_shape((8, 32), expr.DType.fp8e4m3)),
        type=types.TensorType.from_shape((16, 32), expr.DType.f32),
    ),
)
test_op_case = op_case_test(CASES)
