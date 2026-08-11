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

import pytest

from tests.ops.cost_utils import CostCase, run_cost_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Topology, make_mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Partial,
    Split,
)
from tilefoundry.visitor_registry.contexts import TrafficBytes

_MM = MatMul()


_M = make_mesh((4,))


_CTA = Topology("cta", 5)
_CTA_MESH = make_mesh((5,), topology=_CTA)


COST_CASES = [
    CostCase(
        name="batch_from_broadcast_rhs",
        op=_MM,
        inputs=(
            make_tensor_type((1, 1, 4), DType.f32),
            make_tensor_type((5, 4, 3), DType.f32),
        ),
        flops={DType.f32: 2 * 5 * 1 * 4 * 3},
        traffic=(
            TrafficBytes(read=1 * 1 * 4 * 4),
            TrafficBytes(read=5 * 4 * 3 * 4),
            TrafficBytes(write=5 * 1 * 3 * 4),
        ),
    ),
    CostCase(
        name="cta_projection_uses_split_rhs_batch",
        op=_MM,
        inputs=(
            make_tensor_type((1, 1, 4), DType.f32),
            make_shard_tensor_type(
                (5, 4, 3),
                mesh=_CTA_MESH,
                attrs=(Split(axis=0),),
                dtype=DType.f32,
            ),
        ),
        flops={DType.f32: 2 * 1 * 1 * 4 * 3},
        traffic=(
            TrafficBytes(read=1 * 1 * 4 * 4),
            TrafficBytes(read=1 * 4 * 3 * 4),
            TrafficBytes(write=1 * 1 * 3 * 4),
        ),
        level="cta",
        topologies=(_CTA,),
    ),
]


@pytest.mark.parametrize("case", COST_CASES, ids=lambda c: c.name)
def test_matmul_cost(case):
    run_cost_case(case)


def _sharded(shape, attrs):
    return make_shard_tensor_type(shape, mesh=_M, attrs=attrs, dtype=DType.bf16)


CASES = [
    TypeInferCase(
        name="dtype_mismatch",
        op=_MM,
        inputs=(make_tensor_type((16, 8), DType.bf16), make_tensor_type((8, 32), DType.f32)),
        expected=ExpectedError(match="dtype mismatch"),
    ),
    TypeInferCase(
        name="empty_m",
        op=_MM,
        inputs=(make_tensor_type((0, 16), DType.bf16), make_tensor_type((16, 8), DType.bf16)),
        expected=make_tensor_type((0, 8), DType.bf16),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_matmul_typeinfer(case):
    run_typeinfer_case(case)


def test_lhs_splits_k_rhs_unsplit_is_invalid():

    lhs = _sharded((16, 8), (Split(axis=1),))
    rhs = make_tensor_type((8, 32), DType.bf16)
    bad = TypeInferCase(
        name="lhs_k_split_rhs_unsplit",
        op=_MM,
        inputs=(lhs, rhs),
        expected=ExpectedError(match="contraction"),
    )
    run_typeinfer_case(bad)


def test_incompatible_shard_errors():

    lhs = _sharded((16, 8), (Split(axis=0),))
    rhs = _sharded((8, 32), (Split(axis=1),))
    bad = TypeInferCase(
        name="incompatible_shard",
        op=_MM,
        inputs=(lhs, rhs),
        expected=ExpectedError(match="incompatible|more than one"),
    )
    run_typeinfer_case(bad)


CARRIES = [
    pytest.param(
        make_tensor_type((16, 8), DType.bf16),
        _sharded((8, 32), (Split(axis=1),)),
        (16, 32),
        (Split(axis=1),),
        id="rhs_n_split_becomes_output_split",
    ),
    pytest.param(
        _sharded((16, 8), (Split(axis=0),)),
        make_tensor_type((8, 32), DType.bf16),
        (16, 32),
        (Split(axis=0),),
        id="lhs_m_split_becomes_output_split",
    ),
    pytest.param(
        _sharded((16, 8), (Split(axis=1),)),
        _sharded((8, 32), (Split(axis=0),)),
        (16, 32),
        (Partial(reduction="sum"),),
        id="k_split_both_operands_becomes_partial",
    ),
    pytest.param(
        make_tensor_type((16, 8), DType.bf16),
        _sharded((4, 8, 32), (Split(axis=2),)),
        (4, 16, 32),
        (Split(axis=2),),
        id="lower_rank_batched_rhs_split_maps_to_output",
    ),
]


@pytest.mark.parametrize(("lhs", "rhs", "shape", "attrs"), CARRIES)
def test_a_sharded_operand_carries_to_the_output(lhs, rhs, shape, attrs):
    out = infer_call(_MM, lhs, rhs)

    assert out.shape == shape
    assert out.layout.attrs == attrs


def test_double_partial_same_mesh_axis_errors():
    lhs = _sharded((16, 8), (Partial("sum"),))
    rhs = _sharded((8, 32), (Partial("sum"),))
    run_typeinfer_case(
        TypeInferCase(
            "double_partial_same_mesh_axis",
            _MM,
            (lhs, rhs),
            ExpectedError(match="mesh axis 0"),
        )
    )


def test_partial_max_lhs_errors():
    lhs = _sharded((16, 8), (Partial("max"),))
    rhs = make_tensor_type((8, 32), DType.bf16)
    run_typeinfer_case(
        TypeInferCase(
            "partial_max_lhs",
            _MM,
            (lhs, rhs),
            ExpectedError(match="MatMul"),
        )
    )
