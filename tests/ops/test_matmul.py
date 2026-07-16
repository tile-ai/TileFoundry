"""MatMul typeinfer over the relation-driven path.

MatMul derives its output shape and ``ShardLayout`` from a forward access
relation (iteration domain ``[batch..., M, N, K]``; lhs ``[batch.., M, K]``,
rhs ``[batch.., K, N]``, output ``[batch.., M, N]`` with K reduced). With no
sharded input the output layout passes through ``lhs.layout`` (unchanged from
the hand-written rule). An rhs N-split becomes an output ``Split`` on N; a
K-split on both operands becomes an output ``Partial`` on the N axis; an
lhs M-split / batch-split passes through to the matching output axis.
"""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Partial,
    Split,
)

_MM = MatMul()

# A single mesh object shared by an input shard and its expectation so the
# output ShardLayout's mesh compares equal.
_M = make_mesh((4,))


def _sharded(shape, attrs):
    return make_shard_tensor_type(shape, mesh=_M, attrs=attrs, dtype=DType.bf16)


CASES = [
    # plain 2D — output layout passes through lhs.layout (None here)
    TypeInferCase(
        name="plain_2d",
        op=_MM,
        inputs=(make_tensor_type((16, 8), DType.bf16), make_tensor_type((8, 32), DType.bf16)),
        expected=make_tensor_type((16, 32), DType.bf16),
    ),
    # plain batched — equal batch dims
    TypeInferCase(
        name="plain_batched",
        op=_MM,
        inputs=(make_tensor_type((4, 16, 8), DType.bf16), make_tensor_type((4, 8, 32), DType.bf16)),
        expected=make_tensor_type((4, 16, 32), DType.bf16),
    ),
    # batch broadcast — lhs batch 1 broadcasts to rhs batch
    TypeInferCase(
        name="batch_broadcast",
        op=_MM,
        inputs=(make_tensor_type((1, 16, 8), DType.bf16), make_tensor_type((4, 8, 32), DType.bf16)),
        expected=make_tensor_type((4, 16, 32), DType.bf16),
    ),
    # batch broadcast across different ranks — 2D lhs against batched rhs
    TypeInferCase(
        name="batch_broadcast_lhs_unbatched",
        op=_MM,
        inputs=(make_tensor_type((16, 8), DType.bf16), make_tensor_type((4, 8, 32), DType.bf16)),
        expected=make_tensor_type((4, 16, 32), DType.bf16),
    ),
    # right-aligned broadcast with mixed ranks and a size-1 dim
    TypeInferCase(
        name="batch_broadcast_mixed_rank",
        op=_MM,
        inputs=(make_tensor_type((2, 1, 16, 8), DType.bf16), make_tensor_type((3, 8, 32), DType.bf16)),
        expected=make_tensor_type((2, 3, 16, 32), DType.bf16),
    ),
    TypeInferCase(
        name="batch_broadcast_higher_rank_lhs",
        op=_MM,
        inputs=(make_tensor_type((2, 3, 16, 8), DType.bf16), make_tensor_type((3, 8, 32), DType.bf16)),
        expected=make_tensor_type((2, 3, 16, 32), DType.bf16),
    ),
    # dynamic batch dim — same DimVar both sides
    TypeInferCase(
        name="dynamic_batch",
        op=_MM,
        inputs=(
            make_tensor_type((DimVar("B", 1, 64), 16, 8), DType.bf16),
            make_tensor_type((DimVar("B", 1, 64), 8, 32), DType.bf16),
        ),
        expected=make_tensor_type((DimVar("B", 1, 64), 16, 32), DType.bf16),
    ),
    # dtype mismatch → error
    TypeInferCase(
        name="dtype_mismatch",
        op=_MM,
        inputs=(make_tensor_type((16, 8), DType.bf16), make_tensor_type((8, 32), DType.f32)),
        expected=ExpectedError(match="dtype mismatch"),
    ),
    # K-dim mismatch → error
    TypeInferCase(
        name="k_dim_mismatch",
        op=_MM,
        inputs=(make_tensor_type((16, 8), DType.bf16), make_tensor_type((4, 32), DType.bf16)),
        expected=ExpectedError(match="contraction"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_matmul_typeinfer(case):
    run_typeinfer_case(case)


def test_lhs_splits_k_rhs_unsplit_is_invalid():
    # The contraction dim K is split on lhs but not on rhs — the shards of K
    # have nothing to contract against, so the sharding is inconsistent.
    lhs = _sharded((16, 8), (Split(axis=1),))  # Split on K
    rhs = make_tensor_type((8, 32), DType.bf16)  # K unsharded
    bad = TypeInferCase(
        name="lhs_k_split_rhs_unsplit",
        op=_MM,
        inputs=(lhs, rhs),
        expected=ExpectedError(match="contraction"),
    )
    run_typeinfer_case(bad)


def test_lower_rank_batched_rhs_split_maps_to_output():
    # rhs is batched and N-split; lhs is plain 2D (no batch). The rhs batch dim
    # right-aligns to the output's batch axis and its N-split survives.
    lhs = make_tensor_type((16, 8), DType.bf16)
    rhs = _sharded((4, 8, 32), (Split(axis=2),))
    out = infer_call(_MM, lhs, rhs)
    assert out.shape == (4, 16, 32)
    # N is output axis 2.
    assert out.layout.attrs == (Split(axis=2),)


def test_incompatible_shard_errors():
    # lhs M-split and rhs N-split on the SAME mesh axis can't both land on the
    # output: one mesh axis would bind two output layout axes.
    lhs = _sharded((16, 8), (Split(axis=0),))
    rhs = _sharded((8, 32), (Split(axis=1),))
    bad = TypeInferCase(
        name="incompatible_shard",
        op=_MM,
        inputs=(lhs, rhs),
        expected=ExpectedError(match="incompatible|more than one"),
    )
    run_typeinfer_case(bad)


# ── sharded carries ───────────────────────────────────────────────────────
# MatMul derives its output ShardLayout from the shard-propagation engine
# (mesh-axis bindings), not by carrying a hand-picked layout literal, so these
# check output shape and which mesh axis holds Split / Partial, not the
# internal layout position count a valid derivation happens to produce.


def test_rhs_n_split_becomes_output_split():
    lhs = make_tensor_type((16, 8), DType.bf16)
    rhs = _sharded((8, 32), (Split(axis=1),))
    out = infer_call(_MM, lhs, rhs)
    assert out.shape == (16, 32)
    assert out.layout.attrs == (Split(axis=1),)


def test_k_split_both_operands_becomes_partial():
    lhs = _sharded((16, 8), (Split(axis=1),))
    rhs = _sharded((8, 32), (Split(axis=0),))
    out = infer_call(_MM, lhs, rhs)
    assert out.shape == (16, 32)
    assert out.layout.attrs == (Partial(reduction="sum"),)


def test_lhs_m_split_becomes_output_split():
    lhs = _sharded((16, 8), (Split(axis=0),))
    rhs = make_tensor_type((8, 32), DType.bf16)
    out = infer_call(_MM, lhs, rhs)
    assert out.shape == (16, 32)
    assert out.layout.attrs == (Split(axis=0),)


def test_batch_split_passes_through():
    lhs = _sharded((4, 16, 8), (Split(axis=0),))
    rhs = make_tensor_type((4, 8, 32), DType.bf16)
    out = infer_call(_MM, lhs, rhs)
    assert out.shape == (4, 16, 32)
    assert out.layout.attrs == (Split(axis=0),)


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [((3, 4), (4, 5)), ((2, 3, 4), (2, 4, 5))],
    ids=["mm_2d", "mm_batched"],
)
def test_matmul_evaluate(lhs_shape, rhs_shape):
    torch.manual_seed(0)
    lhs, rhs = torch.randn(*lhs_shape), torch.randn(*rhs_shape)
    run_eval_case(EvalCase("", _MM, (lhs, rhs), torch.matmul(lhs, rhs), atol=1e-5))
