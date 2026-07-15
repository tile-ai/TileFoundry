"""Transpose typeinfer over sharded (factorized) layouts.

When a tensor axis is split, its ``Layout`` carries more positions than
the tensor has axes (the split axis factorizes into mesh-extent × per-shard
sub-axes). Transposing must reorder the layout positions by their owning tensor
axis — keeping each tensor axis's sub-axes together — and remap the
``Split`` / ``Partial`` references to the moved layout positions, rather than
indexing layout positions with the tensor-axis permutation directly.
"""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    TypeInferCase,
    infer_call,
    mesh,
    raw_shard_tensor_type,
    run_typeinfer_case,
    ten,
)
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.types import DType, make_shard_tensor_type
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    ShardLayout,
    Split,
    shard_layout_local_shape,
)

# A 4-axis mesh whose split axis factorizes a tensor dim into mesh-extent ×
# per-shard sub-positions (cluster, cta, warp, lane).
_M = mesh((1, 128, 8, 32), ("cluster", "cta", "warp", "lane"))
_B4 = (Broadcast(), Broadcast(), Broadcast(), Broadcast())
_T10 = Transpose(perm=(1, 0))

CASES = [
    # unsharded: shape permutes, layout passes through.
    TypeInferCase("unsharded", _T10, (ten((4, 8), DType.f32),), ten((8, 4), DType.f32)),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_transpose_typeinfer(case):
    run_typeinfer_case(case)


# ── sharded carries ───────────────────────────────────────────────────────
# Transpose is a view: it reorders layout positions by owning tensor axis
# rather than recomputing a fresh canonical layout, so its output strides are
# a permutation of the input's, not necessarily C-order. These cases check
# output shape and which mesh axis stays genuinely `Split` (and its local
# extent), not the internal layout position / stride a valid `Transpose`
# permutation produces.


def test_factorized_split_reorders_subaxes():
    """tensor (4096, 2048), axis 0 split on cta -> layout (128, 32, 2048). The
    transpose moves tensor axis 1 (layout pos 2) first; axis 0's
    sub-positions (layout pos 0, 1) follow in order; the Split moves from
    layout pos 0 to pos 1."""
    x_ty = make_shard_tensor_type(
        (4096, 2048),
        mesh=_M,
        attrs=(Broadcast(), Split(axis=0), Broadcast(), Broadcast()),
        dtype=DType.bf16,
    )
    ty = infer_call(_T10, x_ty)
    assert tuple(ty.shape) == (2048, 4096)
    assert isinstance(ty.layout, ShardLayout)
    assert ty.layout.attrs == (Broadcast(), Split(axis=1), Broadcast(), Broadcast())
    assert shard_layout_local_shape(ty.layout)[1] == 1


def test_implicit_strides_no_crash():
    """implicit (None) strides: shape + attrs permute, output keeps implicit
    strides (regression: no None-stride indexing crash)."""
    x_ty = raw_shard_tensor_type(
        (16, 8), (16, 8), None, (Split(0), *_B4[1:]), _M, dtype=DType.bf16,
    )
    ty = infer_call(_T10, x_ty)
    assert tuple(ty.shape) == (8, 16)
    assert isinstance(ty.layout, ShardLayout)
    assert ty.layout.attrs == (Split(1), *_B4[1:])


@pytest.mark.parametrize(
    "shape,perm",
    [((2, 3, 4), (2, 1, 0)), ((1, 5, 3, 8), (0, 2, 1, 3))],
    ids=["perm_reverse", "perm_head_to_front"],
)
def test_transpose_evaluate(shape, perm):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    run_eval_case(EvalCase("", Transpose(perm=perm), (x,), x.permute(*perm)))
