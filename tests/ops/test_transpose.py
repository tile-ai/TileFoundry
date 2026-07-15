"""Transpose typeinfer over sharded (factorized cute) layouts.

When a tensor axis is split, its cute ``Layout`` carries more positions than
the tensor has axes (the split axis factorizes into mesh-extent × per-shard
sub-axes). Transposing must reorder the cute positions by their owning tensor
axis — keeping each tensor axis's sub-axes together — and remap the
``Split`` / ``Partial`` references to the moved cute positions, rather than
indexing cute positions with the tensor-axis permutation directly.
"""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    TypeInferCase,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Split

# A 4-axis mesh whose split axis factorizes a tensor dim into mesh-extent ×
# per-shard sub-positions (cluster, cta, warp, lane).
_M = mesh((1, 128, 8, 32), ("cluster", "cta", "warp", "lane"))
_B4 = (Broadcast(), Broadcast(), Broadcast(), Broadcast())
_T10 = Transpose(perm=(1, 0))

CASES = [
    # unsharded: shape permutes, layout passes through.
    TypeInferCase("unsharded", _T10, (ten((4, 8), DType.f32),), ten((8, 4), DType.f32)),
    # tensor (4096, 2048), axis 0 split on cta -> cute (128, 32, 2048). The
    # transpose moves tensor axis 1 (cute pos 2) first; axis 0's sub-positions
    # (cute pos 0, 1) follow in order; the Split moves from cute pos 0 to pos 1.
    TypeInferCase(
        "factorized_split_reorders_subaxes",
        _T10,
        (
            sharded(
                (4096, 2048),
                (Broadcast(), Split(axis=0), Broadcast(), Broadcast()),
                _M,
                cute=(128, 32, 2048),
                strides=(65536, 2048, 1),
                dtype=DType.bf16,
            ),
        ),
        sharded(
            (2048, 4096),
            (Broadcast(), Split(axis=1), Broadcast(), Broadcast()),
            _M,
            cute=(2048, 128, 32),
            strides=(1, 65536, 2048),
            dtype=DType.bf16,
        ),
    ),
    # implicit (None) strides: shape + attrs permute, output keeps implicit
    # strides (regression: no None-stride indexing crash).
    TypeInferCase(
        "implicit_strides",
        _T10,
        (sharded((16, 8), (Split(0), *_B4[1:]), _M, strides=None, dtype=DType.bf16),),
        sharded((8, 16), (Split(1), *_B4[1:]), _M, strides=None, dtype=DType.bf16),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_transpose_typeinfer(case):
    run_typeinfer_case(case)


@pytest.mark.parametrize(
    "shape,perm",
    [((2, 3, 4), (2, 1, 0)), ((1, 5, 3, 8), (0, 2, 1, 3))],
    ids=["perm_reverse", "perm_head_to_front"],
)
def test_transpose_evaluate(shape, perm):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    run_eval_case(EvalCase("", Transpose(perm=perm), (x,), x.permute(*perm)))
