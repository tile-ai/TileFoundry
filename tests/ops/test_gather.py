"""HIR Gather value oracle: select along ``axis`` by (multi-dim) indices."""
from __future__ import annotations

import pytest
import torch

from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Split
from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)

_F = DType.f32
_M = mesh((2,))


def _gather_ref(x, axis, idx):
    """Reference gather: select along ``axis`` by (possibly multi-dim) ``idx``,
    expanding the indexed axis into ``idx``'s shape."""
    axis %= x.ndim
    flat = x.index_select(axis, idx.flatten().long())
    return flat.reshape(*x.shape[:axis], *idx.shape, *x.shape[axis + 1 :])


@pytest.mark.parametrize(
    "axis,x_shape,idx",
    [
        (0, (6, 3, 4), [[0, 5], [2, 3]]),  # 2-D index grid -> [2, 2, 3, 4]
        (1, (6, 3, 4), [2, 0]),  # gather along a middle axis
        (-1, (6, 3, 4), [3, 0, 1]),  # negative axis normalizes to the last axis
        (1, (6, 3, 4), 2),  # scalar index on a middle axis -> [6, 4]
    ],
    ids=["axis0_2d_index", "axis1_1d_index", "neg_axis_last", "axis1_scalar_index"],
)
def test_gather_evaluate(axis, x_shape, idx):
    torch.manual_seed(0)
    x = torch.randn(*x_shape)
    idx_t = torch.tensor(idx, dtype=torch.int32)
    run_eval_case(EvalCase("", Gather(axis=axis), (x, idx_t), _gather_ref(x, axis, idx_t)))


TYPEINFER_CASES = [
    TypeInferCase(
        "neg_axis_normalizes",
        Gather(axis=-1),
        (ten((2, 3, 4), DType.f32), ten((2,), DType.i32)),
        ten((2, 3, 2), DType.f32),
    ),
    TypeInferCase(
        "axis_out_of_range",
        Gather(axis=5),
        (ten((2, 3, 4), DType.f32), ten((2,), DType.i32)),
        ExpectedError(match="out of range", exc=TypeError),
    ),
    # ── sharded input, single-index gather on a NON-sharded middle axis ────────
    # scalar index removes the middle axis's cute position; the Split on axis 0
    # is remapped onto the surviving positions (the wsum / embed gap).
    TypeInferCase(
        "sharded_mid_axis_scalar_drops_position",
        Gather(axis=1),
        (sharded((6, 4, 8), (Split(0),), _M), ten((), DType.i32)),
        sharded((6, 8), (Split(0),), _M, cute=(6, 8), strides=(32, 1)),
    ),
    # (1,)-shaped index keeps the middle axis at size 1; strides/attrs unchanged.
    TypeInferCase(
        "sharded_mid_axis_single_index_keeps_unit",
        Gather(axis=1),
        (sharded((6, 4, 8), (Split(0),), _M), ten((1,), DType.i32)),
        sharded((6, 1, 8), (Split(0),), _M, cute=(6, 1, 8), strides=(32, 8, 1)),
    ),
    # a non-sharded LEADING axis scalar-gather remaps the Split from axis 1 -> 0.
    TypeInferCase(
        "sharded_leading_axis_scalar_remaps_split",
        Gather(axis=0),
        (sharded((6, 4, 8), (Split(1),), _M), ten((), DType.i32)),
        sharded((4, 8), (Split(0),), _M, cute=(4, 8), strides=(8, 1)),
    ),
    # regression (AC-4-2): a gather ALONG the Split (sharded) axis is out of the
    # slice's scope, so the input layout carries through unchanged.
    TypeInferCase(
        "sharded_axis_gather_passes_layout_through",
        Gather(axis=0),
        (sharded((6, 4, 8), (Split(0),), _M), ten((), DType.i32)),
        sharded((4, 8), (Split(0),), _M, cute=(6, 4, 8), strides=(32, 8, 1)),
    ),
    # regression (AC-4-2): a multi-index gather whose total size is 1 (e.g.
    # (1, 1)) is NOT a slice — the input layout carries through unchanged.
    TypeInferCase(
        "multi_index_total_size_one_passes_layout_through",
        Gather(axis=1),
        (sharded((6, 4, 8), (Split(0),), _M), ten((1, 1), DType.i32)),
        sharded((6, 1, 1, 8), (Split(0),), _M, cute=(6, 4, 8), strides=(32, 8, 1)),
    ),
]


@pytest.mark.parametrize("case", TYPEINFER_CASES, ids=lambda c: c.name)
def test_gather_typeinfer(case):
    run_typeinfer_case(case)
