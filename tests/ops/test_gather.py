"""HIR Gather value oracle: select along ``axis`` by (multi-dim) indices."""
from __future__ import annotations

import itertools

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import gather
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Split
from tilefoundry.passes.transforms import HirToTirPass

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


# ── batched gather (explicit TF-style ``batch_dims``) ────────────────────────
#
# ``Gather(axis=a, batch_dims=b)`` batches the leading ``b`` dims (which must
# match between ``x`` and ``index``); the output is
# ``x[:a] + index.shape[b:] + x[a+1:]``. ``batch_dims=0`` (default) keeps the
# existing non-batched insert semantics for every prior call — including a
# leading-dim shape coincidence, which does NOT implicitly switch to batched.


def _ref_batched(x, index, axis, batch_dims):
    """Definitional reference for the TF-style batched gather:
    ``out[c.., i.., t..] = x[c.., index[b.., i..], t..]`` where ``c..`` are the
    ``axis`` leading dims of ``x`` (the first ``batch_dims`` of which also index
    ``index``), ``i..`` are ``index``'s remaining dims, and ``t..`` are ``x``'s
    trailing dims. A plain nested loop, independent of the vectorized op."""
    axis %= x.ndim
    b = batch_dims
    batch = x.shape[:b]
    mid = x.shape[b:axis]
    trail = x.shape[axis + 1 :]
    rem = index.shape[b:]
    out = torch.empty(*x.shape[:axis], *rem, *trail, dtype=x.dtype)
    for bi in itertools.product(*[range(d) for d in batch]):
        for mi in itertools.product(*[range(d) for d in mid]):
            for ii in itertools.product(*[range(d) for d in rem]):
                j = int(index[bi + ii])
                out[bi + mi + ii] = x[bi + mi + (j,)]
    return out


def test_gather_batched_axis1() -> None:
    """AC-3-1 shape/value on a small analog: a batched middle-axis gather with
    one batch dim collapses the aligned leading dim instead of inserting it."""
    torch.manual_seed(0)
    x = torch.randn(2, 7, 5)
    index = torch.randint(0, 7, (2, 4), dtype=torch.int32)  # index.shape[:1]==x.shape[:1]
    expected = _ref_batched(x, index, axis=1, batch_dims=1)
    assert tuple(expected.shape) == (2, 4, 5)
    run_eval_case(EvalCase("", Gather(axis=1, batch_dims=1), (x, index), expected))


def test_gather_two_batch_anti_flatten() -> None:
    """Two batch dims are collapsed, not flattened into the output: the aligned
    ``[2, 3]`` prefix appears once, not duplicated."""
    torch.manual_seed(1)
    x = torch.randn(2, 3, 7, 5)
    index = torch.randint(0, 7, (2, 3, 4), dtype=torch.int32)
    expected = _ref_batched(x, index, axis=2, batch_dims=2)
    assert tuple(expected.shape) == (2, 3, 4, 5)  # NOT [2,3,2,3,4,5]
    run_eval_case(EvalCase("", Gather(axis=2, batch_dims=2), (x, index), expected))


def test_gather_batch_dims_less_than_axis() -> None:
    """``batch_dims < axis``: the dims between the batch prefix and the gather
    axis pass through, and the same per-batch index applies across them."""
    torch.manual_seed(3)
    x = torch.randn(2, 4, 7, 5)
    index = torch.randint(0, 7, (2, 3), dtype=torch.int32)  # batch_dims=1 < axis=2
    expected = _ref_batched(x, index, axis=2, batch_dims=1)
    assert tuple(expected.shape) == (2, 4, 3, 5)
    run_eval_case(EvalCase("", Gather(axis=2, batch_dims=1), (x, index), expected))


def test_gather_default_batch_dims_zero_keeps_non_batched() -> None:
    """Default ``batch_dims=0`` keeps non-batched insert semantics even when the
    leading dims coincide (``index(6,)`` on ``x(6,3,4)`` axis 1 → ``[6,6,4]``),
    and the explicit ``batch_dims=1`` form of the same index → ``[6,4]``."""
    torch.manual_seed(2)
    x = torch.randn(6, 3, 4)
    idx6 = torch.randint(0, 3, (6,), dtype=torch.int32)
    non_batched = _gather_ref(x, 1, idx6)
    assert tuple(non_batched.shape) == (6, 6, 4)
    run_eval_case(EvalCase("", Gather(axis=1), (x, idx6), non_batched))

    batched = _ref_batched(x, idx6, axis=1, batch_dims=1)
    assert tuple(batched.shape) == (6, 4)
    run_eval_case(EvalCase("", Gather(axis=1, batch_dims=1), (x, idx6), batched))


BATCHED_TYPEINFER_CASES = [
    # AC-3-1: KV row gather, one batch dim.
    TypeInferCase(
        "ac_kv_row_gather_batched",
        Gather(axis=1, batch_dims=1),
        (ten((1, 16512, 512), DType.bf16), ten((1, 1, 640), DType.i32)),
        ten((1, 1, 640, 512), DType.bf16),
    ),
    # AC-3-2: embedding lookup, axis 0, default batch_dims.
    TypeInferCase(
        "ac_embedding_lookup",
        Gather(axis=0),
        (ten((129280, 4096), DType.bf16), ten((1, 1), DType.i64)),
        ten((1, 1, 4096), DType.bf16),
    ),
    # AC-3-3: stacked-weight gather, axis 0, default batch_dims.
    TypeInferCase(
        "ac_stacked_weight_gather",
        Gather(axis=0),
        (ten((256, 2048, 4096), DType.f32), ten((1, 6), DType.i64)),
        ten((1, 6, 2048, 4096), DType.f32),
    ),
    # Shape coincidence with default batch_dims=0 stays non-batched.
    TypeInferCase(
        "coincident_leading_dim_default_non_batched",
        Gather(axis=1),
        (ten((6, 3, 4), DType.f32), ten((6,), DType.i32)),
        ten((6, 6, 4), DType.f32),
    ),
    # Same index, explicit batch_dims=1, collapses the batch dim.
    TypeInferCase(
        "coincident_leading_dim_explicit_batched",
        Gather(axis=1, batch_dims=1),
        (ten((6, 3, 4), DType.f32), ten((6,), DType.i32)),
        ten((6, 4), DType.f32),
    ),
    # batch_dims < axis: dims between batch prefix and gather axis pass through.
    TypeInferCase(
        "batch_dims_less_than_axis",
        Gather(axis=2, batch_dims=1),
        (ten((2, 4, 7, 5), DType.f32), ten((2, 3), DType.i32)),
        ten((2, 4, 3, 5), DType.f32),
    ),
    # batch_dims == rank(index) is the boundary "one scalar index per batch".
    TypeInferCase(
        "batch_dims_equals_index_rank_boundary",
        Gather(axis=1, batch_dims=1),
        (ten((6, 3, 4), DType.f32), ten((6,), DType.i32)),
        ten((6, 4), DType.f32),
    ),
    # batch_dims must not exceed axis.
    TypeInferCase(
        "batch_dims_exceeds_axis_rejected",
        Gather(axis=0, batch_dims=1),
        (ten((6, 3, 4), DType.f32), ten((6,), DType.i32)),
        ExpectedError(match="batch_dims", exc=TypeError),
    ),
    # batch dims must match between x and index.
    TypeInferCase(
        "batch_dims_prefix_mismatch_rejected",
        Gather(axis=1, batch_dims=1),
        (ten((6, 3, 4), DType.f32), ten((5, 2), DType.i32)),
        ExpectedError(match="batch", exc=TypeError),
    ),
    # index must be an integer tensor (spec "integer index tensor") — reject
    # a float index rather than silently truncating it in eval; both batched
    # and non-batched paths.
    TypeInferCase(
        "float_index_rejected_non_batched",
        Gather(axis=1),
        (ten((6, 3, 4), DType.f32), ten((2,), DType.f32)),
        ExpectedError(match="integer", exc=TypeError),
    ),
    TypeInferCase(
        "float_index_rejected_batched",
        Gather(axis=1, batch_dims=1),
        (ten((2, 3, 4), DType.f32), ten((2, 5), DType.f32)),
        ExpectedError(match="integer", exc=TypeError),
    ),
    # A batched gather over a sharded operand is not yet supported: fail-closed
    # with a named error (the batch_dims attribute stays a stable interface for
    # a future sharded/collective implementation). Either operand triggers it —
    # a sharded source, or an unsharded source with a sharded index.
    TypeInferCase(
        "sharded_source_batched_gather_not_implemented",
        Gather(axis=1, batch_dims=1),
        (sharded((6, 4, 8), (Split(0),), _M), ten((6, 2), DType.i32)),
        ExpectedError(match="Gather: batched gather .* sharded operand", exc=NotImplementedError),
    ),
    TypeInferCase(
        "sharded_index_batched_gather_not_implemented",
        Gather(axis=1, batch_dims=1),
        (ten((6, 4, 8), DType.f32), sharded((6, 2), (Split(0),), _M, dtype=DType.i32)),
        ExpectedError(match="Gather: batched gather .* sharded operand", exc=NotImplementedError),
    ),
]


@pytest.mark.parametrize("case", BATCHED_TYPEINFER_CASES, ids=lambda c: c.name)
def test_gather_batched_typeinfer(case):
    run_typeinfer_case(case)


@func
def _batched_gather_lower_fn(
    x: Tensor[(2, 3, 4), "f32"], idx: Tensor[(2, 5), "i32"]
) -> Tensor[(2, 5, 4), "f32"]:
    return gather(x, idx, axis=1, batch_dims=1)


def test_batched_gather_lowering_rejected() -> None:
    """A ``batch_dims>0`` gather must not silently fall through to the existing
    single-coordinate TensorView lowering: HIR->TIR fail-closes with a named
    ``Gather`` error."""
    module = Module(name="t", functions=(_batched_gather_lower_fn,), entry=_batched_gather_lower_fn.name)
    with pytest.raises(NotImplementedError, match="Gather: batched gather lowering"):
        HirToTirPass().run(module)
