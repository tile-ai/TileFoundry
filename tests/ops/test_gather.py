"""Gather's batched, sharded and unlowerable cases: the ``batch_dims`` prefix is
collapsed rather than flattened and is never entered by accident, a sharded
operand migrates its shard attrs or derives a ``Partial``, and the batched
lowering fails closed rather than silently take the single-coordinate path."""
from __future__ import annotations

import itertools

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
)
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import gather
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial, ShardLayout, Split
from tilefoundry.passes.transforms import HirToTirPass

_M = make_mesh((2,))


def _gather_ref(x, axis, idx):
    """Reference gather: select along ``axis`` by (possibly multi-dim) ``idx``,
    expanding the indexed axis into ``idx``'s shape."""
    axis %= x.ndim
    flat = x.index_select(axis, idx.flatten().long())
    return flat.reshape(*x.shape[:axis], *idx.shape, *x.shape[axis + 1 :])


# ── sharded input, shard-attr migration ───────────────────────────────────
#
# Gather output is a new tensor: the internal layout is always natural
# contiguous over the output shape, so these check tensor shape and shard
# attrs (the actual contract) rather than the derived layout.


def test_gather_shard_leading_axis_scalar_remaps_split():
    """A non-sharded LEADING axis scalar-gather renumbers the Split from axis
    1 -> 0 (axis 0 is removed by the gather)."""
    ty = infer_call(Gather(axis=0), make_shard_tensor_type((6, 4, 8), mesh=_M, attrs=(Split(1),)), make_tensor_type((), DType.i32))
    assert tuple(ty.shape) == (4, 8)
    assert isinstance(ty.layout, ShardLayout) and ty.layout.attrs == (Split(0),)


def test_gather_shard_axis_gather_derives_partial():
    """A gather ALONG the Split (sharded) axis is a masked-gather: every
    device already holds a zero-filled partial of the gathered rows, so the
    output sums to the true gather across that mesh axis."""
    ty = infer_call(Gather(axis=0), make_shard_tensor_type((6, 4, 8), mesh=_M, attrs=(Split(0),)), make_tensor_type((), DType.i32))
    assert tuple(ty.shape) == (4, 8)
    assert isinstance(ty.layout, ShardLayout) and ty.layout.attrs == (
        Partial(reduction="sum"),
    )


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


def test_gather_two_batch_anti_flatten() -> None:
    """Two batch dims are collapsed, not flattened into the output: the aligned
    ``[2, 3]`` prefix appears once, not duplicated."""
    torch.manual_seed(1)
    x = torch.randn(2, 3, 7, 5)
    index = torch.randint(0, 7, (2, 3, 4), dtype=torch.int32)
    expected = _ref_batched(x, index, axis=2, batch_dims=2)
    assert tuple(expected.shape) == (2, 3, 4, 5)  # NOT [2,3,2,3,4,5]
    run_eval_case(EvalCase("", Gather(axis=2, batch_dims=2), (x, index), expected))


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
    # batch_dims must not exceed axis.
    TypeInferCase(
        "batch_dims_exceeds_axis_rejected",
        Gather(axis=0, batch_dims=1),
        (make_tensor_type((6, 3, 4), DType.f32), make_tensor_type((6,), DType.i32)),
        ExpectedError(match="batch_dims"),
    ),
    # index must be an integer tensor (spec "integer index tensor") — reject
    # a float index rather than silently truncating it in eval.
    TypeInferCase(
        "float_index_rejected_non_batched",
        Gather(axis=1),
        (make_tensor_type((6, 3, 4), DType.f32), make_tensor_type((2,), DType.f32)),
        ExpectedError(match="integer"),
    ),
    # A batched gather over a sharded operand is not yet supported: fail-closed
    # with a named error (the batch_dims attribute stays a stable interface for
    # a future sharded/collective implementation). Either operand triggers it;
    # a sharded source stands for a sharded index too.
    TypeInferCase(
        "sharded_source_batched_gather_not_implemented",
        Gather(axis=1, batch_dims=1),
        (make_shard_tensor_type((6, 4, 8), mesh=_M, attrs=(Split(0),)), make_tensor_type((6, 2), DType.i32)),
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
