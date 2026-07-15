"""Reshape typeinfer.

Reshape is a view: an unsharded input reshapes to an unsharded output; a
genuine sharding carries when every cute position lies entirely within one new
axis (size-1 axes are inserted/dropped freely and the cute factorization of the
surviving positions is preserved, with ``Split`` cute-axis references remapped),
or when a ``Split``-bound position divides across a new-axis boundary at a
point its bound mesh extent evenly divides (``Split`` relocates to the
mesh-extent-sized sub-position, keeping local extent 1, with any remainder
carried forward as a plain cute position); a reshape that cannot be expressed
either way fails closed (no fake layout). See ``docs/spec/hir.md`` §1.3
``Reshape``.
"""
from __future__ import annotations

import pytest
import torch

from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Layout, ShardLayout
from tilefoundry.ir.types.shard.shard_layout import (
    Partial,
    Split,
    shard_layout_local_shape,
)
from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    mesh,
    run_typeinfer_case,
    sharded,
    ten,
)

_F = DType.f32
_M = mesh((4,))


def _reshape(new_shape):
    return Reshape(new_shape=new_shape)


CASES = [
    # ── unsharded ────────────────────────────────────────────────────────────
    TypeInferCase("unsharded", _reshape((32,)), (ten((4, 8), _F),), ten((32,), _F)),
    # ── aligned sharded carries ───────────────────────────────────────────────
    # merge: cute (16, 8) -> (128,); the positions' product equals the new axis.
    TypeInferCase(
        "merge_carries",
        _reshape((128,)),
        (sharded((16, 8), (Split(0),), _M),),
        sharded((128,), (Split(0),), _M, cute=(16, 8), strides=(8, 1)),
    ),
    # insert a unit axis after the split axis: the Split stays on axis 0.
    TypeInferCase(
        "insert_unit_axis",
        _reshape((32, 1, 128)),
        (sharded((32, 128), (Split(0),), _M),),
        sharded((32, 1, 128), (Split(0),), _M, cute=(32, 1, 128), strides=(128, 0, 1)),
    ),
    # insert a leading unit axis: the split moves from axis 0 to axis 1.
    TypeInferCase(
        "reshuffle_leading_unit",
        _reshape((1, 32, 128)),
        (sharded((32, 128), (Split(0),), _M),),
        sharded((1, 32, 128), (Split(1),), _M, cute=(1, 32, 128), strides=(0, 128, 1)),
    ),
    # drop a leading unit axis: the split moves from axis 1 back to axis 0.
    TypeInferCase(
        "remove_leading_unit",
        _reshape((32, 128)),
        (sharded((1, 32, 128), (Split(1),), _M),),
        sharded((32, 128), (Split(0),), _M),
    ),
    # ── mesh-axis value states carry without a cute axis ──────────────────────
    # a Partial is a mesh-axis value state with no cute axis; it carries through
    # the reshape unchanged (no cute-position remap).
    TypeInferCase(
        "partial_carries",
        _reshape((32, 1, 128)),
        (sharded((32, 128), (Partial("sum"),), _M),),
        sharded((32, 1, 128), (Partial("sum"),), _M, cute=(32, 1, 128), strides=(128, 0, 1)),
    ),
    # mix on a two-axis mesh: the Split remaps to its new cute position while the
    # Partial carries through unchanged.
    TypeInferCase(
        "split_remaps_partial_carries",
        _reshape((1, 32, 128)),
        (sharded((32, 128), (Split(0), Partial("sum")), mesh((2, 2))),),
        sharded(
            (1, 32, 128),
            (Split(1), Partial("sum")),
            mesh((2, 2)),
            cute=(1, 32, 128),
            strides=(0, 128, 1),
        ),
    ),
    # ── dividing sharded splits carry ──────────────────────────────────────────
    # cute position 0 (size 16) divides across the new size-4 boundary: the
    # outer sub-factor (4) is exactly the mesh extent, so Split relocates
    # there (local extent 1); the inner residual (4) merges with position 1
    # into the trailing axis.
    TypeInferCase(
        "split_divides_carries",
        _reshape((4, 32)),
        (sharded((16, 8), (Split(0),), _M),),
        sharded((4, 32), (Split(0),), _M, cute=(4, 4, 8), strides=(32, 8, 1)),
    ),
    # a flat split dim (4096) splits into (32, 128): the outer sub-factor (32)
    # is divisible by the mesh extent (4) but exceeds it, so it factors
    # further into (4, Split-bound, local extent 1) and (8, plain); the
    # inner residual (128) merges in as its own plain position.
    TypeInferCase(
        "flat_split_divides_carries",
        _reshape((32, 128)),
        (sharded((4096,), (Split(0),), _M),),
        sharded(
            (32, 128), (Split(0),), _M,
            cute=(4, 8, 128), strides=(1024, 128, 1),
        ),
    ),
    # ── misaligned sharded fails closed ───────────────────────────────────────
    # cute position 0 (size 6) would divide across the new size-3 boundary,
    # but the mesh extent (2) does not divide the outer sub-factor (3) -> the
    # split genuinely straddles a device boundary and stays rejected.
    TypeInferCase(
        "straddle_fails_closed",
        _reshape((3, 8)),
        (sharded((6, 4), (Split(0),), mesh((2,))),),
        ExpectedError(match="align"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reshape_typeinfer(case):
    run_typeinfer_case(case)


def test_reshape_then_reshard_rmem_no_split_aliasing():
    """A `Split`-bound position that must subdivide across a new-axis
    boundary keeps local extent 1 through `Reshape` (`docs/spec/shard.md`
    §7.1.1), so a follow-on `Reshard(rmem)` — which assigns stride 0 to
    every `Split`-bound cute dim — never aliases distinct per-device
    coordinates onto one physical slot."""
    reshaped = infer_call(_reshape((32, 128)), sharded((4096,), (Split(0),), _M))
    sl = reshaped.layout
    resharded = infer_call(
        Reshard(
            layout=ShardLayout(
                layout=Layout(shape=sl.layout.shape, strides=None),
                attrs=sl.attrs,
                mesh=sl.mesh,
            ),
            storage=StorageKind.RMEM,
        ),
        reshaped,
    )
    local = shard_layout_local_shape(resharded.layout)
    strides = resharded.layout.layout.strides
    aliased = [
        i for i, (extent, stride) in enumerate(zip(local, strides))
        if stride == 0 and extent > 1
    ]
    assert not aliased, (
        f"stride-0 axes with local extent > 1: {aliased} "
        f"(local={local}, strides={strides})"
    )


# A symbolic target axis (op metadata, not input data) inferred from the input.
_S = DimVar(name="seq_len", lo=1, hi=4096)


@pytest.mark.parametrize(
    "in_shape,new_shape,out_shape",
    [
        ((2, 3), (6,), (6,)),
        # A symbolic target axis is inferred from the concrete input.
        ((1, 6, 8), (1, _S, 2, 4), (1, 6, 2, 4)),
    ],
    ids=["flatten", "dynamic_axis_inferred"],
)
def test_reshape_evaluate(in_shape, new_shape, out_shape):
    torch.manual_seed(0)
    x = torch.randn(*in_shape)
    run_eval_case(EvalCase("", Reshape(new_shape=new_shape), (x,), x.reshape(*out_shape)))
