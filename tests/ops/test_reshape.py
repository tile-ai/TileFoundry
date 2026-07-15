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


def _split_mesh_axes(ty) -> set:
    """Mesh axes carrying a genuine `Split` in *ty*'s output layout — the
    public "did the sharding survive" signal, independent of which cute
    position a `Split` happens to reference internally."""
    return {i for i, a in enumerate(ty.layout.attrs) if isinstance(a, Split)}


def _partial_reductions(ty) -> dict:
    """Mesh axes carrying a `Partial` in *ty*'s output layout, keyed by mesh
    axis and valued by reduction op."""
    return {i: a.reduction for i, a in enumerate(ty.layout.attrs) if isinstance(a, Partial)}


def _split_local_extents(ty) -> list:
    """`shard_layout_local_shape` at every `Split`-bound cute dim of *ty*'s
    output layout — every entry MUST be 1 (`docs/spec/shard.md` §7.1.1)."""
    local = shard_layout_local_shape(ty.layout)
    return [local[a.axis] for a in ty.layout.attrs if isinstance(a, Split)]


CASES = [
    # ── unsharded ────────────────────────────────────────────────────────────
    TypeInferCase("unsharded", _reshape((32,)), (ten((4, 8), _F),), ten((32,), _F)),
    # drop a leading unit axis: mesh axis 0 stays genuinely split.
    TypeInferCase(
        "remove_leading_unit",
        _reshape((32, 128)),
        (sharded((1, 32, 128), (Split(1),), _M),),
        sharded((32, 128), (Split(0),), _M),
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


# ── sharded carries ───────────────────────────────────────────────────────
# Each case checks output shape and which mesh axes stay `Split` / `Partial`,
# not the internal cute factorization a valid `Reshape` might produce.


def test_merge_carries():
    """Merge: cute (16, 8) -> (128,); the Split-bound mesh axis survives."""
    ty = infer_call(_reshape((128,)), sharded((16, 8), (Split(0),), _M))
    assert tuple(ty.shape) == (128,)
    assert _split_mesh_axes(ty) == {0}


def test_insert_unit_axis():
    """Inserting a unit axis after the split axis leaves the sharding intact."""
    ty = infer_call(_reshape((32, 1, 128)), sharded((32, 128), (Split(0),), _M))
    assert tuple(ty.shape) == (32, 1, 128)
    assert _split_mesh_axes(ty) == {0}


def test_reshuffle_leading_unit():
    """Inserting a leading unit axis still leaves the mesh axis genuinely
    split, even though its bound cute position shifts internally."""
    ty = infer_call(_reshape((1, 32, 128)), sharded((32, 128), (Split(0),), _M))
    assert tuple(ty.shape) == (1, 32, 128)
    assert _split_mesh_axes(ty) == {0}


def test_partial_carries():
    """A `Partial` is a mesh-axis value state with no cute axis; it carries
    through the reshape unchanged."""
    ty = infer_call(_reshape((32, 1, 128)), sharded((32, 128), (Partial("sum"),), _M))
    assert tuple(ty.shape) == (32, 1, 128)
    assert _partial_reductions(ty) == {0: "sum"}


def test_split_remaps_partial_carries():
    """On a two-axis mesh, the `Split` mesh axis survives the reshape while
    the `Partial` mesh axis carries through unchanged."""
    ty = infer_call(
        _reshape((1, 32, 128)),
        sharded((32, 128), (Split(0), Partial("sum")), mesh((2, 2))),
    )
    assert tuple(ty.shape) == (1, 32, 128)
    assert _split_mesh_axes(ty) == {0}
    assert _partial_reductions(ty) == {1: "sum"}


def test_split_divides_carries():
    """cute position 0 (size 16) divides across the new size-4 boundary: the
    outer sub-factor (4) is exactly the mesh extent, so the Split-bound mesh
    axis survives with local extent 1 (`docs/spec/shard.md` §7.1.1)."""
    ty = infer_call(_reshape((4, 32)), sharded((16, 8), (Split(0),), _M))
    assert tuple(ty.shape) == (4, 32)
    assert _split_mesh_axes(ty) == {0}
    assert _split_local_extents(ty) == [1]


def test_flat_split_divides_carries():
    """A flat split dim (4096) splits into (32, 128): the outer sub-factor
    (32) is divisible by the mesh extent (4) but exceeds it, so the
    Split-bound mesh axis still keeps local extent 1 (`docs/spec/shard.md`
    §7.1.1) after the further factorization."""
    ty = infer_call(_reshape((32, 128)), sharded((4096,), (Split(0),), _M))
    assert tuple(ty.shape) == (32, 128)
    assert _split_mesh_axes(ty) == {0}
    assert _split_local_extents(ty) == [1]


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
