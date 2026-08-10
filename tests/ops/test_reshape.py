"""Reshape's sharded carries and the aliasing they must not create.

A genuine sharding carries when every layout position lies entirely within one
new axis, or when a ``Split``-bound position divides across a new-axis boundary
at a point its bound mesh extent evenly divides (``Split`` relocates to the
mesh-extent-sized sub-position, keeping local extent 1, with any remainder
carried forward as a plain layout position); a reshape that cannot be expressed
either way fails closed (no fake layout). See
[hir §1.3](docs/spec/hir.md#13-op) ``Reshape``.
"""

from __future__ import annotations

import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
    split_local_extents,
)
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.types import make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Layout, ShardLayout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Partial,
    Split,
    shard_layout_local_shape,
)
from tilefoundry.ir.types.storage import StorageKind

_M = make_mesh((4,))


def _reshape(new_shape):
    return Reshape(new_shape=new_shape)


def _split_mesh_axes(ty) -> set:
    """Mesh axes carrying a genuine `Split` in *ty*'s output layout.

    Mesh axes carrying a genuine `Split` in *ty*'s output layout — the
    public "did the sharding survive" signal, independent of which layout
    position a `Split` happens to reference internally.
    """
    return {i for i, a in enumerate(ty.layout.attrs) if isinstance(a, Split)}


def _partial_reductions(ty) -> dict:
    """Partial reductions.

    Mesh axes carrying a `Partial` in *ty*'s output layout, keyed by mesh
    axis and valued by reduction op.
    """
    return {i: a.reduction for i, a in enumerate(ty.layout.attrs) if isinstance(a, Partial)}


def test_plain_c_order_layout_is_derived_when_reshape_is_a_view():
    source = make_tensor_type((16, 8), layout=Layout(shape=(16, 8), strides=(8, 1)))
    ty = infer_call(_reshape((8, 16)), source)

    assert ty.layout == Layout(shape=(8, 16), strides=(16, 1))
    assert infer_call(_reshape((8, 16)), make_tensor_type((16, 8))).layout is None


def test_noncontiguous_plain_layout_is_not_claimed_as_a_reshape_view():
    source = make_tensor_type((8, 16), layout=Layout(shape=(8, 16), strides=(1, 8)))

    assert infer_call(_reshape((128,)), source).layout is None


def test_straddling_split_fails_closed():
    """Test straddling split fails closed.

    Layout position 0 (size 6) would divide across the new size-3 boundary,
    but the mesh extent (2) does not divide the outer sub-factor (3) -> the
    split genuinely straddles a device boundary and stays rejected.
    """
    run_typeinfer_case(
        TypeInferCase(
            "straddle_fails_closed",
            _reshape((3, 8)),
            (make_shard_tensor_type((6, 4), mesh=make_mesh((2,)), attrs=(Split(0),)),),
            ExpectedError(match="align"),
        )
    )


def test_merge_carries():
    """Merge: layout (16, 8) -> (128,); the Split-bound mesh axis survives."""
    ty = infer_call(_reshape((128,)), make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(0),)))
    assert tuple(ty.shape) == (128,)
    assert _split_mesh_axes(ty) == {0}


def test_split_remaps_partial_carries():
    """On a two-axis mesh, the `Split` mesh axis survives the reshape while the `Partial` mesh axis.

    On a two-axis mesh, the `Split` mesh axis survives the reshape while
    the `Partial` mesh axis — a value state with no layout axis of its own —
    carries through unchanged.
    """
    ty = infer_call(
        _reshape((1, 32, 128)),
        make_shard_tensor_type((32, 128), mesh=make_mesh((2, 2)), attrs=(Split(0), Partial("sum"))),
    )
    assert tuple(ty.shape) == (1, 32, 128)
    assert _split_mesh_axes(ty) == {0}
    assert _partial_reductions(ty) == {1: "sum"}


def test_split_divides_carries():
    """Layout position 0 (size 16) divides across the new size-4 boundary.

    Layout position 0 (size 16) divides across the new size-4 boundary: the
    outer sub-factor (4) is exactly the mesh extent, so the Split-bound mesh
    axis survives with local extent 1 ([shard §7.1.1](docs/spec/shard.md#711-layoutshape)).
    """
    ty = infer_call(_reshape((4, 32)), make_shard_tensor_type((16, 8), mesh=_M, attrs=(Split(0),)))
    assert tuple(ty.shape) == (4, 32)
    assert _split_mesh_axes(ty) == {0}
    assert split_local_extents(ty) == [1]


def test_reshape_then_reshard_rmem_no_split_aliasing():
    """A flat split dim (4096) splits into (32, 128).

    A flat split dim (4096) splits into (32, 128): the outer sub-factor (32)
    is divisible by the mesh extent (4) but exceeds it, so the `Split`-bound mesh
    axis must still keep local extent 1
    ([shard §7.1.1](docs/spec/shard.md#711-layoutshape)) after further
    factorization. A follow-on `Reshard(rmem)` assigns stride 0 to split-bound
    layout dims, making a lost local extent observable as physical aliasing.
    """
    reshaped = infer_call(
        _reshape((32, 128)), make_shard_tensor_type((4096,), mesh=_M, attrs=(Split(0),))
    )
    assert _split_mesh_axes(reshaped) == {0}
    assert split_local_extents(reshaped) == [1]
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
        i for i, (extent, stride) in enumerate(zip(local, strides)) if stride == 0 and extent > 1
    ]
    assert not aliased, (
        f"stride-0 axes with local extent > 1: {aliased} (local={local}, strides={strides})"
    )


_S = DimVar(name="seq_len", lo=1, hi=4096)


def test_reshape_evaluate_dynamic_axis_inferred():
    torch.manual_seed(0)
    x = torch.randn(1, 6, 8)
    run_eval_case(EvalCase("", Reshape(new_shape=(1, _S, 2, 4)), (x,), x.reshape(1, 6, 2, 4)))
