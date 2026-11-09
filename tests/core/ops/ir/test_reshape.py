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

import pytest
import torch

from tests.core.ops.ir.op_case import (
    ExpectedError,
    OpCase,
    infer_call,
    op_case_test,
    split_local_extents,
)
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import sharding, tensor

_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))


def _reshape(new_shape):
    return tensor.Reshape(new_shape=new_shape)


def _split_mesh_axes(ty) -> set:
    """Mesh axes carrying a genuine `Split` in *ty*'s output layout.

    Mesh axes carrying a genuine `Split` in *ty*'s output layout — the
    public "did the sharding survive" signal, independent of which layout
    position a `Split` happens to reference internally.
    """
    return {i for i, a in enumerate(ty.layout.attrs) if isinstance(a, types.Split)}


def _partial_reductions(ty) -> dict:
    """Partial reductions.

    Mesh axes carrying a `Partial` in *ty*'s output layout, keyed by mesh
    axis and valued by reduction op.
    """
    return {i: a.reduction for i, a in enumerate(ty.layout.attrs) if isinstance(a, types.Partial)}


@pytest.mark.parametrize(
    ("shape", "strides", "new_shape", "expected"),
    (
        ((16, 8), (8, 1), (8, 16), (16, 1)),
        ((1, 64, 1, 576), (65536, 576, 576, 1), (64, 576), (576, 1)),
    ),
    ids=("compact", "window-unit-axes"),
)
def test_view_reshape_derives_c_order(
    shape, strides, new_shape, expected
):
    source = types.TensorType.from_shape(shape, layout=types.Layout(shape=shape, strides=strides))
    ty = infer_call(_reshape(new_shape), source)
    assert ty.layout == types.Layout(shape=new_shape, strides=expected)
    assert infer_call(_reshape(new_shape), types.TensorType.from_shape(shape)).layout == types.Layout(
        shape=new_shape, strides=expected
    )


def test_noncontiguous_reshape_uses_default_strides():
    source = types.TensorType.from_shape((8, 16), layout=types.Layout(shape=(8, 16), strides=(1, 8)))
    assert infer_call(_reshape((128,)), source).layout == types.Layout(shape=(128,), strides=(1,))


def _make_unmaterialized_sharding_is_not_dropped_by_scalar_reshape():
    source = utils.make_shard_tensor_type((4, 8), storage=expr.StorageKind.UMAT, mesh=_M, attrs=(types.Split(0),))
    return OpCase(
        "umat_sharding_scalar_reshape", _reshape(()), (source,), error=ExpectedError(match="align")
    )


def test_reshape_merge_keeps_split():
    """Merge: layout (16, 8) -> (128,); the Split-bound mesh axis survives."""
    ty = infer_call(_reshape((128,)), utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Split(0),)))
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
        utils.make_shard_tensor_type(
            (32, 128),
            mesh=types.Mesh((types.Topology("gpu", 4),), types.Layout((2, 2), (2, 1)), ("a", "b")),
            attrs=(types.Split(0), types.Partial("sum")),
        ),
    )
    assert tuple(ty.shape) == (1, 32, 128)
    assert _split_mesh_axes(ty) == {0}
    assert _partial_reductions(ty) == {1: "sum"}


def test_reshape_split_divides_shard():
    """Layout position 0 (size 16) divides across the new size-4 boundary.

    Layout position 0 (size 16) divides across the new size-4 boundary: the
    outer sub-factor (4) is exactly the mesh extent, so the Split-bound mesh
    axis survives with local extent 1 ([shard §7.1.1](docs/spec/shard.md#711-layoutshape)).
    """
    ty = infer_call(_reshape((4, 32)), utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Split(0),)))
    assert tuple(ty.shape) == (4, 32)
    assert _split_mesh_axes(ty) == {0}
    assert split_local_extents(ty) == [1]


def test_reshape_reshard_keeps_split_unaliased():
    """A flat split dim (4096) splits into (32, 128).

    A flat split dim (4096) splits into (32, 128): the outer sub-factor (32)
    is divisible by the mesh extent (4) but exceeds it, so the `Split`-bound mesh
    axis must still keep local extent 1
    ([shard §7.1.1](docs/spec/shard.md#711-layoutshape)) after further
    factorization. A follow-on `Reshard(rmem)` assigns stride 0 to split-bound
    layout dims, making a lost local extent observable as physical aliasing.
    """
    reshaped = infer_call(
        _reshape((32, 128)), utils.make_shard_tensor_type((4096,), mesh=_M, attrs=(types.Split(0),))
    )
    assert _split_mesh_axes(reshaped) == {0}
    assert split_local_extents(reshaped) == [1]
    sl = reshaped.layout
    resharded = infer_call(
        sharding.Reshard(
            layout=types.ShardLayout(
                layout=types.Layout(shape=sl.layout.shape, strides=None), attrs=sl.attrs, mesh=sl.mesh
            ),
            storage=expr.StorageKind.RMEM,
        ),
        reshaped,
    )
    local = types.shard_layout_local_shape(resharded.layout)
    strides = resharded.layout.layout.strides
    aliased = [
        i for i, (extent, stride) in enumerate(zip(local, strides)) if stride == 0 and extent > 1
    ]
    assert not aliased, (
        f"stride-0 axes with local extent > 1: {aliased} (local={local}, strides={strides})"
    )


def test_broadcast_reshape_reshard_commute():
    source = types.TensorType.from_shape(
        (1, 32, 128), layout=types.Layout(shape=(1, 32, 128), strides=(4096, 128, 1))
    )
    packed_layout = types.ShardLayout(
        layout=types.Layout(shape=(1, 32, 128), strides=None), attrs=(types.Broadcast(),), mesh=_M
    )
    reshaped_layout = types.ShardLayout(
        layout=types.Layout(shape=(32, 128), strides=None), attrs=(types.Broadcast(),), mesh=_M
    )
    resharded_then_reshaped = infer_call(
        _reshape((32, 128)),
        infer_call(sharding.Reshard(layout=packed_layout, storage=expr.StorageKind.SMEM), source),
    )
    reshaped_then_resharded = infer_call(
        sharding.Reshard(layout=reshaped_layout, storage=expr.StorageKind.SMEM),
        infer_call(_reshape((32, 128)), source),
    )
    assert resharded_then_reshaped == reshaped_then_resharded


_S = expr.DimVar(name="seq_len", lo=1, hi=4095)


def _make_reshape_evaluate_dynamic_axis_inferred():
    torch.manual_seed(0)
    x = torch.randn(1, 6, 8)
    return OpCase(
        "dynamic_axis_inferred", tensor.Reshape(new_shape=(1, _S, 2, 4)), (x,), value=x.reshape(1, 6, 2, 4)
    )


CASES = (
    OpCase(
        "straddle_fails_closed",
        _reshape((3, 8)),
        (
            utils.make_shard_tensor_type(
                (6, 4),
                mesh=types.Mesh((types.Topology("gpu", 2),), types.Layout((2,), (1,)), ("g",)),
                attrs=(types.Split(0),),
            ),
        ),
        error=ExpectedError(match="align"),
    ),
    _make_unmaterialized_sharding_is_not_dropped_by_scalar_reshape(),
    _make_reshape_evaluate_dynamic_axis_inferred(),
)
test_op_case = op_case_test(CASES)
