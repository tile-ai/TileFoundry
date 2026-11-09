"""Transpose typeinfer over sharded (factorized) layouts.

When a tensor axis is split, its ``Layout`` carries more positions than
the tensor has axes (the split axis factorizes into mesh-extent × per-shard
sub-axes). Transposing must reorder the layout positions by their owning tensor
axis — keeping each tensor axis's sub-axes together — and remap the
``Split`` / ``Partial`` references to the moved layout positions, rather than
indexing layout positions with the tensor-axis permutation directly.
"""

from __future__ import annotations

from tests.core.ops.ir.op_case import (
    infer_call,
    raw_shard_tensor_type,
)
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import tensor

_M = types.Mesh(
    (types.Topology("gpu", 32768),),
    types.Layout((1, 128, 8, 32), (32768, 256, 32, 1)),
    ("cluster", "cta", "warp", "lane"),
)
_B4 = (types.Broadcast(), types.Broadcast(), types.Broadcast(), types.Broadcast())
_T10 = tensor.Transpose(perm=(1, 0))


def test_transpose_writes_compact_result():
    """Transpose writes a new C-order result rather than preserving source strides."""
    source = types.TensorType.from_shape((16, 8), expr.DType.bf16, layout=types.Layout(shape=(16, 8), strides=(8, 1)))
    ty = infer_call(_T10, source)

    assert ty.layout == types.Layout(shape=(8, 16), strides=(16, 1))
    assert infer_call(_T10, types.TensorType.from_shape((16, 8), expr.DType.bf16)).layout == ty.layout


def test_factorized_split_reorders_subaxes():
    """Reorder split subaxes into a compact new value.

    Tensor (4096, 2048), axis 0 split on cta -> layout (128, 32, 2048). The
    transpose moves tensor axis 1 (layout pos 2) first; axis 0's
    sub-positions (layout pos 0, 1) follow in order; the Split moves from
    layout pos 0 to pos 1.
    """
    x_ty = utils.make_shard_tensor_type(
        (4096, 2048),
        mesh=_M,
        attrs=(types.Broadcast(), types.Split(axis=0), types.Broadcast(), types.Broadcast()),
        dtype=expr.DType.bf16,
    )
    ty = infer_call(_T10, x_ty)
    assert tuple(ty.shape) == (2048, 4096)
    assert isinstance(ty.layout, types.ShardLayout)
    assert ty.layout.attrs == (types.Broadcast(), types.Split(axis=1), types.Broadcast(), types.Broadcast())
    assert types.shard_layout_local_shape(ty.layout)[1] == 1
    assert ty.layout.layout.strides == (4096, 32, 1)


def test_transpose_derives_fresh_strides():
    """Implicit source strides do not leave a new value's strides unspecified."""
    x_ty = raw_shard_tensor_type(
        (16, 8),
        (16, 8),
        None,
        (types.Split(0), *_B4[1:]),
        _M,
        dtype=expr.DType.bf16,
    )
    ty = infer_call(_T10, x_ty)
    assert tuple(ty.shape) == (8, 16)
    assert isinstance(ty.layout, types.ShardLayout)
    assert ty.layout.attrs == (types.Split(1), *_B4[1:])
    assert ty.layout.layout.strides == (16, 1)
