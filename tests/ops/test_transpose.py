"""Transpose typeinfer over sharded (factorized) layouts.

When a tensor axis is split, its ``Layout`` carries more positions than
the tensor has axes (the split axis factorizes into mesh-extent × per-shard
sub-axes). Transposing must reorder the layout positions by their owning tensor
axis — keeping each tensor axis's sub-axes together — and remap the
``Split`` / ``Partial`` references to the moved layout positions, rather than
indexing layout positions with the tensor-axis permutation directly.
"""

from __future__ import annotations

from tests.ops.typeinfer_utils import (
    infer_call,
    raw_shard_tensor_type,
)
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import Layout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    ShardLayout,
    Split,
    shard_layout_local_shape,
)

_M = make_mesh((1, 128, 8, 32), ("cluster", "cta", "warp", "lane"))
_B4 = (Broadcast(), Broadcast(), Broadcast(), Broadcast())
_T10 = Transpose(perm=(1, 0))


def test_plain_input_permutes_its_layout_when_one_is_stated():
    source = make_tensor_type((16, 8), DType.bf16, layout=Layout(shape=(16, 8), strides=(8, 1)))
    ty = infer_call(_T10, source)

    assert ty.layout == Layout(shape=(8, 16), strides=(1, 8))
    assert infer_call(_T10, make_tensor_type((16, 8), DType.bf16)).layout is None


def test_factorized_split_reorders_subaxes():
    """Tensor (4096, 2048), axis 0 split on cta -> layout (128, 32, 2048).

    Tensor (4096, 2048), axis 0 split on cta -> layout (128, 32, 2048). The
    transpose moves tensor axis 1 (layout pos 2) first; axis 0's
    sub-positions (layout pos 0, 1) follow in order; the Split moves from
    layout pos 0 to pos 1.
    """
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
    """Implicit strides: shape + attrs permute, output keeps implicit strides.

    Implicit (None) strides: shape + attrs permute, output keeps implicit
    strides (regression: no None-stride indexing crash).
    """
    x_ty = raw_shard_tensor_type(
        (16, 8),
        (16, 8),
        None,
        (Split(0), *_B4[1:]),
        _M,
        dtype=DType.bf16,
    )
    ty = infer_call(_T10, x_ty)
    assert tuple(ty.shape) == (8, 16)
    assert isinstance(ty.layout, ShardLayout)
    assert ty.layout.attrs == (Split(1), *_B4[1:])
