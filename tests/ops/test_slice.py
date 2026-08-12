"""Slice's sharded-layout preservation and rejection boundary.

Windows on unsplit logical axes retain distribution. Narrowing a split axis is
rejected because the window need not align with that mesh division.
"""

from __future__ import annotations

import pytest

from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.ir.core import Call, Constant, Tuple, TypeInferContext, Var
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import DType, TupleType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.dim import DimMul, DimVar, simplify_dim
from tilefoundry.ir.types.shard import ComposedLayout, Layout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, Split

_F = DType.f32
_M = make_mesh((4,))


def _slice_type(source, starts, sizes, strides):
    start_exprs = tuple(
        start
        if isinstance(start, Var)
        else Constant(type=make_tensor_type((), DType.i64), value=start)
        for start in starts
    )
    starts_expr = Tuple(
        type=TupleType(fields=tuple(start.type for start in start_exprs)),
        elements=start_exprs,
    )
    source_expr = Var(type=source, name="source")
    call = Call(
        type=source,
        target=Slice(sizes=sizes, strides=strides),
        args=(source_expr, starts_expr),
    )
    return TypeInferContext().type_of(call)


def _windowed_shard(source, shape, strides) -> ShardLayout:
    return ShardLayout(
        layout=Layout(shape=shape, strides=strides),
        attrs=source.layout.attrs,
        mesh=source.layout.mesh,
    )


def test_slice_of_unbound_axis_preserves_the_shard_layout():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),))
    actual = _slice_type(
        source,
        (0, 0),
        (16, 16),
        (1, 1),
    )

    assert actual.layout == ComposedLayout(
        inner=None,
        offset=0,
        outer=_windowed_shard(source, (4, 4, 16), (128, 32, 1)),
    )


def test_slice_step_scales_the_unbound_layout_stride():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),))

    actual = _slice_type(source, (0, 0), (16, 8), (1, 2))

    assert actual.layout == ComposedLayout(
        inner=None,
        offset=0,
        outer=_windowed_shard(source, (4, 4, 8), (128, 32, 2)),
    )


def test_slice_step_composes_with_a_symbolic_layout_stride():
    stride_dim = DimVar("slice_stride", 1, 65)
    source = make_tensor_type(
        (16, 32),
        _F,
        layout=ShardLayout(
            layout=Layout(
                shape=(16, 4, 8),
                strides=(simplify_dim(DimMul, (stride_dim, 2)), 8, 1),
            ),
            attrs=(Split(1),),
            mesh=_M,
        ),
    )

    actual = _slice_type(source, (0, 0), (8, 32), (2, 1))

    assert isinstance(actual.layout, ComposedLayout)
    stride = actual.layout.outer.layout.strides[0]
    assert resolve_dim(stride, {"slice_stride": 3}) == 12


def test_slice_of_split_axis_is_rejected():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),))

    with pytest.raises(
        ValueError,
        match=(
            "Slice narrows axis 0, which mesh axis 0 splits.*"
            "Slice before placing, or reshard to a layout that leaves axis 0 whole"
        ),
    ):
        _slice_type(source, (0, 0), (8, 32), (1, 1))


def test_runtime_window_preserves_distribution_without_claiming_an_offset():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),))
    start = Var(type=make_tensor_type((), DType.i64), name="start")

    actual = _slice_type(source, (0, start), (16, 16), (1, 1))

    assert actual.layout == _windowed_shard(source, (4, 4, 16), (128, 32, 1))


def test_runtime_window_before_a_split_axis_preserves_the_split_target():
    seq = DimVar("slice_seq", 1, 4097)
    mesh = make_mesh((16,))
    source = make_shard_tensor_type(
        (1, seq, 16, 128), mesh=mesh, attrs=(Split(2),)
    )
    start = Var(type=make_tensor_type((), DType.i64), name="start")

    actual = _slice_type(source, (0, start, 0, 0), (1, 128, 16, 128), (1, 1, 1, 1))

    assert actual.layout == ShardLayout(
        layout=Layout(shape=(1, 128, 16, 128), strides=None),
        attrs=(Split(2),),
        mesh=mesh,
    )


def test_static_slice_inherits_an_existing_sharded_view_offset():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),))
    first = _slice_type(source, (0, 4), (16, 16), (1, 1))

    second = _slice_type(first, (0, 2), (16, 8), (1, 1))

    assert second.layout == ComposedLayout(
        inner=None,
        offset=6,
        outer=_windowed_shard(source, (4, 4, 8), (128, 32, 1)),
    )


def test_plain_row_and_column_slices_derive_subbox_layouts():
    source = make_tensor_type(
        (1024, 2048),
        _F,
        layout=Layout(shape=(1024, 2048), strides=(2048, 1)),
    )

    row = _slice_type(source, (0, 0), (256, 2048), (1, 1))
    column = _slice_type(source, (0, 0), (1024, 512), (1, 1))

    assert row.layout == ComposedLayout(
        inner=None,
        offset=0,
        outer=Layout(shape=(256, 2048), strides=(2048, 1)),
    )
    assert column.layout == ComposedLayout(
        inner=None,
        offset=0,
        outer=Layout(shape=(1024, 512), strides=(2048, 1)),
    )


def test_runtime_start_slice_does_not_claim_a_static_layout():
    start = Var(type=make_tensor_type((), DType.i64), name="start")

    sliced = _slice_type(
        make_tensor_type(
            (1024, 2048),
            _F,
            layout=Layout(shape=(1024, 2048), strides=(2048, 1)),
        ),
        (start, 0),
        (256, 2048),
        (1, 1),
    )

    assert sliced.layout is None
    assert (
        _slice_type(make_tensor_type((1024, 2048), _F), (0, 0), (256, 2048), (1, 1)).layout
        is None
    )
