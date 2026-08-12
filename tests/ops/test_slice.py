"""Slice's sharded-layout boundary.

Slice's sharded-layout boundary: the sliced shape can no longer be described
by the input's layout, so a genuinely-sharded input drops to an unsharded output
rather than carrying a fake layout forward.
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, Constant, Tuple, TypeInferContext, Var
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import DType, TupleType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import ComposedLayout, Layout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split

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


def test_slice_of_sharded_input_drops_the_layout():
    actual = _slice_type(
        make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),)),
        (0, 0),
        (16, 16),
        (1, 1),
    )
    assert actual == make_tensor_type((16, 16), _F)


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
