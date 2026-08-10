"""Slice's sharded-layout boundary.

Slice's sharded-layout boundary: the sliced shape can no longer be described
by the input's layout, so a genuinely-sharded input drops to an unsharded output
rather than carrying a fake layout forward.
"""

from __future__ import annotations

from tests.ops.typeinfer_utils import (
    TypeInferCase,
    infer_call,
    run_typeinfer_case,
)
from tilefoundry.ir.core import Var
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import ComposedLayout, Layout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split

_F = DType.f32
_M = make_mesh((4,))


def test_slice_of_sharded_input_drops_the_layout():
    run_typeinfer_case(
        TypeInferCase(
            "sharded_drops_layout",
            Slice(begin=(0, 0), end=(16, 16), strides=(1, 1)),
            (make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),)),),
            make_tensor_type((16, 16), _F),
        )
    )


def test_plain_row_and_column_slices_derive_subbox_layouts():
    source = make_tensor_type(
        (1024, 2048),
        _F,
        layout=Layout(shape=(1024, 2048), strides=(2048, 1)),
    )

    row = infer_call(Slice(begin=(0, 0), end=(256, 2048), strides=(1, 1)), source)
    column = infer_call(Slice(begin=(0, 0), end=(1024, 512), strides=(1, 1)), source)

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


def test_runtime_bound_slice_does_not_claim_a_static_layout():
    end = Var(type=make_tensor_type((), DType.i64), name="end")

    sliced = infer_call(
        Slice(begin=(0, 0), end=(end, 2048), strides=(1, 1)),
        make_tensor_type(
            (1024, 2048),
            _F,
            layout=Layout(shape=(1024, 2048), strides=(2048, 1)),
        ),
    )

    assert sliced.layout is None
    assert (
        infer_call(
            Slice(begin=(0, 0), end=(256, 2048), strides=(1, 1)),
            make_tensor_type((1024, 2048), _F),
        ).layout
        is None
    )
