"""Slice's sharded-layout preservation, rejection boundary, and moved windows.

Windows on unsplit logical axes retain distribution. Narrowing a split axis is
rejected because the window need not align with that mesh division. A window
moved by a compile-time offset reads a fused ``[gate | up]`` tensor on GPU.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tilefoundry import cli, services
from tilefoundry.dsl import *
from tilefoundry.interfaces import cost, evaluator, typeinfer
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import math, tensor

_F = DType.f32
_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))


def _slice_call(source, starts, sizes, strides, *, source_expr=None):
    start_exprs = tuple(
        start
        if isinstance(start, expr.Var)
        else expr.Constant(type=TensorType.from_shape((), DType.i64), value=start)
        for start in starts
    )
    starts_expr = expr.Tuple(
        type=TupleType(fields=tuple(start.type for start in start_exprs)),
        elements=start_exprs,
    )
    if source_expr is None:
        source_expr = expr.Var(type=source, name="source")
    call = expr.Call(
        type=source,
        target=tensor.Slice(sizes=sizes, strides=strides),
        args=(source_expr, starts_expr),
    )
    return replace(call, type=typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext()))


def _slice_type(source, starts, sizes, strides):
    return _slice_call(source, starts, sizes, strides).type


def _windowed_shard(source, shape, strides) -> types.ShardLayout:
    return types.ShardLayout(
        layout=Layout(shape=shape, strides=strides),
        attrs=source.layout.attrs,
        mesh=source.layout.mesh,
    )


def test_reversed_window_slices_empty() -> None:
    scalar = TensorType.from_shape((), DType.i64)
    size = services.normalize_dim(
        tensor.slice_size(
            expr.Constant(type=scalar, value=8),
            expr.Constant(type=scalar, value=4),
            expr.Constant(type=scalar, value=1),
        )
    )

    actual = _slice_type(TensorType.from_shape((8, 4), _F), (8, 0), (size, 4), (1, 1))

    assert actual.shape == (0, 4)


def test_slice_unbound_axis_keeps_shard():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(types.Split(0),))
    actual = _slice_type(
        source,
        (0, 0),
        (16, 16),
        (1, 1),
    )

    assert actual.layout == _windowed_shard(source, (4, 4, 16), (128, 32, 1))


def test_slice_step_scales_stride():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(types.Split(0),))

    actual = _slice_type(source, (0, 0), (16, 8), (1, 2))

    assert actual.layout == _windowed_shard(source, (4, 4, 8), (128, 32, 2))


def test_slice_step_scales_symbolic_stride():
    stride_dim = DimVar("slice_stride", 1, 64)
    source = TensorType.from_shape(
        (16, 32),
        _F,
        layout=types.ShardLayout(
            layout=Layout(
                shape=(16, 4, 8),
                strides=(services.simplify_dim("mul", (stride_dim, 2)), 8, 1),
            ),
            attrs=(types.Split(1),),
            mesh=_M,
        ),
    )

    actual = _slice_type(source, (0, 0), (8, 32), (2, 1))

    assert isinstance(actual.layout, types.ShardLayout)
    stride = actual.layout.layout.strides[0]
    assert evaluator.resolve_dim(stride, {"slice_stride": 3}) == 12


def test_slice_split_axis_rejected():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(types.Split(0),))

    with pytest.raises(
        ValueError,
        match=(
            "Slice narrows axis 0, which mesh axis 0 splits.*"
            "Slice before placing, or reshard to a layout that leaves axis 0 whole"
        ),
    ):
        _slice_type(source, (0, 0), (8, 32), (1, 1))


def test_runtime_window_keeps_distribution():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(types.Split(0),))
    start = expr.Var(type=TensorType.from_shape((), DType.i64), name="start")

    actual = _slice_type(source, (0, start), (16, 16), (1, 1))

    assert actual.layout == _windowed_shard(source, (4, 4, 16), (128, 32, 1))


def test_runtime_window_keeps_split_target():
    seq = DimVar("slice_seq", 1, 4096)
    mesh = Mesh((Topology("gpu", 16),), Layout((16,), (1,)), ("g",))
    source = make_shard_tensor_type((1, seq, 16, 128), mesh=mesh, attrs=(types.Split(2),))
    start = expr.Var(type=TensorType.from_shape((), DType.i64), name="start")

    actual = _slice_type(source, (0, start, 0, 0), (1, 128, 16, 128), (1, 1, 1, 1))

    assert actual.layout == types.ShardLayout(
        layout=Layout(shape=(1, 128, 16, 128), strides=(0, 2048, 128, 1)),
        attrs=(types.Split(2),),
        mesh=mesh,
    )


def test_qkv_slices_keep_distribution():
    """32 Q / 8 KV heads use group slices 4/1/1 and retain HKV sharding."""
    mesh = Mesh((Topology("gpu", 8),), Layout((8,), (1,)), ("g",))
    source = TensorType.from_shape(
        (64, 8, 6, 16),
        _F,
        storage="smem",
        layout=types.ShardLayout(
            layout=Layout((64, 8, 6, 16), (768, 96, 16, 1)),
            attrs=(types.Split(1),),
            mesh=mesh,
        ),
    )
    source_expr = expr.Var(type=source, name="source")
    q = _slice_call(
        source,
        (0, 0, 0, 0),
        (64, 8, 4, 16),
        (1, 1, 1, 1),
        source_expr=source_expr,
    )
    k = _slice_call(
        source,
        (0, 0, 4, 0),
        (64, 8, 1, 16),
        (1, 1, 1, 1),
        source_expr=source_expr,
    )
    v = _slice_call(
        source,
        (0, 0, 5, 0),
        (64, 8, 1, 16),
        (1, 1, 1, 1),
        source_expr=source_expr,
    )

    assert isinstance(q.type.layout, types.ShardLayout)
    assert isinstance(k.type.layout, types.ShardLayout)
    assert isinstance(v.type.layout, types.ShardLayout)
    assert types.shard_layout_of(k.type.layout) is not None
    assert types.shard_layout_of(v.type.layout) is not None

    add = math.Binary(kind=expr.BinaryKind.ADD)
    q_used = expr.Call(type=q.type, target=add, args=(q, q))
    q_used = replace(q_used, type=typeinfer.TypeInferVisitor().visit(q_used, typeinfer.TypeInferContext()))
    kv_used = expr.Call(type=k.type, target=add, args=(k, v))
    kv_used = replace(kv_used, type=typeinfer.TypeInferVisitor().visit(kv_used, typeinfer.TypeInferContext()))

    assert isinstance(q_used.type.layout, types.ShardLayout)
    assert isinstance(kv_used.type.layout, types.ShardLayout)


def test_slice_charges_only_coordinates():
    call = _slice_call(
        TensorType.from_shape((64, 8, 6, 16), _F),
        (0, 0, 4, 0),
        (64, 8, 1, 16),
        (1, 1, 1, 1),
    )

    cost_value = cost.Visitor().visit(call, cost.Context())

    assert cost_value.flops == {}
    assert cost_value.traffic == (
        cost.TrafficBytes(),
        cost.TrafficBytes(read=4 * 8),
        cost.TrafficBytes(),
    )


def test_nested_slices_keep_shard_window():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(types.Split(0),))
    first = _slice_type(source, (0, 4), (16, 16), (1, 1))

    second = _slice_type(first, (0, 2), (16, 8), (1, 1))

    assert second.layout == _windowed_shard(source, (4, 4, 8), (128, 32, 1))


def test_row_column_slices_derive_layouts():
    source = TensorType.from_shape(
        (1024, 2048),
        _F,
        layout=Layout(shape=(1024, 2048), strides=(2048, 1)),
    )

    row = _slice_type(source, (0, 0), (256, 2048), (1, 1))
    column = _slice_type(source, (0, 0), (1024, 512), (1, 1))

    assert row.layout == Layout(shape=(256, 2048), strides=(2048, 1))
    assert column.layout == Layout(shape=(1024, 512), strides=(2048, 1))


def test_runtime_start_keeps_static_strides():
    start = expr.Var(type=TensorType.from_shape((), DType.i64), name="start")

    sliced = _slice_type(
        TensorType.from_shape(
            (1024, 2048),
            _F,
            layout=Layout(shape=(1024, 2048), strides=(2048, 1)),
        ),
        (start, 0),
        (256, 2048),
        (1, 1),
    )

    assert sliced.layout == Layout(shape=(256, 2048), strides=(2048, 1))
    assert (
        _slice_type(TensorType.from_shape((1024, 2048), _F), (0, 0), (256, 2048), (1, 1)).layout
        == Layout(shape=(256, 2048), strides=(2048, 1))
    )


_HALF, _COLS = 4, 4
MOVED_WINDOW = Path(__file__).parents[2] / "fixtures" / "ops" / "moved_window.py"


def _moved_reference(gu):
    return torch.cat([gu[_HALF:, :], torch.zeros_like(gu[_HALF:, :])])


def test_moved_window_reads_far_half():
    gu = torch.arange(2 * _HALF * _COLS, dtype=torch.float32).reshape(2 * _HALF, _COLS)

    _MovedWindow = cli.load_authored_ir(str(MOVED_WINDOW))
    actual = evaluator.evaluate(_MovedWindow.lookup("moved_copy"), gu)

    torch.testing.assert_close(actual, _moved_reference(gu))
