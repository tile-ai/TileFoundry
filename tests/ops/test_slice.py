"""Slice's sharded-layout preservation, rejection boundary, and moved windows.

Windows on unsplit logical axes retain distribution. Narrowing a split axis is
rejected because the window need not align with that mesh division. A window
moved by a compile-time offset reads a fused ``[gate | up]`` tensor on GPU.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tilefoundry.analysis.preflight import validate_authored
from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.ir.core import Call, Constant, Tuple, TypeInferContext, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.tensor.slice import Slice, slice_size
from tilefoundry.ir.types import DType, TupleType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.dim import DimMul, DimVar, simplify_dim
from tilefoundry.ir.types.dim_isl import normalize_dim
from tilefoundry.ir.types.shard import ComposedLayout, Layout, make_mesh
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, Split, shard_layout_of
from tilefoundry.visitor_registry.contexts import CostContext, TrafficBytes
from tilefoundry.visitor_registry.visitors import CostEvaluator

_F = DType.f32
_M = make_mesh((4,))


def _slice_call(source, starts, sizes, strides, *, source_expr=None):
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
    if source_expr is None:
        source_expr = Var(type=source, name="source")
    call = Call(
        type=source,
        target=Slice(sizes=sizes, strides=strides),
        args=(source_expr, starts_expr),
    )
    return replace(call, type=TypeInferContext().type_of(call))


def _slice_type(source, starts, sizes, strides):
    return _slice_call(source, starts, sizes, strides).type


def _windowed_shard(source, shape, strides) -> ShardLayout:
    return ShardLayout(
        layout=Layout(shape=shape, strides=strides),
        attrs=source.layout.attrs,
        mesh=source.layout.mesh,
    )


def test_reversed_static_window_is_an_empty_slice() -> None:
    scalar = make_tensor_type((), DType.i64)
    size = normalize_dim(
        slice_size(
            Constant(type=scalar, value=8),
            Constant(type=scalar, value=4),
            Constant(type=scalar, value=1),
        )
    )

    actual = _slice_type(make_tensor_type((8, 4), _F), (8, 0), (size, 4), (1, 1))

    assert actual.shape == (0, 4)


def test_slice_of_unbound_axis_preserves_the_shard_layout():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),))
    actual = _slice_type(
        source,
        (0, 0),
        (16, 16),
        (1, 1),
    )

    assert actual.layout == _windowed_shard(source, (4, 4, 16), (128, 32, 1))


def test_slice_step_scales_the_unbound_layout_stride():
    source = make_shard_tensor_type((16, 32), mesh=_M, attrs=(Split(0),))

    actual = _slice_type(source, (0, 0), (16, 8), (1, 2))

    assert actual.layout == _windowed_shard(source, (4, 4, 8), (128, 32, 2))


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

    assert isinstance(actual.layout, ShardLayout)
    stride = actual.layout.layout.strides[0]
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


def test_fused_gqa_qkv_slices_keep_distribution_visible_to_consumers():
    """32 Q / 8 KV heads use group slices 4/1/1 and retain HKV sharding."""
    mesh = make_mesh((8,))
    source = make_tensor_type(
        (64, 8, 6, 16),
        _F,
        storage="smem",
        layout=ShardLayout(
            layout=Layout((64, 8, 6, 16), (768, 96, 16, 1)),
            attrs=(Split(1),),
            mesh=mesh,
        ),
    )
    source_expr = Var(type=source, name="source")
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

    assert isinstance(q.type.layout, ShardLayout)
    assert isinstance(k.type.layout, ComposedLayout) and k.type.layout.offset == 64
    assert isinstance(v.type.layout, ComposedLayout) and v.type.layout.offset == 80
    assert shard_layout_of(k.type.layout) is not None
    assert shard_layout_of(v.type.layout) is not None

    add = Binary(kind=BinaryKind.ADD)
    q_used = Call(type=q.type, target=add, args=(q, q))
    q_used = replace(q_used, type=TypeInferContext().type_of(q_used))
    kv_used = Call(type=k.type, target=add, args=(k, v))
    kv_used = replace(kv_used, type=TypeInferContext().type_of(kv_used))

    assert isinstance(q_used.type.layout, ShardLayout)
    assert isinstance(kv_used.type.layout, ShardLayout)
    validate_authored(
        (
            Function.build(
                name="consume_fused_qkv_views",
                params=(source_expr,),
                body=kv_used,
                return_type=kv_used.type,
            ),
        )
    )


def test_slice_cost_charges_coordinates_but_not_the_view():
    call = _slice_call(
        make_tensor_type((64, 8, 6, 16), _F),
        (0, 0, 4, 0),
        (64, 8, 1, 16),
        (1, 1, 1, 1),
    )

    cost = CostEvaluator(CostContext()).visit(call)

    assert cost.flops == {}
    assert cost.traffic == (
        TrafficBytes(),
        TrafficBytes(read=4 * 8),
        TrafficBytes(),
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


import torch  # noqa: E402

from tilefoundry import func, module  # noqa: E402
from tilefoundry.dsl import Mesh, Tensor, Topology  # noqa: E402
from tilefoundry.dsl.tf import *  # noqa: E402,F401,F403
from tilefoundry.evaluator import evaluate  # noqa: E402
from tilefoundry.target import CudaTarget  # noqa: E402

_HALF, _COLS, _STEP = 4, 4, 2


@module(entry="moved_copy", topologies=(Topology("thread", 1),))
class _MovedWindow:
    """Read the far half of a fused tensor through a window moved by a constant."""

    @func
    def moved_copy(gu: Tensor[(2 * _HALF, _COLS), "f32"]):
        with Mesh(("thread",), (1,), ("t",)) as m:
            gr = reshard(gu, (2 * _HALF, _COLS @ m.t), "rmem")
            acc = full_like(gr, 0.0)
            for r in tile(_HALF, _STEP):
                acc = insert_slice(acc, gr[r + _HALF, :], (r, 0))
            return reshard(acc, (2 * _HALF, _COLS @ m.t), "gmem")


def _moved_reference(gu):
    return torch.cat([gu[_HALF:, :], torch.zeros_like(gu[_HALF:, :])])


def test_a_moved_window_reads_the_far_half_of_one_tensor():
    gu = torch.arange(2 * _HALF * _COLS, dtype=torch.float32).reshape(2 * _HALF, _COLS)

    actual = evaluate(_MovedWindow.lookup("moved_copy"), gu, device="cpu")

    torch.testing.assert_close(actual, _moved_reference(gu))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moved_window_gpu_oracle():
    """The offset reaches the emitted address, not just the evaluator's arithmetic."""
    import tilefoundry  # noqa: PLC0415

    rm = tilefoundry.compile(_MovedWindow, target=CudaTarget("nvidia.h200_sxm"))
    gu = torch.randn(2 * _HALF, _COLS, device="cuda")
    out = torch.zeros(2 * _HALF, _COLS, device="cuda")
    rm(gu, out)
    torch.cuda.synchronize()

    expected = _moved_reference(gu)
    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-4), (out - expected).abs().max()
