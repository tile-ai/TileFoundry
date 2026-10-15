"""Check each distinct MMA register fragment against its PTX thread table.

The six cases cover SM80 A/B/C, WGMMA C with a non-unit column repeat,
and the two WGMMA RS A layouts distinguished by element width.
See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import isl
import pytest

from tilefoundry.dsl import T
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.isl_interop import layout_to_isl_map
from tilefoundry.ir.pattern.utils import fixed_pattern_value, selected_pattern
from tilefoundry.ir.types import DType, Layout, ShardLayout, Split, TensorType
from tilefoundry.ir.types.int_tuple import flatten, product
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.typeinfer import inference_type


@pytest.mark.parametrize(
    ("atom", "role"),
    (
        pytest.param(T.cuda.sm80.Mma(), "A", id="sm80-mma-A"),
        pytest.param(T.cuda.sm80.Mma(), "B", id="sm80-mma-B"),
        pytest.param(T.cuda.sm80.Mma(), "C", id="sm80-mma-C"),
        pytest.param(
            T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.SS),
            "C",
            id="sm90-wgmma-n16-C",
        ),
        pytest.param(
            T.cuda.sm90.Wgmma(n=16, dtype="bf16", form=T.cuda.sm90.Form.RS),
            "A",
            id="sm90-wgmma-rs-bf16-A",
        ),
        pytest.param(
            T.cuda.sm90.Wgmma(n=16, dtype="fp8e4m3", form=T.cuda.sm90.Form.RS),
            "A",
            id="sm90-wgmma-rs-fp8e4m3-A",
        ),
    ),
)
def test_mma_fragment_layout(atom, role: str) -> None:
    """Every thread owns exactly its PTX elements and the declared local extent."""
    layout = _bound_fragment(atom, role)
    shape = atom.operand_shapes()[("C", "A", "B").index(role)]
    _check_split_extents_match_mesh(layout)
    sm80 = isinstance(atom, T.cuda.sm80.Mma)
    dtype = DType.f32 if role == "C" else DType.bf16 if sm80 else atom.bindings["dtype"]
    _assert_reshard_typeinfer_ok(shape, dtype.name, layout)
    for thread in range(32 if sm80 else 128):
        if sm80:
            g, q = divmod(thread, 4)
            columns = (2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9)
            expected = (
                {(r, c) for r in (g, g + 8) for c in columns}
                if role == "A"
                else {(k, g) for k in columns}
                if role == "B"
                else {(r, c) for r in (g, g + 8) for c in (2 * q, 2 * q + 1)}
            )
            coordinates = (q, g)
        else:
            warp, lane = divmod(thread, 32)
            g, q = divmod(lane, 4)
            rows = (16 * warp + g, 16 * warp + g + 8)
            width = 4 // (dtype.bit_width // 8)
            expected = (
                {
                    (r, 8 * block + c)
                    for r in rows
                    for block in range(shape[1] // 8)
                    for c in (2 * q, 2 * q + 1)
                }
                if role == "C"
                else {
                    (r, c)
                    for r in rows
                    for base in (width * q, width * q + 4 * width)
                    for c in range(base, base + width)
                }
            )
            coordinates = (warp, g, q)
        assert _per_thread_size(layout) == len(expected)
        assert _held_elements(layout, shape, coordinates) == expected


def _product(shape: tuple[int, ...]) -> int:
    return product(tuple(int(s) for s in shape))


def _per_thread_size(sl: ShardLayout) -> int:
    """Layout product divided by mesh size (= per-thread element count)."""
    mesh_size = _product(flatten(sl.mesh.layout.shape))
    return _product(sl.layout.shape) // mesh_size


def _check_split_extents_match_mesh(sl: ShardLayout) -> None:
    mesh_shape = flatten(sl.mesh.layout.shape)
    assert len(sl.attrs) == len(mesh_shape), (
        f"attrs len {len(sl.attrs)} != mesh rank {len(mesh_shape)}"
    )
    for mesh_i, attr in enumerate(sl.attrs):
        assert isinstance(attr, Split), f"mma fragment shard attrs must be Split, got {attr}"
        ax = attr.axis
        assert sl.layout.shape[ax] == mesh_shape[mesh_i], (
            f"split tensor axis {ax} extent {sl.layout.shape[ax]} != "
            f"mesh axis {mesh_i} extent {mesh_shape[mesh_i]}"
        )


def _assert_reshard_typeinfer_ok(
    src_shape: tuple[int, ...], src_dtype_name: str, dst_layout: ShardLayout
) -> None:
    """Assert reshard typeinfer ok.

    Run the registered Reshard typeinfer rule against a synthesised
    Call whose source is a plain global-storage tensor. Assert the
    resulting TensorType pins the requested ShardLayout while
    preserving the logical shape.
    """
    dtype = getattr(DType, src_dtype_name)
    src_ty = TensorType(shape=src_shape, dtype=dtype, layout=None, storage=StorageKind.GMEM)
    src = Var(type=src_ty, name="x")
    op = Reshard(layout=dst_layout, storage=StorageKind.RMEM)

    call = Call(type=src_ty, target=op, args=(src,))
    out_ty = inference_type(call)
    assert isinstance(out_ty, TensorType), f"expected TensorType, got {out_ty}"
    assert out_ty.layout is dst_layout, "output layout must reference the requested ShardLayout"

    assert out_ty.shape == src_shape
    assert out_ty.dtype == dtype
    assert out_ty.storage == StorageKind.RMEM


def _bound_fragment(atom, role: str) -> ShardLayout:
    pattern = selected_pattern(atom.role(role), atom.bindings).layout
    bindings = dict(atom.bindings)
    if role == "C" and "n" in bindings:
        bindings["c"] = bindings["n"] // 8
    layout = Layout(
        fixed_pattern_value(pattern.layout.shape, bindings),
        fixed_pattern_value(pattern.layout.strides, bindings),
    )
    return ShardLayout(layout, pattern.attrs, atom.required_execution_mesh)


def _held_elements(
    layout: ShardLayout, shape: tuple[int, int], coordinates: tuple[int, ...]
) -> set:
    relation = layout_to_isl_map(
        shape, layout, {}, divided=lambda layer: set(range(len(layer.attrs)))
    )
    box = isl.set(f"{{ [r, c] : 0 <= r < {shape[0]} and 0 <= c < {shape[1]} }}")
    held = relation.domain().intersect(box.align_params(relation.domain().get_space()))
    for axis, coordinate in enumerate(coordinates):
        held = held.fix_val(isl.dim_type.PARAM, axis, isl.val(coordinate))
    elements = set()
    held.foreach_point(
        lambda point: elements.add(
            tuple(
                point.get_coordinate_val(isl.dim_type.SET, axis).get_num_si() for axis in range(2)
            )
        )
    )
    return elements
