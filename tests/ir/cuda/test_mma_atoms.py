"""Check MMA register fragments against PTX and shared layouts against CUTLASS.

The canonical smem atoms below are literal bit-unit shapes/strides from CUTLASS
``e406c186f510a15091cce01f782020ceb7ba8eb5``, ``include/cute/atom/mma_traits_sm90_gmma.hpp:75-85``.
``include/cute/layout.hpp:1778-1796`` requires exact atom divisibility for
``tile_to_shape``, repeating MN first (LayoutLeft) or K first (LayoutRight).
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
from tilefoundry.ir.tir.cuda.nn import wgmma
from tilefoundry.ir.types import (
    ComposedLayout,
    DType,
    Layout,
    ShardLayout,
    Split,
    Swizzle,
    TensorType,
)
from tilefoundry.ir.types.int_tuple import flatten, product
from tilefoundry.ir.types.layout_correspondence import layout_correspondence
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


_GMMA_ATOMS = {
    ("MN", 0): ((128, 8), (1, 128)),
    ("MN", 1): ((256, 8), (1, 256)),
    ("MN", 2): ((512, 8), (1, 512)),
    ("MN", 3): ((1024, 8), (1, 1024)),
    ("K", 0): ((8, 128), (128, 1)),
    ("K", 1): ((8, 256), (256, 1)),
    ("K", 2): ((8, 512), (512, 1)),
    ("K", 3): ((8, 1024), (1024, 1)),
}


def _canonical_smem_cases():
    """Keep 2166 of 3168 combinations under CUTLASS atom divisibility (1002 dropped)."""
    cases = []
    for role in ("A", "B"):
        for major in ("MN", "K"):
            for bits in range(4):
                for element_bytes in ((2,) if major == "MN" else (1, 2, 4)):
                    atom_shape, _ = _GMMA_ATOMS[major, bits]
                    am, ak = atom_shape
                    if major == "MN":
                        am //= 8 * element_bytes
                    else:
                        ak //= 8 * element_bytes
                    for mn in ((64,) if role == "A" else range(8, 257, 8)):
                        for issues in (1, 2, 4):
                            bk = issues * 32 // element_bytes
                            if mn % am or bk % ak:
                                continue
                            for order in ("MN", "K"):
                                values = (role, major, bits, element_bytes, mn, bk, order)
                                cases.append(pytest.param(*values, id="-".join(map(str, values))))
    assert len(cases) == 2166
    return cases


def _swizzled_address(offset, bits):
    return offset ^ ((offset >> 3) & (((1 << bits) - 1) << 4))


def _layout_address(layout, correspondence, shape, coordinates):
    """Interpret authored row-major positions independently of pattern matching."""
    linear = coordinates[0] * shape[1] + coordinates[1]
    digits = correspondence.digits(linear)
    return sum(digit * stride for digit, stride in zip(digits, flatten(layout.strides), strict=True))


@pytest.mark.parametrize(
    ("role", "major", "bits", "element_bytes", "mn", "bk", "order"),
    _canonical_smem_cases(),
)
def test_wgmma_admits_every_canonical_smem_tiling(role, major, bits, element_bytes, mn, bk, order):
    """Each legal CUTLASS tiling agrees elementwise and admits every issue view."""
    atom_shape, atom_strides = _GMMA_ATOMS[major, bits]
    am, ak = atom_shape
    sm, sk = atom_strides
    if major == "MN":
        am //= 8 * element_bytes
        sk //= 8 * element_bytes
    else:
        ak //= 8 * element_bytes
        sm //= 8 * element_bytes
    rm, rk = mn // am, bk // ak
    atom_size = am * ak
    ms = atom_size * (rk if order == "K" else 1)
    ks = atom_size * (rm if order == "MN" else 1)
    full = Layout(((rm, am), (rk, ak)), ((ms, sm), (ks, sk)))
    if role == "B":
        full = Layout(tuple(reversed(full.shape)), tuple(reversed(full.strides)))
    shape = (mn, bk) if role == "A" else (bk, mn)
    correspondence = layout_correspondence(full.shape, shape)
    for m in range(mn):
        for k in range(bk):
            expected = (m % am) * sm + (k % ak) * sk + (m // am) * ms + (k // ak) * ks
            coordinates = (m, k) if role == "A" else (k, m)
            actual = _layout_address(full, correspondence, shape, coordinates)
            assert _swizzled_address(actual, bits) == _swizzled_address(expected, bits)

    issue_k = 32 // element_bytes
    for start in range(0, bk, issue_k):
        if major == "MN":
            view = Layout(((rm, am), (issue_k // ak, ak)), ((ms, sm), (ks, sk)))
            offset = 0
            factory = getattr(wgmma, f"_{role.lower()}_mn_major_smem_pattern")
        elif bits <= 1:
            view = Layout(((rm, am), (issue_k // ak, ak)), ((ms, sm), (ks, sk)))
            offset = 0
            factory = getattr(wgmma, f"_{role.lower()}_k_major_whole_smem_pattern")
        else:
            view = Layout(((rm, am), issue_k), ((ms, sm), 1))
            offset = start % ak
            factory = getattr(wgmma, f"_{role.lower()}_k_major_sliced_smem_pattern")
        if role == "B":
            view = Layout(tuple(reversed(view.shape)), tuple(reversed(view.strides)))
        candidate = ComposedLayout(Swizzle(bits, 4, 3), offset, view)
        assert factory(element_bytes).match(candidate, {"n": mn}) is not None, (
            role, major, bits, element_bytes, mn, bk, order, start, candidate
        )
