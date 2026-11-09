"""The parameterized SM90 warpgroup MMA declaration."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.types import layout_algebra

from .mma_atom import MmaAtom, execution_mesh_pattern

WARPGROUP = types.Mesh(
    (types.Topology("thread", 128),),
    types.Layout((4, 8, 4), (32, 4, 1)),
    ("warp", "lane8", "lane4"),
)


@expr.register_type(dialect="T", namespace="cuda.sm90")
class Major(Enum):
    """Which of a shared descriptor's axes runs contiguously."""

    MN = "MN-major"
    K = "K-major"


@expr.register_type(dialect="T", namespace="cuda.sm90")
class Form(Enum):
    """Whether WGMMA reads A from shared memory or registers."""

    SS = "SS"
    RS = "RS"


_ = pattern.Wildcard()
n, b, W, p, r, l, s, c = (
    pattern.Wildcard(name) for name in ("n", "a_swizzle", "W", "k0", "r", "l", "s", "c")
)
n_extent = n
bb, Wb, rb, lb, sb, pb = (
    pattern.Wildcard(name) for name in ("b_swizzle", "Wb", "rb", "lb", "sb", "k0b")
)
SWIZZLE_BYTES = (16, 32, 64, 128)
DTYPES = (expr.DType.bf16, expr.DType.f16, expr.DType.fp8e4m3)
BROADCAST = (types.Broadcast(), types.Broadcast(), types.Broadcast())
PER_THREAD = (types.Split(0), types.Split(2), types.Split(4))


def swizzle_row_elements(swizzle: pattern.Wildcard, element_bytes: int):
    """Elements in one swizzled row for an operand of *element_bytes* bytes."""
    return pattern.Table(tuple(width // element_bytes for width in SWIZZLE_BYTES))[swizzle]


def mma_k_elements(element_bytes: int) -> int:
    """One issue reads 32 bytes of K."""
    return 32 // element_bytes


def canonical_smem_layout(layout, element_bytes: int, major: Major, *, b_operand=False):
    """Return CuTe make_gmma_desc's mode tree, in element units, for matching.

    Its (inner, repeat) coordinates are not logical-axis row-major digits;
    the view must not flow into address analysis or emission.
    """
    if isinstance(layout, types.ShardLayout):
        return replace(
            layout,
            layout=canonical_smem_layout(layout.layout, element_bytes, major, b_operand=b_operand),
        )
    if isinstance(layout, types.Layout):
        outer, swizzle = layout, 0
    elif (
        isinstance(layout, types.ComposedLayout)
        and isinstance(layout.outer, types.Layout)
        and (layout.inner is None or isinstance(layout.inner, types.Swizzle))
    ):
        outer, swizzle = layout.outer, 0 if layout.inner is None else layout.inner.bits
    else:
        return layout
    if not 0 <= swizzle < 4:
        return layout
    if b_operand:
        outer = types.Layout(outer.shape[::-1], outer.strides[::-1])
    width = SWIZZLE_BYTES[swizzle] // element_bytes
    tiles = (width, 8) if major is Major.MN else (8, 16 // element_bytes)
    outer = layout_algebra.logical_divide(outer, tuple(types.Layout((tile,), (1,)) for tile in tiles), major="row")
    if b_operand:
        outer = types.Layout(outer.shape[::-1], outer.strides[::-1])
    return replace(layout, outer=outer) if isinstance(layout, types.ComposedLayout) else outer


def smem_pattern(element_bytes: int, major: Major, *, b_operand=False):
    sw, w, rows, leading, stride, offset, extent = (
        (bb, Wb, rb, lb, sb, pb, n_extent) if b_operand else (b, W, r, l, s, p, 64)
    )
    vector = 16 // element_bytes
    prefix = "b" if b_operand else "a"
    stride_00, stride_10 = (
        pattern.Wildcard(f"{prefix}_stride_{position}") for position in ("00", "10")
    )
    if major is Major.MN:
        shape = ((w, rows), (8, 2))
        strides = (
            (stride_00, pattern.Switch(sw.name, {0: stride, 1: leading, 2: leading, 3: leading})),
            (stride_10, pattern.Switch(sw.name, {0: leading, 1: stride, 2: stride, 3: stride})),
        )
        origin = 0
        predicates = (
            rows * w == extent,
            stride_00 == 1,
            stride_10 == w,
            leading % vector == 0,
            stride % vector == 0,
        )
    else:
        leading = pattern.Wildcard(f"{prefix}_stride_11")
        shape = ((8, rows), (vector, 2))
        strides = ((stride_00, stride), (stride_10, leading))
        origin = offset
        predicates = (
            8 * rows == extent,
            stride_00 == swizzle_row_elements(sw, element_bytes),
            stride_10 == 1,
            (sw == 0) | (leading == vector),
            leading % vector == 0,
            stride % vector == 0,
            offset % mma_k_elements(element_bytes) == 0,
            0 <= offset,
            offset < swizzle_row_elements(sw, element_bytes),
        )
    return pattern.ComposedLayout(
        pattern.Swizzle(sw, 4, 3),
        origin,
        pattern.Layout(shape[::-1], strides[::-1]) if b_operand else pattern.Layout(shape, strides),
        predicates=((w == swizzle_row_elements(sw, element_bytes),) if major is Major.MN else ())
        + predicates,
    )


def a_register_fragment_pattern(element_bytes: int) -> pattern.Layout:
    """A's register fragment; each thread holds 4 bytes of K per pair of rows."""
    per_register = 4 // element_bytes
    return pattern.Layout(
        (4, 2, 8, 2, 4, per_register), (16, 8, 1, 256 * per_register, 64 * per_register, 64)
    )


fragment = pattern.Layout((4, 2, 8, c, 4, 2), (16, 8, 1, 512, 128, 64))


N = expr.DimVar("n", 8, 256)


def by_major(element_bytes: int, *, b_operand=False) -> dict:
    """Descriptor modes by major; only 16-bit operands may be MN-major."""
    return {
        major: smem_pattern(element_bytes, major, b_operand=b_operand)
        for major in Major
        if major is Major.K or element_bytes == 2
    }


def a_layout(dtype: expr.DType) -> pattern.Switch:
    element_bytes = dtype.bit_width // 8
    return pattern.Switch(
        "form",
        {
            Form.SS: pattern.ShardLayout(
                pattern.Switch(
                    "a_major",
                    by_major(element_bytes),
                ),
                BROADCAST,
                execution_mesh_pattern(WARPGROUP),
            ),
            Form.RS: pattern.ShardLayout(
                a_register_fragment_pattern(element_bytes),
                PER_THREAD,
                execution_mesh_pattern(WARPGROUP),
            ),
        },
    )


def b_layout(dtype: expr.DType) -> pattern.ShardLayout:
    return pattern.ShardLayout(
        pattern.Switch(
            "b_major",
            by_major(dtype.bit_width // 8, b_operand=True),
        ),
        BROADCAST,
        execution_mesh_pattern(WARPGROUP),
    )


A_MAJOR_16 = pattern.Switch("form", {Form.SS: pattern.Or(*tuple(Major)), Form.RS: Major.K})


OPERAND_DTYPE = pattern.Switch("dtype", {dtype: dtype for dtype in DTYPES})
K_ELEMENTS = pattern.Switch(
    "dtype", {dtype: mma_k_elements(dtype.bit_width // 8) for dtype in DTYPES}
)


class Wgmma(MmaAtom):
    """A warpgroup MMA, 64 x n x k, accumulating in F32; k is 32 bytes of the operand dtype."""

    namespace = "T.cuda.sm90"
    execution_mesh = WARPGROUP
    capability = "wgmma.mma_async"
    resource = "wgmma_engine"

    def match_view(self, param_name: str, type_: types.TensorType) -> types.TensorType:
        """Use descriptor modes only for shared-operand pattern matching."""
        if type_.storage is not expr.StorageKind.SMEM:
            return type_
        b_operand = param_name == "rhs"
        return replace(
            type_,
            layout=canonical_smem_layout(
                type_.layout,
                type_.dtype.bit_width // 8,
                self.bindings["b_major" if b_operand else "a_major"],
                b_operand=b_operand,
            ),
        )

    n = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "n",
            predicates=(
                pattern.Range(lo=N.lo, hi=N.hi),
                pattern.Wildcard("n") % 8 == 0,
            ),
        ),
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("dtype", predicates=(pattern.Or(*DTYPES),)),
    )
    form = expr.ParamDef(
        kind="attribute",
        annotation=Form,
        pattern=pattern.Wildcard(
            "form",
            predicates=(pattern.Or(*tuple(Form)),),
        ),
    )
    a_major = expr.ParamDef(
        kind="attribute",
        annotation=Major,
        pattern=pattern.Switch(
            "dtype",
            {
                expr.DType.bf16: A_MAJOR_16,
                expr.DType.f16: A_MAJOR_16,
                expr.DType.fp8e4m3: Major.K,
            },
        ),
        default=Major.MN,
    )
    b_major = expr.ParamDef(
        kind="attribute",
        annotation=Major,
        pattern=pattern.Switch(
            "dtype",
            {
                expr.DType.bf16: pattern.Or(*tuple(Major)),
                expr.DType.f16: pattern.Or(*tuple(Major)),
                expr.DType.fp8e4m3: Major.K,
            },
        ),
        default=Major.MN,
    )

    C = pattern.Tensor(
        shape=(64, N),
        dtype=expr.DType.f32,
        storage=expr.StorageKind.RMEM,
        layout=pattern.ShardLayout(
            fragment,
            PER_THREAD,
            execution_mesh_pattern(WARPGROUP),
        ),
        predicates=(8 * c == n_extent,),
    )
    A = pattern.Tensor(
        dtype=OPERAND_DTYPE,
        storage=pattern.Switch("form", {Form.SS: expr.StorageKind.SMEM, Form.RS: expr.StorageKind.RMEM}),
        shape=(
            64,
            K_ELEMENTS,
        ),
        layout=pattern.Switch("dtype", {dtype: a_layout(dtype) for dtype in DTYPES}),
    )
    B = pattern.Tensor(
        dtype=OPERAND_DTYPE,
        storage=expr.StorageKind.SMEM,
        shape=(
            K_ELEMENTS,
            N,
        ),
        layout=pattern.Switch("dtype", {dtype: b_layout(dtype) for dtype in DTYPES}),
    )


__all__ = [
    "BROADCAST",
    "DTYPES",
    "Form",
    "Major",
    "N",
    "PER_THREAD",
    "SWIZZLE_BYTES",
    "WARPGROUP",
    "Wgmma",
    "smem_pattern",
    "canonical_smem_layout",
    "fragment",
]
