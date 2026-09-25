"""Construction and specialization helpers for IR patterns."""

from __future__ import annotations

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.types import ComposedLayout, Mesh, StorageKind, Swizzle

from .pattern import (
    AttrPattern,
    CapturePattern,
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    MultipleOfPattern,
    OneOfPattern,
    OrPattern,
    Pattern,
    RangePattern,
    SwizzlePattern,
    TensorPattern,
    WildcardPattern,
)

MOVED_STORAGES = (StorageKind.GMEM, StorageKind.SMEM, StorageKind.RMEM)
WHOLE_BYTES = AttrPattern("bit_width", MultipleOfPattern(8))


def moved_tile(index: int, storage=None, layout=None) -> TensorPattern:
    """A moved tensor tile, with dtype and storage captures named by end."""
    storages = MOVED_STORAGES if storage is None else storage
    return TensorPattern(
        dtype=CapturePattern(dtype_place(index), WHOLE_BYTES),
        storage=(
            CapturePattern(storage_place(index), OneOfPattern(tuple(storages)))
            if isinstance(storages, tuple)
            else storages
        ),
        layout=layout,
    )


def dtype_place(index: int) -> str:
    """The capture name for transfer end *index*'s dtype."""
    return f"dtype{index}"


def storage_place(index: int) -> str:
    """The capture name for transfer end *index*'s storage."""
    return f"storage{index}"


_ANY_THREADS = OrPattern(
    ComposedLayoutPattern(
        offset=WildcardPattern(),
        outer=LayoutPattern(
            ((CapturePattern("n", RangePattern(lo=1)),),),
            ((1,),),
            per_mode=True,
        ),
    ),
    LayoutPattern(
        ((CapturePattern("n", RangePattern(lo=1)),),),
        ((1,),),
        per_mode=True,
    ),
)


def any_threads() -> ParamDef:
    """Declare an optional scope spanning one or more threads."""
    return ParamDef(
        kind="attribute",
        annotation=Mesh,
        pattern=MeshPattern(("thread",), _ANY_THREADS),
        optional=True,
        default=None,
    )


def arrangement_pattern(
    layout,
    *,
    forward: bool = True,
    injective: bool = True,
    per_mode: bool = False,
):
    """Build the exact pattern for one authored arrangement."""
    if isinstance(layout, ComposedLayout):
        inner = layout.inner
        return ComposedLayoutPattern(
            SwizzlePattern(inner.bits, inner.base, inner.shift)
            if isinstance(inner, Swizzle)
            else inner,
            layout.offset,
            arrangement_pattern(
                layout.outer,
                forward=forward,
                injective=injective,
                per_mode=per_mode,
            ),
        )
    return LayoutPattern(
        tuple(layout.shape),
        tuple(layout.strides),
        forward=forward,
        injective=injective,
        per_mode=per_mode,
    )


def locate_dim_var(params: tuple, name: str) -> tuple[int, int] | None:
    """Return the first parameter/axis carrying a ``DimVar`` named *name*."""
    for index, param in enumerate(params):
        shape = getattr(param.type, "shape", None)
        if shape is None:
            continue
        for axis, dim in enumerate(shape):
            if getattr(dim, "name", None) == name:
                return index, axis
    return None


def _mangle_variant_name(name: str, specializations: tuple[Pattern, ...]) -> str:
    if len(specializations) != 1 or not isinstance(specializations[0], RangePattern):
        raise TypeError("variant requires exactly one RangePattern")
    pattern = specializations[0]
    if not pattern.dim_var or pattern.lo is None or pattern.hi is None:
        raise TypeError("variant RangePattern requires dim_var, lo, and hi")
    return f"{name}${pattern.dim_var}${pattern.lo}_{pattern.hi}"


__all__ = [
    "MOVED_STORAGES",
    "WHOLE_BYTES",
    "_mangle_variant_name",
    "any_threads",
    "arrangement_pattern",
    "dtype_place",
    "locate_dim_var",
    "moved_tile",
    "storage_place",
]
