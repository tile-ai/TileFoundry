"""Construction and specialization helpers for IR patterns."""

from __future__ import annotations

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.types import Mesh, StorageKind

from . import predicates as P
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
    TensorPattern,
    WildcardPattern,
)

MOVED_STORAGES = (StorageKind.GMEM, StorageKind.SMEM, StorageKind.RMEM)
WHOLE_BYTES = AttrPattern("bit_width", MultipleOfPattern(8))


def operand_tile(index: int, storage=None, layout=None) -> TensorPattern:
    """A tensor tile whose dtype and storage captures are named by operand slot."""
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


def whole_vectors(index: int, widths: tuple[int, ...]) -> LayoutPattern:
    """Whole vectors at *widths*, counted in operand *index*'s element dtype."""
    return LayoutPattern(
        predicates=(
            P.WholeVectors(
                CapturePattern("width", OneOfPattern(widths)),
                dtype_place(index),
                widths,
            ),
        )
    )


_ANY_THREADS = OrPattern(
    ComposedLayoutPattern(
        offset=WildcardPattern(),
        outer=LayoutPattern(
            ((CapturePattern("n", RangePattern(lo=1)),),),
            ((1,),),
            predicates=(P.Forward(per_mode=True), P.Injective(per_mode=True)),
        ),
    ),
    LayoutPattern(
        ((CapturePattern("n", RangePattern(lo=1)),),),
        ((1,),),
        predicates=(P.Forward(per_mode=True), P.Injective(per_mode=True)),
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
    "dtype_place",
    "locate_dim_var",
    "operand_tile",
    "storage_place",
    "whole_vectors",
]
