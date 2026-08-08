"""IntTuple alias + helpers."""

from __future__ import annotations

from typing import Union, overload

from tilefoundry.ir.types.shape_dim import ShapeDim

IntTuple = Union[int, tuple["IntTuple", ...]]


@overload
def flatten(t: IntTuple) -> tuple[int, ...]: ...


@overload
def flatten(t: object) -> tuple[object, ...]: ...


def flatten(t: object) -> tuple[object, ...]:
    if not isinstance(t, tuple):
        return (t,)
    return tuple(value for item in t for value in flatten(item))


def product(t) -> "ShapeDim | None":
    from .mesh import Topology  # noqa: PLC0415

    result = 1
    for v in flatten(t):
        if isinstance(v, Topology):
            v = v.size
        if v is None:
            return None
        result *= v
    return result


__all__ = ["IntTuple", "flatten", "product"]
