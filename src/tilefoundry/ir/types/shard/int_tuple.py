"""IntTuple alias + helpers.

CuTe-style nested int tuples (ints or tuples thereof). Used by `Layout`
shape / stride arguments, whose entries may also be symbolic / dynamic dims
(a ``ShapeDim``) or ``None`` for a launch-provided extent; the ``flatten`` /
``product`` helpers here are for the fully-static case. Consumers needing a
concrete integer (``Mesh.__getitem__`` / ``T.sync``) require static ints and
fail closed otherwise.
"""

from __future__ import annotations

from typing import Union, overload

IntTuple = Union[int, tuple["IntTuple", ...]]


@overload
def flatten(t: IntTuple) -> tuple[int, ...]: ...


@overload
def flatten(t: object) -> tuple[object, ...]: ...


def flatten(t: object) -> tuple[object, ...]:
    if not isinstance(t, tuple):
        return (t,)
    return tuple(value for item in t for value in flatten(item))


def product(t: IntTuple) -> int:
    result = 1
    for v in flatten(t):
        result *= v
    return result


__all__ = ["IntTuple", "flatten", "product"]
