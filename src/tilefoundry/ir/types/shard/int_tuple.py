"""IntTuple alias + helpers.

CuTe-style nested int tuples (ints or tuples thereof). Used by `Layout`
shape / stride arguments, whose entries may also be symbolic / dynamic dims
(a ``ShapeDim``) or ``None`` for a launch-provided extent; the ``flatten`` /
``product`` helpers here are for the fully-static case. Consumers needing a
concrete integer (``Mesh.__getitem__`` / ``T.sync``) require static ints and
fail closed otherwise.
"""

from __future__ import annotations

from typing import Union

IntTuple = Union[int, tuple["IntTuple", ...]]


def flatten(t: IntTuple) -> tuple[int, ...]:
    if isinstance(t, int):
        return (t,)
    out: list[int] = []
    for x in t:
        out.extend(flatten(x))
    return tuple(out)


def product(t: IntTuple) -> int:
    result = 1
    for v in flatten(t):
        result *= v
    return result


__all__ = ["IntTuple", "flatten", "product"]
