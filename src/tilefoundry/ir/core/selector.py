"""Selector: the text ``root[.segment]*[:variable]`` that names a node of a Module tree.

The segments after the root are the dotted spelling of a ``FunctionRef``: a
child slot's key enters it, a number picks one element of a ModuleList, and a
last segment that is no child slot names a function. Which Module the root
names is the caller's to say.
"""

from __future__ import annotations

from dataclasses import dataclass

from .function_ref import FunctionRef
from .module import Module, ModuleFunction, descend, resolve

__all__ = ["Selector", "parse_selector", "select"]


@dataclass(frozen=True)
class Selector:
    root: str
    segments: tuple[str, ...]
    variable: str | None


def parse_selector(text: str) -> Selector:
    """Split *text* at its last ``:`` and its dots; an empty part names nothing."""
    path, colon, variable = text.rpartition(":")
    if not colon:
        path = text
    segments = path.split(".")
    if not all(segments):
        raise ValueError(
            "an empty segment names nothing. A path is its segments, so a leading, "
            "trailing or doubled dot would make two different selectors name one node"
        )
    if colon and not variable:
        raise ValueError("nothing follows ':'; name a variable or leave out the ':'")
    return Selector(segments[0], tuple(segments[1:]), variable if colon else None)


def select(start: Module, segments: tuple[str, ...]) -> tuple[FunctionRef, Module, ModuleFunction | None]:
    """The reference *segments* spell from *start*, the Module it reaches, the function it names.

    The function is ``None`` when the last segment is a child slot or an index.
    """
    path: list[str | int] = []
    name = None
    for position, segment in enumerate(segments):
        if segment.isdigit():
            path.append(int(segment))
        elif position == len(segments) - 1 and segment not in reached(start, path).children:
            name = segment
        else:
            path.append(segment)
    ref = FunctionRef(tuple(path), name)
    return ref, reached(start, path), None if name is None else resolve(start, ref)[1]


def reached(start: Module, path: list[str | int]) -> Module:
    module = start
    for module in descend(start, FunctionRef(tuple(path))):
        pass
    return module
