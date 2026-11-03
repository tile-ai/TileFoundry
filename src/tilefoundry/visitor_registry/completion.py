"""When each memory effect of an issued instruction is complete for later accesses.

A synchronisation plan orders a later access after an earlier one only once the
earlier one is complete. An effect no declaration states is ``unknown``. See
[visitor-registry §4.2](docs/spec/visitor-registry.md#42-completion-service--completion).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tilefoundry.ir.core.param_def import MemoryEffect

from .registries import DispatchRegistry


@dataclass(frozen=True)
class Issued:
    """One instruction lowering issues: the Op, its parameters and their types."""

    op: object
    params: tuple
    types: tuple


@dataclass(frozen=True)
class Effect:
    """When one read or write of an issued parameter is complete.

    ``on_return`` effects are complete for the issuing threads when the statement
    returns, ``group`` ones once those threads wait for ``group``, and ``unknown``
    ones are ordered as ``on_return``, so their emitter must wait for them.
    ``proxy`` is the path to shared memory, ``generic`` or ``async``. ``unit`` is
    the mesh of one group of threads that run the group's steps, and the issue
    itself, together. A ``fence`` effect on registers needs the group's fence
    since any other access to them, except an earlier one with the same ``chain``.
    """

    param: str
    write: bool
    mechanism: str
    group: str | None = None
    proxy: str = "generic"
    unit: object = None
    fence: bool = False
    chain: object = None


completion_registry: DispatchRegistry = DispatchRegistry("completion")


def effects(issued: Issued, **fields) -> tuple[Effect, ...]:
    """Each read, then each write, of *issued*'s parameters, all with *fields*."""
    return tuple(
        Effect(param.name, write, **fields)
        for param in issued.params
        for write in (False, True)
        if param.effect & (MemoryEffect.WRITE if write else MemoryEffect.READ)
    )


def on_return(issued: Issued) -> tuple[Effect, ...]:
    """Every effect of *issued* is complete once it returns."""
    return effects(issued, mechanism="on_return")


def register_completion(key: type, handler: Callable[[Issued], tuple[Effect, ...]]) -> None:
    """Declare the effects of an issued *key* through *handler*."""
    completion_registry.register(key, handler)


def completion_of(issued: Issued, *, key: type | None = None) -> tuple[Effect, ...]:
    """The effects declared for *issued*, looked up by *key* or its Op's exact class."""
    handler = completion_registry.lookup(type(issued.op) if key is None else key)
    return effects(issued, mechanism="unknown") if handler is None else handler(issued)


__all__ = [
    "Effect",
    "Issued",
    "completion_of",
    "completion_registry",
    "effects",
    "on_return",
    "register_completion",
]
