"""Canonical home for DispatchRegistry and the registry instances built on it.

A registry is one key -> handler map; what the key is depends on the question
it answers. Type inference, verification and cost key on a class alone. Code
generation keys on three things at once: which target's language the answer is
written in, which position of a call it is written at, and the class asked
about.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable


def spelled(key: object) -> str:
    """A key as a message names it: a class by its name, a tuple by its parts."""
    if isinstance(key, tuple):
        return "(" + ", ".join(spelled(part) for part in key) + ")"
    return key.__name__ if isinstance(key, type) else str(key)


class DispatchRegistry[Key]:
    """Key → handler map. Double registration raises; lookup miss returns None."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._map: dict[Key, Callable] = {}

    def register(self, key: Key, fn: Callable) -> None:
        if key in self._map:
            raise RuntimeError(f"{self.name}: {spelled(key)} already registered")
        self._map[key] = fn

    def lookup(self, key: Key) -> Callable | None:
        return self._map.get(key)

    def has(self, key: Key) -> bool:
        return key in self._map

    def decorator(self) -> Callable[[type], Callable[[Callable], Callable]]:
        """``@registry.decorator()`` factory.

        ``@registry.decorator()`` factory: ``register_X = registry.decorator()``
        gives the conventional ``register_X(cls)`` decorator for this registry.
        """

        def register_for(cls: type) -> Callable[[Callable], Callable]:
            def decorator(fn: Callable) -> Callable:
                self.register(cls, fn)
                return fn

            return decorator

        return register_for


class Role(Enum):
    """Which position of a call a code-generation handler answers for.

    A node is emitted where it stands. A callee declares the parameters it
    takes; a caller supplies the arguments for them. Those two read the same
    signature from opposite sides, which is why they are one key apart rather
    than two registries apart.
    """

    EMIT = "emit"
    CALLEE = "callee"
    CALLER = "caller"


typeinfer_registry: DispatchRegistry = DispatchRegistry("typeinfer")
verify_stmt_registry: DispatchRegistry = DispatchRegistry("verify_stmt")
cost_evaluator_registry: DispatchRegistry = DispatchRegistry("cost_evaluator")

codegen_registry: DispatchRegistry = DispatchRegistry("codegen")
"""Keyed by ``(target, role, class)``; see :func:`register_codegen`."""


register_verify_stmt = verify_stmt_registry.decorator()
register_cost_evaluator = cost_evaluator_registry.decorator()


def register_typeinfer(cls: type) -> Callable[[Callable], Callable]:
    """Register one type rule, normalizing its result without affecting peers."""
    from .contexts import TypeInferResults  # noqa: PLC0415

    def decorator(fn: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            result = fn(*args, **kwargs)
            return result if isinstance(result, TypeInferResults) else TypeInferResults(result)

        typeinfer_registry.register(cls, wrapped)
        return fn

    return decorator


def register_codegen(
    target: type["Target"], role: Role, cls: type
) -> Callable[[Callable], Callable]:
    """Register *fn* as how *target* writes *cls* at *role*'s position.

    *target* is the one whose language the answer is written in, which for a
    call is the callee's: how a call to it is spelled is its own convention,
    and a caller of another target fills in only the names it holds.
    """

    def decorator(fn: Callable) -> Callable:
        codegen_registry.register((target, role, cls), fn)
        return fn

    return decorator


__all__ = [
    "DispatchRegistry",
    "Role",
    "codegen_registry",
    "cost_evaluator_registry",
    "register_codegen",
    "register_cost_evaluator",
    "register_typeinfer",
    "register_verify_stmt",
    "spelled",
    "typeinfer_registry",
    "verify_stmt_registry",
]
