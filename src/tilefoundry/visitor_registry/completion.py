"""When the memory effects an Op declares are complete for later accesses.

A synchronisation plan orders a later access after an earlier one only once the
earlier one is complete. An Op that states nothing is ``UNKNOWN``. See
[visitor-registry §4.2](docs/spec/visitor-registry.md#42-completion-service--completion).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from .registries import DispatchRegistry


class Completion(Enum):
    """How an Op's memory effects become complete for the thread that issued it."""

    UNKNOWN = "unknown"
    ON_RETURN = "on_return"
    CP_ASYNC_GROUP = "cp_async_group"


completion_registry: DispatchRegistry = DispatchRegistry("completion")


def register_completion(
    key: type, completion: Completion | Callable[[object], Completion]
) -> None:
    """Declare *key*'s completion, or a handler that answers for one instance of it."""
    completion_registry.register(
        key, completion if callable(completion) else (lambda _instance: completion)
    )


def completion_of(instance: object) -> Completion:
    """The completion declared for *instance*'s exact class, or ``UNKNOWN``."""
    handler = completion_registry.lookup(type(instance))
    return Completion.UNKNOWN if handler is None else handler(instance)


__all__ = ["Completion", "completion_of", "completion_registry", "register_completion"]
