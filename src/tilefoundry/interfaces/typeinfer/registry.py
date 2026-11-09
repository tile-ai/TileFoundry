"""Type-inference rules keyed by IR class, or by Target class and IR class."""

from __future__ import annotations

from collections.abc import Callable

from tilefoundry import dispatch

from .context import TypeInferResults

__all__ = ["register", "registry"]

registry = dispatch.Registry("typeinfer")


def register(cls: type, *, target: type | None = None) -> Callable[[Callable], Callable]:
    """Register one type rule, normalizing its result without affecting peers.

    With ``target``, the rule answers for ``cls`` only inside a Module whose
    resolved Target is of that type; it is keyed ``(target, cls)``.
    """

    def decorator(fn: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            result = fn(*args, **kwargs)
            return result if isinstance(result, TypeInferResults) else TypeInferResults(result)

        registry.register(cls if target is None else (target, cls), wrapped)
        return fn

    return decorator
