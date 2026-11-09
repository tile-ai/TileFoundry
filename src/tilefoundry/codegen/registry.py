"""Code-generation handlers keyed by target, call position and class.

A registry is one key -> handler map; what the key is depends on the question
it answers. Code generation keys on three things at once: which target's
language the answer is written in, which position of a call it is written at,
and the class asked about.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from tilefoundry import dispatch

__all__ = ["Role", "register", "registry"]


class Role(Enum):
    """Which position of a call a code-generation handler answers for.

    A node is emitted where it stands. A callee declares the parameters it
    takes; a caller supplies the arguments for them. Those two read the same
    signature from opposite sides, which is why they are one key apart rather
    than two registries apart. A launch is written where the callee's target
    states it: the caller's own target forwards to that target's ``LAUNCH``
    handler.
    """

    EMIT = "emit"
    CALLEE = "callee"
    CALLER = "caller"
    LAUNCH = "launch"


registry = dispatch.Registry("codegen")


def register(target: type, role: Role, cls: type) -> Callable[[Callable], Callable]:
    """Register *fn* as how *target* writes *cls* at *role*'s position.

    *target* is the one whose language the answer is written in, which for a
    call is the callee's: how a call to it is spelled is its own convention,
    and a caller of another target fills in only the names it holds.
    """

    def decorator(fn: Callable) -> Callable:
        registry.register((target, role, cls), fn)
        return fn

    return decorator
