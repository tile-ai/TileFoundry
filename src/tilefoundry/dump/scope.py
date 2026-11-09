"""Manage nested and replacement dump scopes with a ``ContextVar``.

Subdirectory scopes compose a parent path and may only restrict parent flags.
Replacement scopes install a fresh backend. Threads start without the parent's
scope; asyncio tasks copy it at creation. Context tokens unwind nested entries
even when exceptions leave a scope.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Optional

from .dumpers import IDumpper, NullDumper
from .flags import Flags

__all__ = ["Scope", "current_scope", "dump"]

scope_var: contextvars.ContextVar[Optional["Scope"]] = contextvars.ContextVar(
    "tilefoundry_dump_scope", default=None
)


class Scope:
    __slots__ = ("subdir", "backend", "flags", "token", "replaces")

    def __init__(
        self,
        subdir: str | None = None,
        flags: Flags | None = None,
        *,
        dumper: IDumpper | None = None,
    ) -> None:
        if dumper is not None and subdir is not None:
            raise ValueError(
                "Scope: pass either `subdir` (nest under current) or "
                "`dumper` (replace), not both"
            )
        self.subdir = subdir
        self.backend = dumper
        self.flags = flags if flags is not None else Flags.ALL
        self.replaces = dumper is not None
        self.token: contextvars.Token | None = None

    @property
    def dumper(self) -> IDumpper:
        return self.backend if self.backend is not None else NullDumper

    def dump(
        self,
        name: str,
        content: str | bytes | Callable[[], str | bytes],
        flag: Flags,
    ) -> None:
        """Write ``content`` to logical path ``name`` if ``flag`` is enabled in this scope.

        A callable is evaluated only when the flag is enabled, immediately
        before writing. Producer errors propagate to the caller.
        """
        if not (self.flags & flag):
            return
        self.dumper.write(name, content() if callable(content) else content)

    def __enter__(self) -> "Scope":
        parent = scope_var.get()
        if not self.replaces:
            if parent is None:
                self.backend = None
                self.flags = Flags.NONE
            else:
                name = self.subdir or ""
                self.backend = parent.dumper.subdir(name) if name else parent.dumper
                self.flags = self.flags & parent.flags
        self.token = scope_var.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.token is not None:
            scope_var.reset(self.token)
            self.token = None


def current_scope() -> Scope | None:
    return scope_var.get()


def dump(
    name: str,
    content: str | bytes | Callable[[], str | bytes],
    flag: Flags,
) -> None:
    """Route through the current scope; without one, do not evaluate content."""
    scope = scope_var.get()
    if scope is None:
        return
    scope.dump(name, content, flag)
