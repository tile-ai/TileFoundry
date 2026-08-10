"""``DumpScope`` context manager + ``ContextVar`` machinery.

Two ctor forms (mirroring nncase):

- ``DumpScope(subdir, flags=None)`` — nest under the current scope. The
  effective dumper is ``parent_dumper.subdir(name)``; effective flags are
  ``flags & parent_flags`` (a child cannot widen what the parent allowed).
  If no scope is active the parent dumper is ``NullDumper`` and the
  effective dumper stays ``NullDumper`` regardless of flags.
- ``DumpScope(dumper=…, flags=DumpFlags.ALL)`` — replace the current
  scope entirely. This is the form tests use when they want a clean
  ``MemoryDumper`` view, or when the top-level entry point sets up a
  fresh ``FileDumper`` rooted at e.g. ``test_results/<nodeid>``.

Concurrency is via ``contextvars.ContextVar``:

- **Threads are isolated.** Plain child threads do *not* inherit the
  parent's active scope; each starts in the default ``None`` scope and
  must install its own. Code that needs propagation can wrap the target
  with ``contextvars.copy_context().run(target, ...)`` explicitly.
- **Asyncio Tasks copy the context** at creation time, so an awaited
  task sees the spawning coroutine's scope but its own writes don't
  leak back to the parent.

``__enter__`` / ``__exit__`` use the ``set`` / ``reset`` token pattern so
nested entries unwind cleanly even if ``__exit__`` runs in a different
async task.
"""
from __future__ import annotations

import contextvars
from typing import Optional

from .dumpers import IDumpper, NullDumper
from .flags import DumpFlags

_current: contextvars.ContextVar[Optional["DumpScope"]] = contextvars.ContextVar(
    "tilefoundry_dump_scope", default=None
)


class DumpScope:
    __slots__ = ("_subdir", "_dumper", "_flags", "_token", "_replace")

    def __init__(
        self,
        subdir: str | None = None,
        flags: DumpFlags | None = None,
        *,
        dumper: IDumpper | None = None,
    ) -> None:
        if dumper is not None and subdir is not None:
            raise ValueError(
                "DumpScope: pass either `subdir` (nest under current) or "
                "`dumper` (replace), not both"
            )
        self._subdir = subdir
        self._dumper = dumper
        self._flags = flags if flags is not None else DumpFlags.ALL
        self._replace = dumper is not None
        self._token: contextvars.Token | None = None

    @property
    def dumper(self) -> IDumpper:
        return self._dumper if self._dumper is not None else NullDumper

    @property
    def flags(self) -> DumpFlags:
        return self._flags

    def dump(self, name: str, content: str | bytes, flag: DumpFlags) -> None:
        """Write ``content`` to logical path ``name`` if ``flag`` is enabled in this scope.

        Write ``content`` to logical path ``name`` if ``flag`` is enabled
        in this scope. Falls through to ``NullDumper`` when the scope has
        no active dumper (e.g. ``DumpScope("foo")`` outside any parent).
        """
        if not (self._flags & flag):
            return
        self.dumper.write(name, content)

    def __enter__(self) -> "DumpScope":
        parent = _current.get()
        if not self._replace:
            if parent is None:
                # No outer scope — subdir form is a no-op view onto NullDumper.
                self._dumper = None
                # Flags are still restricted to NONE so no writes happen.
                self._flags = DumpFlags.NONE
            else:
                # Compose with parent's dumper + restrict flags.
                name = self._subdir or ""
                self._dumper = (
                    parent.dumper.subdir(name) if name else parent.dumper
                )
                self._flags = self._flags & parent.flags
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None


def current_scope() -> DumpScope | None:
    return _current.get()


def dump(name: str, content: str | bytes, flag: DumpFlags) -> None:
    """Convenience: route through the current scope, no-op if none."""
    scope = _current.get()
    if scope is None:
        return
    scope.dump(name, content, flag)


__all__ = ["DumpScope", "current_scope", "dump"]
