"""nncase-style scoped filesystem dump.

Public surface:

- ``DumpScope``: context manager. Enter to nest a sub-scope (or replace
  the current dumper); exit to restore the prior scope. Backed by a
  ``contextvars.ContextVar`` so threads and asyncio Tasks see their own
  active scope.
- ``DumpFlags``: ``IntFlag`` selecting which categories actually emit.
  A child scope's flags are restricted by the parent's
  (``child &= parent``).
- ``IDumpper`` / ``FileDumper`` / ``MemoryDumper`` / ``NullDumper``: backend
  variants. ``FileDumper`` is the default for production runs;
  ``MemoryDumper`` is meant for in-test capture; ``NullDumper`` is a
  singleton no-op used when no scope is active or when all flags are off.
- ``dump(name, content, flag)``: top-level helper that routes to the
  current scope's dumper if ``flag`` is enabled.
- ``current_scope()``: read-only accessor for the active scope (``None``
  outside any scope).
"""
from __future__ import annotations

from .dumpers import FileDumper, IDumpper, MemoryDumper, NullDumper
from .flags import DumpFlags
from .scope import DumpScope, current_scope, dump

__all__ = [
    "DumpFlags",
    "DumpScope",
    "IDumpper",
    "FileDumper",
    "MemoryDumper",
    "NullDumper",
    "current_scope",
    "dump",
]
