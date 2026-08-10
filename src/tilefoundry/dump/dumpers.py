"""Provide filesystem, memory, and null dump backends.

Subdirectory scopes compose the parent backend with a relative prefix. File
backends change their root, memory backends prefix shared keys, and the null
backend returns itself.
"""
from __future__ import annotations

import os
from typing import Protocol


class IDumpper(Protocol):
    """Write surface used by ``DumpScope.dump`` and the top-level ``tilefoundry.dump.dump`` helper.

    Write surface used by ``DumpScope.dump`` and the top-level
    ``tilefoundry.dump.dump`` helper. Implementations must be safe to call
    concurrently from multiple threads (a ``ContextVar`` per call site
    isolates the active scope, but two scopes pointing at the same
    backend may interleave writes).
    """

    def write(self, name: str, content: str | bytes) -> None: ...

    def subdir(self, name: str) -> "IDumpper": ...


class FileDumper:
    """Filesystem-backed dumper.

    Filesystem-backed dumper. ``root`` is created lazily on first write
    so empty scopes do not litter the disk.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = os.fspath(root)

    @property
    def root(self) -> str:
        return self._root

    def write(self, name: str, content: str | bytes) -> None:
        path = os.path.join(self._root, name)
        os.makedirs(os.path.dirname(path) or self._root, exist_ok=True)
        if isinstance(content, bytes):
            with open(path, "wb") as f:
                f.write(content)
        else:
            with open(path, "w") as f:
                f.write(content)

    def subdir(self, name: str) -> "FileDumper":
        return FileDumper(os.path.join(self._root, name))


class MemoryDumper:
    """In-memory dumper.

    In-memory dumper. Stores the latest write per logical path in
    ``self.entries``. Subdirs prefix keys with ``f"{name}/"`` so a nested
    write to ``codegen/module.cu`` from ``DumpScope("codegen")`` lands as
    ``codegen/module.cu`` in the parent's view.
    """

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix
        self.entries: dict[str, str | bytes] = {}

    def write(self, name: str, content: str | bytes) -> None:
        key = f"{self._prefix}{name}" if self._prefix else name
        self.entries[key] = content

    def subdir(self, name: str) -> "MemoryDumper":
        child = MemoryDumper(
            prefix=f"{self._prefix}{name}/" if self._prefix else f"{name}/"
        )


        child.entries = self.entries
        return child


class _NullDumperType:
    def write(self, name: str, content: str | bytes) -> None:
        return None

    def subdir(self, name: str) -> "_NullDumperType":
        return self


NullDumper: _NullDumperType = _NullDumperType()


__all__ = ["IDumpper", "FileDumper", "MemoryDumper", "NullDumper"]
