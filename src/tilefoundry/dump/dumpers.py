"""Provide filesystem, memory, and null dump backends.

Subdirectory scopes compose the parent backend with a relative prefix. File
backends change their root, memory backends prefix shared keys, and the null
backend returns itself.
"""

from __future__ import annotations

import os
from typing import Protocol, override

__all__ = ["FileDumper", "IDumpper", "MemoryDumper", "NullDumper"]


class IDumpper(Protocol):
    """Write surface used by ``Scope.dump`` and the top-level ``tilefoundry.dump.dump`` helper.

    Implementations must be safe to call concurrently from multiple threads (a
    ``ContextVar`` per call site isolates the active scope, but two scopes
    pointing at the same backend may interleave writes).
    """

    def write(self, name: str, content: str | bytes) -> None: ...

    def subdir(self, name: str) -> "IDumpper": ...


class FileDumper:
    """Filesystem-backed dumper.

    ``root`` is created lazily on first write so empty scopes do not litter the
    disk.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = os.fspath(root)

    def write(self, name: str, content: str | bytes) -> None:
        path = os.path.join(self.root, name)
        os.makedirs(os.path.dirname(path) or self.root, exist_ok=True)
        if isinstance(content, bytes):
            with open(path, "wb") as f:
                f.write(content)
        else:
            with open(path, "w") as f:
                f.write(content)

    def subdir(self, name: str) -> "FileDumper":
        return FileDumper(os.path.join(self.root, name))


class MemoryDumper:
    """In-memory dumper.

    Stores the latest write per logical path in ``self.entries``. Subdirs
    prefix keys with ``f"{name}/"`` so a nested write to ``codegen/module.cu``
    from ``Scope("codegen")`` lands as ``codegen/module.cu`` in the parent's
    view. A subdir shares the ``entries`` dict of its parent.
    """

    def __init__(self, prefix: str = "", entries: dict[str, str | bytes] | None = None) -> None:
        self.prefix = prefix
        self.entries: dict[str, str | bytes] = {} if entries is None else entries

    def write(self, name: str, content: str | bytes) -> None:
        key = f"{self.prefix}{name}" if self.prefix else name
        self.entries[key] = content

    def subdir(self, name: str) -> "MemoryDumper":
        return MemoryDumper(
            prefix=f"{self.prefix}{name}/" if self.prefix else f"{name}/",
            entries=self.entries,
        )


class NullDumperType(IDumpper):
    @override
    def write(self, name: str, content: str | bytes) -> None:
        return None

    @override
    def subdir(self, name: str) -> "NullDumperType":
        return self


NullDumper: NullDumperType = NullDumperType()
