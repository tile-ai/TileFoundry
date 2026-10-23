"""Hierarchical dialect lookup shared by the T and tf surface modules.

See [parser §2](docs/spec/parser.md#2-syntax-and-rules).
"""

from __future__ import annotations

from typing import Any

from tilefoundry.ir.core.op_registry import children, lookup
from tilefoundry.ir.core.op_schema import OpSchema


class Dialect:
    """Resolve one level of the registered dialect tree."""

    def __init__(self, dialect: str, path: tuple[str, ...] = ()):
        self._dialect = dialect
        self._path = path

    def __getattr__(self, name: str) -> Any:
        if name == "__all__":
            return self.__dir__()
        entry = lookup(self._dialect, self._path, name)
        if entry is not None:
            return entry.builder if isinstance(entry, OpSchema) else entry
        if name in self.__dir__():
            return Dialect(self._dialect, (*self._path, name))
        path = ".".join((self._dialect, *self._path))
        raise AttributeError(f"tilefoundry.dsl.{path} has no op named {name!r}")

    def __dir__(self) -> list[str]:
        return children(self._dialect, self._path)


def make_dialect(dialect: str) -> Dialect:
    """Create the root namespace for a dialect."""
    return Dialect(dialect)


__all__ = ["Dialect", "make_dialect"]
