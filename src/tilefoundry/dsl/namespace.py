"""Hierarchical dialect lookup shared by the T and tf surface modules.

See [parser §2](docs/spec/parser.md#2-syntax-and-rules).
"""

from __future__ import annotations

from typing import Any

from tilefoundry.ir import expr
from tilefoundry.ir.expr import op_registry


class Dialect:
    """Resolve one level of the registered dialect tree."""

    def __init__(self, dialect: str, path: tuple[str, ...] = ()):
        self.registered_dialect = dialect
        self.registered_path = path

    @classmethod
    def from_name(cls, dialect: str) -> Dialect:
        """The root namespace of the dialect named *dialect*."""
        return cls(dialect)

    def __getattr__(self, name: str) -> Any:
        if name == "__all__":
            return self.__dir__()
        entry = op_registry.lookup(self.registered_dialect, self.registered_path, name)
        if entry is not None:
            return entry.builder if isinstance(entry, expr.OpSchema) else entry
        if name in self.__dir__():
            return Dialect(self.registered_dialect, (*self.registered_path, name))
        path = ".".join((self.registered_dialect, *self.registered_path))
        raise AttributeError(f"tilefoundry.dsl.{path} has no op named {name!r}")

    def __dir__(self) -> list[str]:
        return op_registry.children(self.registered_dialect, self.registered_path)


__all__ = ["Dialect"]
