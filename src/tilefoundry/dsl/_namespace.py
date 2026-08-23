"""Define namespace behavior.

Shared ``__getattr__`` / ``__dir__`` factory for the ``tf`` and ``T`` modules.

Both dialect namespaces resolve names on demand against the OpSchema
registry with the same algorithm; this module ships that algorithm once so
``dsl.tf`` / ``dsl.T`` shrink to a dialect string (and, for ``T``, a
platform-sub-namespace pre-resolver, [parser §2](docs/spec/parser.md#2-syntax-and-rules)).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from tilefoundry.ir.core.op_registry import get_schemas, iter_schema_names
from tilefoundry.ir.core.overload import resolve

PreResolver = Callable[[str], Any]


def make_dialect_namespace(
    dialect: str,
    pre_resolvers: Iterable[PreResolver] = (),
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build dynamic attribute and directory hooks for one dialect namespace.

    Single schemas expose the Op class or alias builder; overloads expose a
    runtime resolver. ``__all__`` is computed on demand so later registrations
    remain visible.

    See [parser §2](docs/spec/parser.md#2-syntax-and-rules).
    """

    def __getattr__(name: str) -> Any:
        if name == "__all__":
            return sorted(iter_schema_names(dialect))
        for pre_resolve in pre_resolvers:
            resolved = pre_resolve(name)
            if resolved is not None:
                return resolved
        schemas = get_schemas(dialect, name)
        if not schemas:
            raise AttributeError(
                f"tilefoundry.dsl.{dialect} has no op named {name!r} "
                f"(did you forget to import the module that defines it?)"
            )

        first = schemas[0]
        if first.op_class is None:
            return first.builder

        if len(schemas) == 1:
            return first.op_class

        def _call(*args: Any, **kwargs: Any) -> Any:
            arg_types = tuple(getattr(a, "type", None) for a in args)
            chosen = resolve(schemas, arg_types)
            return chosen.builder(*args, **kwargs)

        _call.__name__ = name
        _call.__qualname__ = f"tilefoundry.dsl.{dialect}.{name}"
        _call.__doc__ = first.op_class.__doc__
        return _call

    def __dir__() -> list[str]:
        return sorted(iter_schema_names(dialect))

    return __getattr__, __dir__


__all__ = ["make_dialect_namespace"]
