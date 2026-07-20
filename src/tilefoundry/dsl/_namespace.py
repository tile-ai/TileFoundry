"""Shared ``__getattr__`` / ``__dir__`` factory for the ``tf`` / ``T`` DSL
namespace modules (parser.md §2.1-§2.3).

Both dialect namespaces resolve names on demand against the OpSchema
registry with the same algorithm; this module ships that algorithm once so
``dsl.tf`` / ``dsl.T`` shrink to a dialect string (and, for ``T``, a
platform-sub-namespace pre-resolver, §2.6).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from tilefoundry.ir.core.op_registry import get_schemas, iter_schema_names
from tilefoundry.parser.overload import resolve

# Tried in order, before the OpSchema registry lookup; returns the resolved
# value or ``None`` to fall through (e.g. T's platform sub-namespaces, §2.6).
PreResolver = Callable[[str], Any]


def make_dialect_namespace(
    dialect: str, pre_resolvers: Iterable[PreResolver] = (),
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build the ``(__getattr__, __dir__)`` pair for a dialect namespace module.

    For single-schema names the real-Op **class** is returned directly, so
    ``add = tf.add`` / ``from tilefoundry.dsl.tf import add`` binds the actual
    Op subclass. Surface-alias schemas (``schemas[0].op_class is None``)
    return the alias builder function instead — it still carries
    ``_op_schema`` (set by ``@register_alias``), so the parser's
    ``_schema_from_value`` recognises it the same way. Multi-schema
    overloads return a best-effort runtime resolver (real parser-time
    dispatch goes through :mod:`tilefoundry.parser.overload` directly).
    Unknown names raise :class:`AttributeError`.

    ``__all__`` is resolved on demand (not a frozen module attribute) so
    ``from tilefoundry.dsl.<dialect> import *`` sees Ops registered after
    the namespace module was first imported (test-fixture custom ops,
    lazy-loaded modules, etc.).
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

        # Surface aliases (``schema.op_class is None``) prepend to the
        # bucket so they win first-match; return the alias builder so an
        # alias schema wins over a legacy real-Op schema of the same name
        # even when both are registered (a transitional state).
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
