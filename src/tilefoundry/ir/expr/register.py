"""Register operations and declaration types under one class identity."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .op_registry import registry
from .op_schema import OpSchema
from .param_def import collect_param_defs

__all__ = ["get_registered_name", "register_op", "register_type"]


def get_registered_name(value: Any) -> str:
    """Return a canonical class name or the first applicable instance alias."""
    cls = value if isinstance(value, type) else type(value)
    key = registry.class_keys.get(cls)
    entry = registry.entries.get(key)
    if entry is not cls and not (isinstance(entry, OpSchema) and entry.op_class is cls):
        raise ValueError(f"{cls.__module__}.{cls.__qualname__} has no registered identity")
    if not isinstance(value, type):
        for alias_key in registry.class_aliases.get(cls, ()):
            entry = registry.entries.get(alias_key)
            if (isinstance(entry, OpSchema) and entry.alias_of is cls
                    and all(getattr(value, k, object()) == v for k, v in entry.fixed_attrs.items())):
                key = alias_key
                break
    dialect, namespace, name = key
    return ".".join((dialect, *namespace, name))


def alias_builder(target_cls: type, bound: Mapping[str, Any]) -> Callable[..., Any]:
    """Build *target_cls* with the attributes of an alias already fixed."""

    def build(**attrs):
        if set(attrs) & bound.keys():
            raise TypeError("alias fixed attributes cannot be overridden")
        return target_cls(**bound, **attrs)

    return build


def register_op(
    cls: type | None = None,
    *,
    dialect: str,
    category: str,
    namespace: str = "",
    name: str | None = None,
    aliases: Mapping[str, Mapping[str, Any]] | None = None,
) -> Any:
    """Expose an operation and fixed-attribute aliases atomically."""
    if not isinstance(category, str) or not category:
        raise ValueError("register_op: category must be a non-empty string")

    def apply(target_cls: type) -> type:
        if registry.owns(target_cls):
            raise ValueError(f"{get_registered_name(target_cls)} is already registered")
        path = tuple(namespace.split(".")) if namespace else ()
        signature = collect_param_defs(target_cls)
        schema = OpSchema(name=name if name is not None else target_cls.__name__.lower(),
                          dialect=dialect, category=category, signature=signature,
                          builder=target_cls, op_class=target_cls, namespace=path)
        schemas = [schema]
        attribute_names = {pd.name for pd in signature if pd.kind == "attribute"}
        for alias_name, fixed in (aliases or {}).items():
            fixed = dict(fixed)
            if set(fixed) - attribute_names:
                raise ValueError(f"alias {alias_name!r} may only fix declared attributes")
            schemas.append(OpSchema(name=alias_name, dialect=dialect, category=category,
                signature=tuple(pd for pd in signature if pd.name not in fixed),
                builder=alias_builder(target_cls, fixed), namespace=path, alias_of=target_cls, fixed_attrs=fixed))
        registry.add_operation(dialect, path, target_cls, schemas)
        return target_cls
    return apply(cls) if cls is not None else apply


def register_type(*, dialect: str, namespace: str) -> Callable[[type], type]:
    """Expose a declaration type using the same identity index as operations."""
    def apply(cls: type) -> type:
        if registry.owns(cls):
            raise ValueError(f"{get_registered_name(cls)} is already registered")
        path = tuple(namespace.split(".")) if namespace else ()
        registry.add_type(dialect, path, cls)
        return cls
    return apply
