"""Register operations and declaration types under one class identity."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from tilefoundry.ir.core.op_registry import _CLASS_ALIASES, _CLASS_REGISTRY, _REGISTRY, _register
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.core.param_def import collect_param_defs

def get_registered_name(value: Any) -> str:
    """Return a canonical class name or the first applicable instance alias."""
    cls = value if isinstance(value, type) else type(value)
    key = _CLASS_REGISTRY.get(cls)
    entry = _REGISTRY.get(key)
    if entry is not cls and not (isinstance(entry, OpSchema) and entry.op_class is cls):
        raise ValueError(f"{cls.__module__}.{cls.__qualname__} has no registered identity")
    if not isinstance(value, type):
        for alias_key in _CLASS_ALIASES.get(cls, ()):
            entry = _REGISTRY.get(alias_key)
            if (isinstance(entry, OpSchema) and entry.alias_of is cls
                    and all(getattr(value, k, object()) == v for k, v in entry.fixed_attrs.items())):
                key = alias_key
                break
    dialect, namespace, name = key
    return ".".join((dialect, *namespace, name))


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
        if any(entry is target_cls or isinstance(entry, OpSchema) and entry.op_class is target_cls
               for entry in _REGISTRY.values()):
            raise ValueError(f"{get_registered_name(target_cls)} is already registered")
        path = tuple(namespace.split(".")) if namespace else ()
        signature = collect_param_defs(target_cls)
        schema = OpSchema(name=name if name is not None else target_cls.__name__.lower(),
                          dialect=dialect, category=category, signature=signature,
                          builder=target_cls, op_class=target_cls, namespace=path)
        entries = [schema]
        attribute_names = {pd.name for pd in signature if pd.kind == "attribute"}
        for alias_name, fixed in (aliases or {}).items():
            fixed = dict(fixed)
            if set(fixed) - attribute_names:
                raise ValueError(f"alias {alias_name!r} may only fix declared attributes")
            def make_builder(bound):
                def build(**attrs):
                    if set(attrs) & bound.keys():
                        raise TypeError("alias fixed attributes cannot be overridden")
                    return target_cls(**bound, **attrs)
                return build
            entries.append(OpSchema(name=alias_name, dialect=dialect, category=category,
                signature=tuple(pd for pd in signature if pd.name not in fixed),
                builder=make_builder(fixed), namespace=path, alias_of=target_cls, fixed_attrs=fixed))
        for entry in entries[1:]:
            entry.builder._op_schema = entry
        pending = dict(_REGISTRY)
        for entry in entries:
            _register(dialect, path, entry.name, entry, registry=pending)
        _REGISTRY.update(pending)
        _CLASS_REGISTRY[target_cls] = (dialect, path, schema.name)
        _CLASS_ALIASES[target_cls] = tuple((dialect, path, entry.name) for entry in entries[1:])
        target_cls._op_schema = schema
        return target_cls
    return apply(cls) if cls is not None else apply


def register_type(*, dialect: str, namespace: str) -> Callable[[type], type]:
    """Expose a declaration type using the same identity index as operations."""
    def apply(cls: type) -> type:
        if any(entry is cls or isinstance(entry, OpSchema) and entry.op_class is cls
               for entry in _REGISTRY.values()):
            raise ValueError(f"{get_registered_name(cls)} is already registered")
        path = tuple(namespace.split(".")) if namespace else ()
        _register(dialect, path, cls.__name__, cls)
        _CLASS_REGISTRY[cls] = (dialect, path, cls.__name__)
        return cls
    return apply


__all__ = ["register_op", "register_type", "get_registered_name"]
