"""Register callable operations, aliases and exported types in a dialect tree.

Dialect and documentation category are explicit; namespace controls the DSL
path independently of the documentation category.
See [parser §2](docs/spec/parser.md#2-syntax-and-rules).
"""

from __future__ import annotations

from typing import Any, Callable

from tilefoundry.ir.core.op_registry import _VALID_DIALECTS, _register
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.core.param_def import ParamDef, collect_param_defs


def register_op(
    cls: type | None = None,
    *,
    dialect: str,
    category: str,
    name: str | None = None,
    namespace: str = "",
) -> Any:
    """Expose an operation at a unique dialect namespace and name."""
    if not isinstance(category, str) or not category:
        raise ValueError("register_op: category must be a non-empty string")

    def apply(target_cls: type) -> type:
        schema = OpSchema(
            name=name if name is not None else target_cls.__name__.lower(),
            dialect=dialect,
            category=category,
            signature=collect_param_defs(target_cls),
            builder=target_cls,
            op_class=target_cls,
            namespace=tuple(namespace.split(".")) if namespace else (),
        )
        _register(dialect, schema.namespace, schema.name, schema)
        target_cls._op_schema = schema
        return target_cls

    return apply(cls) if cls is not None else apply


def register_type(*, dialect: str, namespace: str) -> Callable[[type], type]:
    """Expose a declaration type without attaching operation metadata."""
    def apply(cls: type) -> type:
        _register(dialect, tuple(namespace.split(".")) if namespace else (), cls.__name__, cls)
        return cls
    return apply


def register_alias(
    *,
    dialect: str,
    category: str,
    name: str,
    params: "list[ParamDef] | tuple[ParamDef, ...]",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a surface name routed to a builder without its own IR class.

    ``params`` reuses static ``ParamDef`` descriptors from the target operation.
    The builder accepts attributes and returns the concrete operation. Aliases
    occupy their own unique name in the dialect tree.

    See [core-ir §2.3](docs/spec/core-ir.md#23-op).
    """
    if not dialect or dialect not in _VALID_DIALECTS:
        raise ValueError(
            f"register_alias: dialect must be one of {_VALID_DIALECTS!r}, got {dialect!r}"
        )
    if not category or not isinstance(category, str):
        raise ValueError(f"register_alias({name!r}): category must be a non-empty string")
    if not name or not isinstance(name, str):
        raise ValueError(f"register_alias: name must be a non-empty string, got {name!r}")
    sig = tuple(params)
    for pd in sig:
        if not isinstance(pd, ParamDef):
            raise TypeError(
                f"register_alias({name!r}): params must be a list of ParamDef "
                f"references (e.g. Binary.lhs), got {type(pd).__name__}"
            )

    def _apply(builder_fn: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(builder_fn):
            raise TypeError(
                f"register_alias({name!r}): builder must be callable, "
                f"got {type(builder_fn).__name__}"
            )
        schema = OpSchema(
            name=name,
            dialect=dialect,
            category=category,
            signature=sig,
            builder=builder_fn,
            op_class=None,
        )
        _register(dialect, (), name, schema)

        setattr(builder_fn, "_op_schema", schema)
        return builder_fn

    return _apply


__all__ = ["register_op", "register_alias", "register_type"]
