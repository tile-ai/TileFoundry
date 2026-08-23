"""Register operation classes and surface aliases as ``OpSchema`` entries.

Builtin module paths derive dialect and category; external operations provide
them explicitly. Names default to the lowercase class name. Registration is
the only route into the callable schema registry.

See [parser §2](docs/spec/parser.md#2-syntax-and-rules).
"""

from __future__ import annotations

from typing import Any, Callable, overload

from tilefoundry.ir.core.op_registry import _VALID_DIALECTS, _register_schema
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.core.param_def import ParamDef, collect_param_defs


def _derive_dialect_and_category(module: str) -> tuple[str | None, str | None]:
    """Try to derive ``(dialect, category)`` from ``cls.__module__``.

    Returns ``(None, None)`` if the path doesn't match the builtin
    convention ``tilefoundry.ir.<hir|tir>.<category>.*``.
    """
    if not module:
        return None, None
    parts = module.split(".")

    if len(parts) < 5:
        return None, None
    if parts[0] != "tilefoundry" or parts[1] != "ir":
        return None, None
    seg = parts[2]
    if seg == "hir":
        return "tf", parts[3]
    if seg == "tir":
        return "T", parts[3]
    return None, None


def _validate_args(
    cls: type, dialect: str | None, category: str | None, name: str | None
) -> tuple[str, str, str]:
    """Resolve and validate the (dialect, category, name) triple.

    Auto-derives missing pieces from the module path; raises if the
    builtin path doesn't apply and explicit args weren't supplied.
    """
    derived_dialect, derived_category = _derive_dialect_and_category(getattr(cls, "__module__", ""))

    final_dialect = dialect if dialect is not None else derived_dialect
    final_category = category if category is not None else derived_category
    final_name = name if name is not None else cls.__name__.lower()

    if final_dialect is None:
        raise ValueError(
            f"@register_op({cls.__module__}.{cls.__name__}): cannot auto-derive "
            f"dialect from module path; pass `dialect='tf'` or `dialect='T'` "
            f"explicitly."
        )
    if final_dialect not in _VALID_DIALECTS:
        raise ValueError(
            f"@register_op: dialect must be one of {_VALID_DIALECTS!r}, got {final_dialect!r}"
        )

    if not final_category or not isinstance(final_category, str):
        raise ValueError(
            f"@register_op({cls.__module__}.{cls.__name__}): cannot auto-derive "
            f"category from module path; pass `category=...` explicitly."
        )

    if not final_name or not isinstance(final_name, str):
        raise ValueError(
            f"@register_op({cls.__module__}.{cls.__name__}): name must be a "
            f"non-empty string, got {final_name!r}"
        )

    return final_dialect, final_category, final_name


def _build_schema(
    cls: type,
    *,
    dialect: str | None = None,
    category: str | None = None,
    name: str | None = None,
) -> OpSchema:
    """Build an OpSchema for ``cls`` (no registration side-effect)."""
    final_dialect, final_category, final_name = _validate_args(cls, dialect, category, name)
    signature = collect_param_defs(cls)
    return OpSchema(
        name=final_name,
        dialect=final_dialect,
        category=final_category,
        signature=signature,
        builder=cls,
        op_class=cls,
    )


@overload
def register_op(cls: type) -> type: ...
@overload
def register_op(
    *,
    dialect: str | None = None,
    category: str | None = None,
    name: str | None = None,
) -> Callable[[type], type]: ...


def register_op(
    cls: type | None = None,
    *,
    dialect: str | None = None,
    category: str | None = None,
    name: str | None = None,
) -> Any:
    """Register ``cls`` as a callable Op into the OpSchema registry.

    Two call styles, mirroring ``dataclass``:

    - Bare: ``@register_op`` — builtin path; auto-derive everything.
    - With args: ``@register_op(dialect=..., category=..., name=...)`` —
      explicit overrides for non-builtin / overload disambiguation.
    """

    def _apply(target_cls: type) -> type:
        schema = _build_schema(target_cls, dialect=dialect, category=category, name=name)

        setattr(target_cls, "_op_schema", schema)
        _register_schema(schema)
        return target_cls

    if cls is not None:
        return _apply(cls)

    return _apply


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
    prepend to their bucket and therefore win first-match resolution.

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
        _register_schema(schema, prepend=True)

        setattr(builder_fn, "_op_schema", schema)
        return builder_fn

    return _apply


__all__ = [
    "register_op",
    "register_alias",
    "_build_schema",
    "_derive_dialect_and_category",
]
