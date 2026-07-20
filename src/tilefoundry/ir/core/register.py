"""``@register_op`` decorator.


Usage:

.. code-block:: python

    # Builtin op — auto-derive dialect / category from module path,
    # name from class name (lowercased).
    @register_op
    class ReLU(Op):
        x = ParamDef(kind="input", pattern=Tensor)

    # External / custom op — must give explicit dialect + category.
    @register_op(dialect="tf", category="custom")
    class MyOp(Op):
        ...

    # Override name (for overload disambiguation):
    @register_op(name="relu_v2")
    class ReLUVariant(Op):
        ...

Auto-derivation rules:

- ``cls.__module__`` matches ``tilefoundry.ir.<hir|tir>.<category>.*`` →
  ``dialect="tf"`` (hir) / ``"T"`` (tir); ``category`` = 4th segment.
- ``name`` defaults to ``cls.__name__.lower()`` (simple lowercase, no
  snake_case conversion).
- Outside the builtin path, ``dialect`` and ``category`` are required.

Validation:

- ``dialect`` must be ``"tf"`` or ``"T"``.
- ``category`` must be a non-empty string.
- ``name`` must be a non-empty string.

This decorator is the **only** way an Op enters the schema
registry; the legacy metaclass auto-register and textual annotation
parser have been removed.
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
    # Need at least: tilefoundry . ir . hir|tir . <category> . <file>
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
    derived_dialect, derived_category = _derive_dialect_and_category(
        getattr(cls, "__module__", "")
    )

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
            f"@register_op: dialect must be one of {_VALID_DIALECTS!r}, "
            f"got {final_dialect!r}"
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
    final_dialect, final_category, final_name = _validate_args(
        cls, dialect, category, name
    )
    signature = collect_param_defs(cls)
    return OpSchema(
        name=final_name,
        dialect=final_dialect,
        category=final_category,
        signature=signature,
        builder=cls,  # A1.b lock: v1 default builder = cls
        op_class=cls,
    )


# --- Decorator surface ---------------------------------------------------


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
        schema = _build_schema(
            target_cls, dialect=dialect, category=category, name=name
        )
        # Attach schema to class for later reflection (parser dispatch,
        # Op.params()). Underscore prefix = internal.
        setattr(target_cls, "_op_schema", schema)
        _register_schema(schema)
        return target_cls

    if cls is not None:
        # Bare-call form: @register_op
        return _apply(cls)
    # With-args form: @register_op(dialect="tf", ...)
    return _apply


# --- Surface alias decorator --------------------------------------------


def register_alias(
    *,
    dialect: str,
    category: str,
    name: str,
    params: "list[ParamDef] | tuple[ParamDef, ...]",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a **surface alias** schema — a callable name routed to a
    custom builder, without an IR Op class of its own.

    HIR math sugar names like ``add`` / ``cmp_eq`` previously each had a
    dedicated ``Op`` subclass that the parser then ``rebind_to_kinded``
    into ``Binary(kind=ADD)``. Aliases collapse that into a single
    schema-level routing entry: the surface name lives in the OpSchema
    main index, but its ``builder`` constructs the kinded target Op
    directly. The IR core ends up with just ``Binary`` / ``Unary``.

    The decorated function is the **builder**: it takes attribute
    kwargs only (input args go into ``Call.args`` separately, parser-
    side) and returns the concrete IR ``Op`` instance — e.g. for
    ``add``::

        @register_alias(dialect="tf", category="math", name="add",
                        params=[Binary.lhs, Binary.rhs])
        def _add() -> Op:
            return Binary(kind=BinaryKind.ADD)

    ``params`` is a list of *static* ``ParamDef`` references taken from
    the target Op (e.g. ``Binary.lhs``, ``Binary.rhs``). They define
    the alias' surface signature (used for ``.pyi`` stubs and parser
    overload matching) without re-declaring fresh ParamDef instances.

    Aliases prepend (not append) to the registry bucket so that
    schema-aware first-match resolution picks the alias over a
    legacy-named real-Op schema during the transition.
    """
    if not dialect or dialect not in _VALID_DIALECTS:
        raise ValueError(
            f"register_alias: dialect must be one of {_VALID_DIALECTS!r}, "
            f"got {dialect!r}"
        )
    if not category or not isinstance(category, str):
        raise ValueError(
            f"register_alias({name!r}): category must be a non-empty string"
        )
    if not name or not isinstance(name, str):
        raise ValueError(
            f"register_alias: name must be a non-empty string, got {name!r}"
        )
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
            op_class=None,  # alias has no IR class of its own
        )
        _register_schema(schema, prepend=True)
        # Stash the schema on the builder for tests / introspection.
        setattr(builder_fn, "_op_schema", schema)
        return builder_fn

    return _apply


__all__ = [
    "register_op",
    "register_alias",
    "_build_schema",
    "_derive_dialect_and_category",
]
