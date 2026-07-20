"""ParamDef descriptor for Op callable signature.

is a class-body descriptor that declares a single parameter (input or
attribute) of an Op. Together with `OpSchema`, it replaces the older
`Param[Expr, "input"]` annotation form (still present in `op.py` until
M1c removes it).

Field semantics:

- ``kind``: ``"input"`` or ``"attribute"``.
- ``annotation``: Python type annotation that constrains the
  *family* of values this param accepts (``Expr`` / ``Layout`` /
  ``ShardLayout`` / ``int`` / ``str`` / ...). It drives:
  (a) parser sugar dispatch — e.g. ``annotation`` being a Layout-like
      type triggers layout sugar parsing at the corresponding
      call-arg position;
  (b) ``.pyi`` rendering for IDE / pyright;
  (c) coarse overload candidate split — schemas of the same op name
      whose annotations disagree don't compete on the same arg.
  It does NOT carry rank / shape / dtype constraints.
- ``pattern``: optional ``Pattern`` for IR-Expr subset filtering
  within the ``Expr`` annotation family (e.g. ``Tensor`` vs
  ``Scalar``, rank/shape/dtype). Non-overlapping with ``annotation``.
- ``optional``: nullable flag. ``True`` means the value may be ``None``
  (i.e. ``annotation | None`` in `.pyi`); orthogonal to omission.
- ``default``: call-site default. ``MISSING`` sentinel means required;
  any other value means the argument may be omitted at the call site
  (auto-treats as optional in the surface signature).

The class attribute name (``src``, ``dst``, ...) is captured via
``__set_name__`` and stored on the descriptor as ``_attr_name``; it is
the canonical parameter name and is not duplicated in a ``name`` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# --- Sentinel for "required" defaults ------------------------------------

class _MissingType:
    """Sentinel marker for required parameters (no default).

    Distinguished from ``None`` (which is a valid default for nullable
    params).
    """

    _instance: "_MissingType | None" = None

    def __new__(cls) -> "_MissingType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"

    def __bool__(self) -> bool:  # treat as falsy for convenience
        return False

MISSING: _MissingType = _MissingType()

_ParamKind = Literal["input", "attribute"]

@dataclass
class ParamDef:
    """Class-body descriptor for an Op parameter.

    Use as: ``src = ParamDef(kind="input", pattern=Tensor)``.

    The ``__set_name__`` hook records the attribute name on the
    descriptor instance for later reflection.
    """

    kind: _ParamKind
    annotation: type = field(default=object)  # filled in lazily; default Expr in op_schema layer
    pattern: "Pattern | None" = None
    optional: bool = False
    default: Any = MISSING

    # Filled by __set_name__; not in __init__ args.
    _attr_name: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate kind.
        if self.kind not in ("input", "attribute"):
            raise ValueError(
                f"ParamDef.kind must be 'input' or 'attribute', got {self.kind!r}"
            )
        # `default != MISSING` auto-treats as optional in the surface
        # signature: it implies the param may be omitted at the call
        # site. We preserve `optional`'s original (nullable) semantics
        # but allow the convenience that writing `default=v` is enough
        # to mark "omittable". `optional=True` independently controls
        # whether the value type is `T | None`.
        # No mutation here — call sites that interpret the descriptor
        # (parser, stub gen) should consult both fields.

    def __set_name__(self, owner: type, name: str) -> None:
        # Record canonical attribute name; do not overwrite if explicitly set.
        if not self._attr_name:
            self._attr_name = name

    @property
    def name(self) -> str:
        """Canonical parameter name (from class attribute name)."""
        return self._attr_name

    @property
    def is_required(self) -> bool:
        """True iff the call site must supply this argument."""
        return self.default is MISSING

    @property
    def has_default(self) -> bool:
        """True iff a call-site default is configured."""
        return self.default is not MISSING


def collect_param_defs(cls: type) -> tuple["ParamDef", ...]:
    """Reflect ``ParamDef`` class-body descriptors off ``cls`` in MRO order.

    Walks base→derived so subclass fields come after base fields in
    declaration order. A field redeclared at a more-derived class
    overrides the base entry in place (the base's position in the
    result is preserved). Attribute names starting with ``_`` are
    skipped (private / internal, not part of the callable signature).
    """
    seen: dict[str, ParamDef] = {}
    order: list[str] = []
    for klass in reversed(cls.__mro__):
        for attr_name, value in klass.__dict__.items():
            if attr_name.startswith("_"):
                continue
            if not isinstance(value, ParamDef):
                continue
            if attr_name not in seen:
                order.append(attr_name)
            seen[attr_name] = value
    return tuple(seen[name] for name in order)


__all__ = ["ParamDef", "MISSING", "_MissingType", "collect_param_defs"]
