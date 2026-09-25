"""Describe one input or attribute in an ``OpSchema`` signature.

Annotations drive surface typing, sugar, and coarse overload selection;
patterns further filter IR values. ``optional`` permits ``None`` while a
non-``MISSING`` default permits omission. ``__set_name__`` supplies the
canonical parameter name.

See [parser §2](docs/spec/parser.md#2-syntax-and-rules).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Literal


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

    def __bool__(self) -> bool:
        return False


MISSING: _MissingType = _MissingType()

_ParamKind = Literal["input", "attribute"]


class MemoryEffect(Flag):
    """Storage effects declared by one tensor operand."""

    READ = auto()
    WRITE = auto()


@dataclass
class ParamDef:
    """Class-body descriptor for an Op parameter.

    Use as: ``src = ParamDef(kind="input", pattern=Tensor)``.

    The ``__set_name__`` hook records the attribute name on the
    descriptor instance for later reflection. ``effect=None`` means the
    operand's memory effect is undeclared, not effect-free.
    """

    kind: _ParamKind
    annotation: type = field(default=object)
    pattern: "Pattern | None" = None
    optional: bool = False
    default: Any = MISSING
    effect: MemoryEffect | None = None

    _attr_name: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in ("input", "attribute"):
            raise ValueError(f"ParamDef.kind must be 'input' or 'attribute', got {self.kind!r}")
        if self.effect is not None and not isinstance(self.effect, MemoryEffect):
            raise TypeError("effect must be a MemoryEffect flag or None (undeclared)")
        if self.kind == "attribute" and self.effect:
            raise ValueError("attributes do not read or write tensor storage")
        if self.kind == "input" and self.effect == MemoryEffect(0):
            raise ValueError("tensor operands must declare READ, WRITE, or both")

    def __set_name__(self, owner: type, name: str) -> None:

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


__all__ = [
    "MISSING",
    "MemoryEffect",
    "ParamDef",
    "_MissingType",
    "collect_param_defs",
]
