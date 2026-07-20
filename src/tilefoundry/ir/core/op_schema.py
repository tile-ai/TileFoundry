"""OpSchema: normalized registry data structure for a callable Op.

is the single source of metadata that parser, surface generator, error
formatter, and overload resolver consume.

Per A3.6 (Path-B) the same OpSchema covers HIR Ops (``dialect="tf"``)
and TIR Ops (``dialect="t"``); HIR/TIR distinction is dialect-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tilefoundry.ir.core.param_def import ParamDef


@dataclass(frozen=True)
class OpSchema:
    """Per-op metadata captured at ``@register_op`` time.

    Fields:

    - ``name``: lowercase callable name. Auto-derived from ``cls.__name__``
      unless overridden via ``@register_op(name=...)``.
    - ``dialect``: ``"tf"`` (HIR) or ``"T"`` (TIR).
    - ``category``: organizational grouping (``"nn"`` / ``"tensor"`` /
      ``"sharding"`` / etc.). Used for docs and IDE grouping; not in surface
      path (A2.4 flatten lock).
    - ``signature``: ordered tuple of :class:`ParamDef` in class-body
      definition order (A1.a lock).
    - ``builder``: callable producing an IR node from bound args. v1 default
      is the Op class itself (`cls`), so calling ``builder(**bound_args)``
      yields an instance.
    - ``op_class``: the original Op class, kept for repr and Path-B base
      class introspection. ``None`` for surface-alias schemas registered
      via ``@register_alias`` — those schemas have no IR class of their
      own; their ``builder`` constructs a *target* Op (e.g. ``Binary``
      with a fixed ``kind``) instead.
    """

    name: str
    dialect: str
    category: str
    signature: tuple[ParamDef, ...]
    builder: Callable[..., Any]
    op_class: type | None = None

    @property
    def is_alias(self) -> bool:
        """True iff this schema is a surface alias (no IR Op class)."""
        return self.op_class is None


__all__ = ["OpSchema"]
