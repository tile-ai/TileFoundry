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
    """Store the normalized callable metadata captured at registration.

    ``signature`` preserves declaration order. ``builder`` constructs the IR
    node; aliases have no ``op_class`` and may build another operation type.
    ``category`` organizes documentation but is not part of the surface path.

    See [parser §2](docs/spec/parser.md#2-syntax-and-rules).
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
