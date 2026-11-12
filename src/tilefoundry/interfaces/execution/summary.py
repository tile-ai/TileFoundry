"""Conservative kernel parameter accesses retained through kernel completion."""

from dataclasses import dataclass

from tilefoundry.ir import expr


@dataclass(frozen=True)
class Summary:
    """Whole-parameter accesses of a kernel with ordinary local memory effects.

    Parameter indices preserve the callee's signature. Views retain their base
    parameter conservatively; no subregion readiness or remote completion is
    inferred from a local instruction's read/write declarations.
    """

    accesses: tuple[tuple[int, expr.MemoryEffect], ...]


__all__ = ["Summary"]
