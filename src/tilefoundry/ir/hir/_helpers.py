"""Shared HIR-wide helpers (cross-category), not tied to any one op family."""

from __future__ import annotations

from tilefoundry.ir.types.storage import StorageKind


def resolve_anchor_storage(ctx, call, *storages):
    """Resolve output storage from the concrete residency among the operands.

    ``StorageKind.UMAT`` abstains. One concrete storage, or several that agree,
    anchors the output. Disagreement is unsupported; there is no operand-order
    tie-break. With no concrete anchor, the result is ``StorageKind.UMAT``.
    """
    concrete = {s for s in storages if s is not StorageKind.UMAT}
    if not concrete:
        return StorageKind.UMAT
    if len(concrete) == 1:
        return next(iter(concrete))
    kinds = ", ".join(sorted(str(s) for s in concrete))
    ctx.error(
        call,
        f"operands have conflicting storage ({kinds}); a multi-input op "
        f"requires its concrete operands to share one residency",
    )


__all__ = ["resolve_anchor_storage"]
