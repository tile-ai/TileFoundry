"""Shared HIR-wide helpers (cross-category), not tied to any one op family."""
from __future__ import annotations

from tilefoundry.ir.core import Constant
from tilefoundry.ir.types.storage import StorageKind


def is_one(expr) -> bool:
    """Return True for any shape entry that represents the literal 1.

    Shape entries can be either ``Constant(value=1)`` (the canonical IR
    form, produced by the parser / annotation lift) or a Python ``int``
    (produced ad-hoc by some typeinfer rules — Reduce, Slice, etc.). Both
    forms must broadcast against larger dims; restricting to ``Constant``
    only breaks the ``(1, N) ⊕ (1, 1)`` pattern that falls out of
    ``Reduce(..., keepdim=True)``.
    """
    if isinstance(expr, Constant) and expr.value == 1:
        return True
    if isinstance(expr, int) and not isinstance(expr, bool) and expr == 1:
        return True
    return False


def broadcast_shapes(a: tuple, b: tuple, *, raising: bool = True):
    """NumPy-style right-aligned broadcast on ``tuple[Expr, ...]``.

    The shorter shape is padded on the left with 1s, then dims combine
    pairwise (equal, or one is 1). For an incompatible pair: raises
    ``ValueError`` when *raising* (the default), else returns ``None``.
    """
    if a == b:
        return a
    n = max(len(a), len(b))
    ap = (1,) * (n - len(a)) + tuple(a)
    bp = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(ap, bp):
        if x == y:
            out.append(x)
        elif is_one(x):
            out.append(y)
        elif is_one(y):
            out.append(x)
        elif raising:
            raise ValueError(f"cannot broadcast shapes {a} and {b}")
        else:
            return None
    return tuple(out)


def resolve_anchor_storage(ctx, call, *storages):
    """Resolve a multi-input op's output storage by anchoring on the concrete
    residency present among its operands.

    An unmaterialized operand (``StorageKind.UMAT``) abstains. A single concrete
    storage — or several that agree — is the anchor and becomes the output.
    Concrete operands that disagree are unsupported (the op has no destination/
    mixed-storage rule); there is no operand-order tie-break. All-unmaterialized
    yields ``StorageKind.UMAT``.
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


__all__ = ["is_one", "broadcast_shapes", "resolve_anchor_storage"]
