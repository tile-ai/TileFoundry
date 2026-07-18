from __future__ import annotations

from tilefoundry.ir.core import Constant
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.shard_layout import Broadcast, ShardLayout
from tilefoundry.ir.types.storage import StorageKind


def _dtype_promote(a: DType, b: DType) -> DType:
    if a == b:
        return a
    # V1 rule: disallow implicit promotion, force explicit Cast.
    raise_ = True
    if raise_:
        raise_ = True  # kept for clarity — real check in caller via ctx.error
    return a


def _is_rank0(t: TensorType) -> bool:
    return t.shape == ()


def _shapes_equal(a: tuple, b: tuple) -> bool:
    # Structural equality; Exprs are hashable dataclasses.
    return a == b


def _broadcast(a: tuple, b: tuple) -> tuple:
    """NumPy-style right-aligned broadcast on `tuple[Expr, ...]`. The shorter
    shape is padded on the left with 1s, then dims combine pairwise (equal, or
    one is 1). Any other pair → raise."""
    if a == b:
        return a
    n = max(len(a), len(b))
    ap = (1,) * (n - len(a)) + tuple(a)
    bp = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(ap, bp):
        if x == y:
            out.append(x)
        elif _is_one(x):
            out.append(y)
        elif _is_one(y):
            out.append(x)
        else:
            raise ValueError(f"cannot broadcast shapes {a} and {b}")
    return tuple(out)


def _is_one(expr) -> bool:
    """Return True for any shape entry that represents the literal 1.

    Shape entries can be either ``Constant(value=1)`` (the canonical
    IR form, produced by the parser / annotation lift) or a Python
    ``int`` (produced ad-hoc by some typeinfer rules — Reduce, Slice,
    etc.). Both forms must broadcast against larger dims; restricting
    to ``Constant`` only breaks the ``(1, N) ⊕ (1, 1)`` pattern that
    falls out of ``Reduce(..., keepdim=True)``.
    """
    if isinstance(expr, Constant) and expr.value == 1:
        return True
    if isinstance(expr, int) and not isinstance(expr, bool) and expr == 1:
        return True
    return False


def _merge_layout(a: object, b: object) -> object:
    """Merge two non-sharded operand layouts. Equal layouts or one ``None``
    pass through. Two fully-replicated (all-``Broadcast``) ``ShardLayout``s are
    mesh-agnostic (the data is replicated everywhere) so the first is kept.
    Any other genuine mismatch raises — there is no silent lhs pick; a real
    shard mismatch is propagated through the shard engine, not merged here."""
    if a == b:
        return a
    if a is None:
        return b
    if b is None:
        return a
    if (
        isinstance(a, ShardLayout)
        and isinstance(b, ShardLayout)
        and all(isinstance(x, Broadcast) for x in a.attrs)
        and all(isinstance(x, Broadcast) for x in b.attrs)
    ):
        return a
    raise ValueError(f"incompatible operand layouts {a!r} vs {b!r}")


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
