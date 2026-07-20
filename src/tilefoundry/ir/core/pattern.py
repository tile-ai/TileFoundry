"""Pattern: declarative predicate over IR types and shape values.

Pattern instances are reusable predicates. They serve two consumers:

1. Parser dispatch / overload resolution: ``ParamDef.pattern`` is
   matched against a ``Type`` (typically ``TensorType`` /
   ``TupleType``). Subclasses used here: ``ScalarPat`` /
   ``TensorPat`` / ``AndPat``.

2. Specialization dispatch: patterns appearing in
   ``Function.specializations`` describe which runtime shape range a
   specialized function body covers. Subclass: ``DimVarRangePat``.
   HIR→TIR lowering dispatches on the pattern subclass internally to
   emit a runtime predicate.

Patterns do not participate in pyright type checking.

``OrPat`` / named patterns / operator sugar are deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Pattern:
    """Base. A reusable predicate.

    Subclasses override :meth:`match` to express their constraint.
    The ``subject`` of :meth:`match` depends on the consumer: parser
    dispatch passes an IR ``Type``; specialization-dispatch lowering
    inspects the pattern's own fields and does not call :meth:`match`.
    """

    def match(self, subject: Any) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class ScalarPat(Pattern):
    """Matches rank-0 tensor (``shape == ()``)."""

    def match(self, subject: Any) -> bool:
        shape = getattr(subject, "shape", None)
        return shape == ()


@dataclass(frozen=True)
class TensorPat(Pattern):
    """Matches non-scalar tensor (``shape != ()``).

    Optional ``rank`` and ``dtype`` further constrain the shape length
    and dtype. Default (no constraints) matches any non-scalar tensor.
    """

    rank: int | None = None
    dtype: Any = None  # DType | None — kept Any to avoid hard import cycle

    def match(self, subject: Any) -> bool:
        shape = getattr(subject, "shape", None)
        if shape is None:
            return False
        if shape == ():
            return False
        if self.rank is not None and len(shape) != self.rank:
            return False
        if self.dtype is not None:
            ty_dtype = getattr(subject, "dtype", None)
            if ty_dtype != self.dtype:
                return False
        return True


@dataclass(frozen=True)
class AndPat(Pattern):
    """All children must match."""

    parts: tuple[Pattern, ...] = field(default_factory=tuple)

    def match(self, subject: Any) -> bool:
        return all(p.match(subject) for p in self.parts)


@dataclass(frozen=True)
class DimVarRangePat(Pattern):
    """Half-open range predicate over a named ``DimVar``.

    ``DimVarRangePat("S", lo, hi)`` matches an integer ``value`` iff
    ``lo <= value < hi`` (``lo`` inclusive, ``hi`` exclusive). A
    single-point range is ``[k, k+1)`` (matches exactly ``k``).

    Used in ``Function.specializations`` to declare which runtime
    range of ``DimVar`` a specialization covers. The ``dim_var`` field
    carries the name of the ``DimVar`` the range applies to — it does
    not participate in :meth:`match`, which only takes a scalar value.
    The HIR→TIR lowering resolves the named ``DimVar`` to a runtime
    shape source (``ShapeOf(param, axis)``) by walking the enclosing
    function's signature.
    """

    dim_var: str = ""
    lo: int = 0
    hi: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.dim_var, str) or not self.dim_var:
            raise ValueError(
                f"DimVarRangePat: dim_var must be a non-empty str, got "
                f"{self.dim_var!r}"
            )
        if not isinstance(self.lo, int) or isinstance(self.lo, bool):
            raise TypeError(
                f"DimVarRangePat: lo must be int, got {type(self.lo).__name__}"
            )
        if not isinstance(self.hi, int) or isinstance(self.hi, bool):
            raise TypeError(
                f"DimVarRangePat: hi must be int, got {type(self.hi).__name__}"
            )
        if self.lo >= self.hi:
            raise ValueError(
                f"DimVarRangePat({self.dim_var!r}, {self.lo}, {self.hi}): "
                f"requires lo < hi (half-open [lo, hi); single point is [k, k+1))"
            )

    def match(self, subject: Any) -> bool:
        if isinstance(subject, bool) or not isinstance(subject, int):
            return False
        return self.lo <= subject < self.hi


# --- Convenience singletons / aliases ------------------------------------

#: Singleton matching any rank-0 (scalar) tensor.
Scalar: ScalarPat = ScalarPat()

#: Singleton matching any non-scalar tensor (rank >= 1).
Tensor: TensorPat = TensorPat()


__all__ = [
    "Pattern",
    "ScalarPat",
    "TensorPat",
    "AndPat",
    "DimVarRangePat",
    "Scalar",
    "Tensor",
]
