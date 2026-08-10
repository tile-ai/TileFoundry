"""Declarative predicates for overload and specialization dispatch.

``ParamDef.pattern`` filters parser overloads; ``DimVarRangePat`` selects HIR
specializations. Patterns do not participate in static type checking.

See [core-ir §3](docs/spec/core-ir.md#3-pattern) and
[hir §2](docs/spec/hir.md#2-function-specialization-api).
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
    dtype: Any = None

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
    """Match ``lo <= value < hi`` for a named specialization dimension.

    ``dim_var`` identifies the runtime shape source but is not inspected by
    :meth:`match`, which receives only the scalar value.

    See [core-ir §3.1](docs/spec/core-ir.md#31-dimvarrangepat).
    """

    dim_var: str = ""
    lo: int = 0
    hi: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.dim_var, str) or not self.dim_var:
            raise ValueError(
                f"DimVarRangePat: dim_var must be a non-empty str, got {self.dim_var!r}"
            )
        if not isinstance(self.lo, int) or isinstance(self.lo, bool):
            raise TypeError(f"DimVarRangePat: lo must be int, got {type(self.lo).__name__}")
        if not isinstance(self.hi, int) or isinstance(self.hi, bool):
            raise TypeError(f"DimVarRangePat: hi must be int, got {type(self.hi).__name__}")
        if self.lo >= self.hi:
            raise ValueError(
                f"DimVarRangePat({self.dim_var!r}, {self.lo}, {self.hi}): "
                f"requires lo < hi (half-open [lo, hi); single point is [k, k+1))"
            )

    def match(self, subject: Any) -> bool:
        if isinstance(subject, bool) or not isinstance(subject, int):
            return False
        return self.lo <= subject < self.hi


def locate_dim_var(params: tuple, name: str) -> tuple[int, int] | None:
    """First ``(param_index, axis)`` where a ``DimVar`` named *name* appears in *params*' shapes.

    First ``(param_index, axis)`` where a ``DimVar`` named *name* appears in
    *params*' shapes.

    Canonical scan order is ``(param_index ascending, axis ascending)`` — the
    single dispatch-subject rule shared by HIR→TIR lowering and the reference
    evaluator's variant selection.
    """
    for i, p in enumerate(params):
        shape = getattr(p.type, "shape", None)
        if shape is None:
            continue
        for axis, dim in enumerate(shape):
            if getattr(dim, "name", None) == name:
                return (i, axis)
    return None


Scalar: ScalarPat = ScalarPat()


Tensor: TensorPat = TensorPat()


__all__ = [
    "Pattern",
    "ScalarPat",
    "TensorPat",
    "AndPat",
    "DimVarRangePat",
    "Scalar",
    "Tensor",
    "locate_dim_var",
]
