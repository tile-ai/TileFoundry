"""Resolve runtime values against authored loop axes."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Call, Constant, Expr
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import window_base


@dataclass(frozen=True)
class LoopAffineTerm:
    """One authored loop coefficient plus a compile-time offset interval."""

    loop_axis: int | None
    stride: int
    low: int
    high: int


def _constant_int(expr: Expr) -> int | None:
    if (
        isinstance(expr, Constant)
        and isinstance(expr.value, int)
        and not isinstance(expr.value, bool)
    ):
        return int(expr.value)
    return None


def static_range(expr: Expr, *, narrow: bool) -> tuple[int, int] | None:
    """Bound one mesh-position expression, or erase its fixed local translation."""
    value = _constant_int(expr)
    if value is not None:
        return value, value
    if not isinstance(expr, Call):
        return None
    if isinstance(expr.target, Local):
        return (0, 0) if narrow else static_range(expr.args[0], narrow=narrow)
    if isinstance(expr.target, (Reshape, Reshard)):
        return static_range(expr.args[0], narrow=narrow)
    if isinstance(expr.target, Arange):
        start, step = expr.target.start, expr.target.step
        (length,) = expr.target.type.shape
        if not all(
            isinstance(item, int) and not isinstance(item, bool) for item in (start, step, length)
        ):
            return None
        if length <= 0:
            return 0, 0
        return start, start + (length - 1) * step
    if isinstance(expr.target, Binary):
        left = static_range(expr.args[0], narrow=narrow)
        right = static_range(expr.args[1], narrow=narrow)
        if left is None or right is None:
            return None
        if expr.target.kind is BinaryKind.ADD:
            return left[0] + right[0], left[1] + right[1]
        if expr.target.kind is BinaryKind.MUL:
            products = tuple(a * b for a in left for b in right)
            return min(products), max(products)
    return None


def _loop_term(
    value: Expr, loops: tuple[GridRegionExpr, ...], *, narrow: bool
) -> LoopAffineTerm | None:
    for index, loop in enumerate(loops):
        if loop.induction_var is value:
            return LoopAffineTerm(index, 1, 0, 0)
    if not isinstance(value, Call) or not isinstance(value.target, Binary):
        return None
    if value.target.kind is BinaryKind.ADD:
        for candidate, invariant in (
            (value.args[0], value.args[1]),
            (value.args[1], value.args[0]),
        ):
            term = _loop_term(candidate, loops, narrow=narrow)
            bounds = static_range(invariant, narrow=narrow)
            if term is not None and bounds is not None:
                return LoopAffineTerm(
                    term.loop_axis,
                    term.stride,
                    term.low + bounds[0],
                    term.high + bounds[1],
                )
    if value.target.kind is BinaryKind.MUL:
        for candidate, invariant in (
            (value.args[0], value.args[1]),
            (value.args[1], value.args[0]),
        ):
            term = _loop_term(candidate, loops, narrow=narrow)
            bounds = static_range(invariant, narrow=narrow)
            if term is not None and bounds is not None and bounds[0] == bounds[1]:
                factor = bounds[0]
                offsets = (term.low * factor, term.high * factor)
                return LoopAffineTerm(
                    term.loop_axis,
                    term.stride * factor,
                    min(offsets),
                    max(offsets),
                )
    return None


def loop_affine_term(
    value: Expr, loops: tuple[GridRegionExpr, ...], *, narrow: bool
) -> LoopAffineTerm | None:
    """Resolve a constant, loop variable, affine offset, or constant stride."""
    base, offset = window_base(value)
    if base is None:
        return LoopAffineTerm(None, 0, offset, offset)
    term = _loop_term(base, loops, narrow=narrow)
    if term is not None:
        return LoopAffineTerm(
            term.loop_axis,
            term.stride,
            term.low + offset,
            term.high + offset,
        )
    bounds = static_range(base, narrow=narrow)
    if bounds is None:
        return None
    return LoopAffineTerm(None, 0, bounds[0] + offset, bounds[1] + offset)


__all__ = ["LoopAffineTerm", "loop_affine_term"]
