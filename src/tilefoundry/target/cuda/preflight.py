"""Private CTA v1 input validation."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Call, Constant, Expr, Tuple
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.target import Target

from .target import CudaTarget


@dataclass(frozen=True)
class CtaPreflightResult:
    """Validated CTA input facts retained for the next private stage."""

    root: Function
    cta_count: int
    reachable_functions: tuple[Function, ...]

    @property
    def functions(self) -> tuple[Function, ...]:
        """Compatibility alias for the reachable HIR function set."""
        return self.reachable_functions


def _static_int(value: object) -> int | None:
    if isinstance(value, Constant):
        value = value.value
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _root_from(subject: Function | Module, explicit_root: Function | None) -> Function:
    if explicit_root is not None:
        return explicit_root
    if isinstance(subject, Module):
        entry = subject.entry_function()
    else:
        entry = subject
    if not isinstance(entry, Function):
        raise TypeError(
            f"cta preflight root must be a HIR Function, got "
            f"{type(entry).__name__}"
        )
    return entry


def _context(root: Function, owner: Function, detail: str) -> str:
    return (
        f"cta preflight root {root.name!r}, function {owner.name!r}: "
        f"{detail}"
    )


def _validate_root_topology(root: Function) -> tuple[CudaTarget, int]:
    target = root.target
    if target is None:
        raise ValueError(
            f"cta preflight root {root.name!r} has no explicit CUDA Target; "
            "CTA v1 does not resolve an omitted root target"
        )
    if not isinstance(target, CudaTarget):
        raise ValueError(
            f"cta preflight root {root.name!r} requires CudaTarget, got "
            f"{type(target).__name__}"
        )
    cta = tuple(topology for topology in root.topologies if topology.name == "cta")
    if len(cta) != 1:
        raise ValueError(
            f"cta preflight root {root.name!r} requires exactly one static "
            f"Topology('cta', n), found {len(cta)} in {root.topologies!r}"
        )
    count = _static_int(cta[0].size)
    if count is None:
        raise ValueError(
            f"cta preflight root {root.name!r} has dynamic CTA extent "
            f"{cta[0].size!r}; CTA v1 requires a static integer"
        )
    if not 1 <= count <= target.device.sm_count:
        raise ValueError(
            f"cta preflight root {root.name!r} requests {count} CTAs; "
            f"device {target.device.name!r} supports 1 <= n <= "
            f"{target.device.sm_count}"
        )
    return target, count


def _validate_region(
    region: GridRegionExpr,
    root: Function,
    owner: Function,
    visit_expr,
) -> None:
    start = _static_int(region.start)
    extent = _static_int(region.extent)
    step = _static_int(region.step)
    if start is None:
        raise ValueError(
            _context(
                root,
                owner,
                f"GridRegion {region.induction_var.name!r} has dynamic "
                f"start {region.start!r}",
            )
        )
    if extent is None:
        raise ValueError(
            _context(
                root,
                owner,
                f"GridRegion {region.induction_var.name!r} has dynamic "
                f"extent {region.extent!r}",
            )
        )
    if step is None:
        raise ValueError(
            _context(
                root,
                owner,
                f"GridRegion {region.induction_var.name!r} has dynamic "
                f"step {region.step!r}",
            )
        )
    if start < 0:
        raise ValueError(
            _context(root, owner, f"GridRegion start must be non-negative, got {start}")
        )
    if extent <= 0:
        raise ValueError(
            _context(root, owner, f"GridRegion extent must be positive, got {extent}")
        )
    if step <= 0:
        raise ValueError(
            _context(root, owner, f"GridRegion step must be positive, got {step}")
        )
    for value in (*region.init_args, region.body, *region.yield_values):
        visit_expr(value, owner)


def preflight_cta(
    subject: Function | Module,
    root: Function | None = None,
) -> CtaPreflightResult:
    """Validate the private CTA v1 entry contract without scheduling."""
    root_fn = _root_from(subject, root)
    target, cta_count = _validate_root_topology(root_fn)
    active: set[int] = set()
    visited: set[tuple[int, Target]] = set()
    reachable: list[Function] = []

    def visit_function(fn: Function, effective_target: Target, call: Call | None) -> None:
        if fn.body is None:
            site = f" at call {call.loc!r}" if call is not None else ""
            raise ValueError(
                _context(
                    root_fn,
                    fn,
                    f"function has no body{site}; dispatch/kernel boundaries "
                    "are not CTA v1 helpers",
                )
            )
        key = id(fn)
        if key in active:
            call_site = f" at call {call.loc!r}" if call is not None else ""
            raise ValueError(
                _context(
                    root_fn,
                    fn,
                    f"recursive helper call{call_site} is unsupported",
                )
            )
        visit_key = (key, effective_target)
        if visit_key in visited:
            return
        if fn is not root_fn:
            if fn.target is not None and fn.target != effective_target:
                call_site = f" at call {call.loc!r}" if call is not None else ""
                raise ValueError(
                    _context(
                        root_fn,
                        fn,
                        f"explicit helper Target {fn.target!r} conflicts with "
                        f"effective parent Target {effective_target!r}{call_site}",
                    )
                )
            if fn.topologies:
                call_site = f" at call {call.loc!r}" if call is not None else ""
                raise ValueError(
                    _context(
                        root_fn,
                        fn,
                        f"helper declares program topologies "
                        f"{fn.topologies!r}{call_site}; helpers must omit "
                        "program topology ownership",
                    )
                )
        active.add(key)
        reachable.append(fn)
        try:
            visit_expr(fn.body, fn)
        finally:
            active.remove(key)
            visited.add(visit_key)

    def visit_expr(expr: Expr | None, owner: Function) -> None:
        if expr is None:
            return
        if isinstance(expr, GridRegionExpr):
            _validate_region(expr, root_fn, owner, visit_expr)
            return
        if isinstance(expr, Tuple):
            for element in expr.elements:
                visit_expr(element, owner)
            return
        if not isinstance(expr, Call):
            return
        target_op = expr.target
        if isinstance(target_op, (PrimFunction, Launch, SymbolRef)):
            raise ValueError(
                _context(
                    root_fn,
                    owner,
                    f"kernel call target {type(target_op).__name__} at "
                    f"{expr.loc or '<unnamed>'!r} is unsupported in HIR CTA v1",
                )
            )
        for arg in expr.args:
            visit_expr(arg, owner)
        if not isinstance(target_op, Function):
            return
        effective_target = owner.target or target
        if effective_target is None:
            raise ValueError(
                _context(
                    root_fn,
                    owner,
                    f"call {expr.loc or target_op.name!r} has no effective Target",
                )
            )
        visit_function(target_op, effective_target, expr)

    visit_function(root_fn, target, None)
    return CtaPreflightResult(
        root=root_fn,
        cta_count=cta_count,
        reachable_functions=tuple(reachable),
    )


def _preflight_cta(
    subject: Function | Module,
    root: Function | None = None,
) -> CtaPreflightResult:
    """Private spelling used by stage-owned callers."""
    return preflight_cta(subject, root)


__all__ = ["CtaPreflightResult", "preflight_cta"]
