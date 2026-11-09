"""Region isolation: a region body reaches captured values only through its params."""

from __future__ import annotations

from tilefoundry.ir import core, expr

__all__ = ["verify_region_isolated"]


def verify_region_isolated(region: core.LoopRegion | core.MeshRegion, ctx=None) -> None:
    """Ensure a region body reaches captured values only through its params."""
    if len(region.params) != len(region.args):
        message = f"region has {len(region.params)} params but {len(region.args)} args"
        if ctx is not None:
            ctx.error(region, message)
        raise expr.VerifyError(f"{type(region).__name__}: {message}")
    arg_ids = {id(arg) for arg in region.args}
    roots = (
        (region.body, *region.yield_values) if isinstance(region, core.LoopRegion) else (region.body,)
    )
    leaked = arg_ids.intersection(id(node) for root in roots for node in core.collect_exprs(root))
    if not leaked:
        return
    message = (
        "region is not isolated: body reads an args value directly; reference its param instead"
    )
    if ctx is not None:
        ctx.error(region, message)
    raise expr.VerifyError(f"{type(region).__name__}: {message}")
