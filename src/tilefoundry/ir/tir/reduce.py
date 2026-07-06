"""Effect-ful TIR Op ``tir.tensor.Reduce``.

Generic reduction Op that dispatches by ``ReduceKind`` tag (``MEAN``
/ ``SUM``). Writes the reduced result into ``dst``. Wrapped by
``Evaluate(Reduce, ...)`` in Stmt position.

The Op carries an optional ``workspace`` input — a scratch buffer the
runtime uses to stage cross-warp partial sums (no hardware
register-direct exchange across warps exists on Hopper / Ada /
Ampere; intra-warp uses ``__shfl_xor_sync`` only). The
``workspace`` argument is **not type/scope-restricted** at the
IR level — the HIR→TIR lowering picks the appropriate storage
(rmsnorm uses ``'smem'``); other use cases may pick
``'gmem'`` or ``'rmem'``. ``workspace`` is omitted (``None``)
when the reduction stays inside a single warp or the input is
not mesh-sharded across an inter-warp topology.

CUDA codegen forwards the workspace tensor to
``tilefoundry::ops::reduce<Op, Axes>(src, dst, workspace)``.
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer, register_verify_stmt
from tilefoundry.ir.types import UnitType

__all__ = ["ReduceKind", "Reduce"]

@register_op(dialect="T", category="tensor")
class Reduce(Op):
    """Generic reduction op dispatched by ``kind`` tag.

    Spec: tir.md §3.3

    ``workspace`` is an optional scratch buffer used by the
    runtime template for cross-warp staging; ``None`` when not
    needed.
    """
    src = ParamDef(kind="input", pattern=Tensor)
    dst = ParamDef(kind="input", pattern=Tensor)
    workspace = ParamDef(
        kind="input", pattern=Tensor, optional=True, default=None
    )
    axes = ParamDef(kind="attribute", annotation=tuple)
    kind = ParamDef(kind="attribute", annotation=ReduceKind)
    warps_per_group = ParamDef(kind="attribute", annotation=int, default=1)

@register_typeinfer(Reduce)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()

@register_verify_stmt(Reduce)
def _(call: "Call", ctx: "VerifyContext") -> None:
    op = call.target
    if not isinstance(op.kind, ReduceKind):
        ctx.error(call, f"Reduce: kind must be ReduceKind enum, got {type(op.kind)}")
    src_ty = ctx.type_of(call.args[0])  # noqa: F841
    dst_ty = ctx.type_of(call.args[1])  # noqa: F841
    # Per-shard reshard lowering may produce rank-N (e.g.
    # ``(1, 1, 1, 8)``) src tensors. The
    # runtime template (``tilefoundry::ops::reduce<Op, Axes>``) iterates
    # via ``cute::size(src)`` so rank is no longer relevant at the
    # verifier level — the old rank<=2 guard predates the sharded
    # reduce path.
