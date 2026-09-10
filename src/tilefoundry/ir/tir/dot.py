"""Effect-form TIR Op ``tir.tensor.Dot`` — ``dst = sum(lhs * rhs)`` in one statement.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, shard_layout_local_shape
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

__all__ = ["Dot"]


@register_op(dialect="T", category="tensor")
class Dot(Op):
    """Fused multiply-contract; every participant leaves holding the total."""

    lhs = ParamDef(kind="input", pattern=Tensor)
    rhs = ParamDef(kind="input", pattern=Tensor)
    dst = ParamDef(kind="input", pattern=Tensor)
    workspace = ParamDef(kind="input", pattern=Tensor, optional=True, default=None)


@register_typeinfer(Dot)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


def _local_numel(ty) -> int | None:
    """The elements one participant holds of *ty*, or ``None`` when undecidable."""
    layout = getattr(ty, "layout", None)
    if isinstance(layout, ShardLayout):
        try:
            shape = shard_layout_local_shape(layout, require_static=False)
        except ValueError:
            return None
    else:
        shape = ty.shape
    n = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        n *= dim
    return n


@register_verify_stmt(Dot)
def _(call: "Call", ctx: "VerifyContext") -> None:
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    dst = ctx.type_of(call.args[2])

    lhs_n = _local_numel(lhs)
    rhs_n = _local_numel(rhs)
    if lhs_n is not None and rhs_n is not None and lhs_n != rhs_n:
        ctx.error(
            call,
            f"Dot operands contract over different lengths: {lhs_n} vs {rhs_n}",
        )
    dst_n = _local_numel(dst)
    if dst_n is not None and dst_n != 1:
        ctx.error(call, f"Dot destination must be one cell, got {dst_n} elements")

    if len(call.args) < 4:
        return
    ws = ctx.type_of(call.args[3])
    if ws.storage != StorageKind.SMEM:
        ctx.error(call, f"Dot workspace must be smem, got {ws.storage}")
    if not isinstance(getattr(lhs, "layout", None), ShardLayout):
        ctx.error(
            call,
            "Dot with a workspace contracts across the block, and the warps "
            "that post a partial are the ones lhs's mesh names; lhs must "
            "carry a ShardLayout",
        )
