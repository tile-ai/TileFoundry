"""The T.metal.ballot instruction and its rules."""

import isl

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, typeinfer
from tilefoundry.ir import expr, types
from tilefoundry_apple.metal.ir.tir import evaluation, layout


@expr.register_op(dialect="T", category="arith", namespace="metal", name="ballot")
class Ballot(expr.Op):
    """One bit per lane of a bool, as a u32 every lane of the group reads (simd_ballot)."""

    capability = expr.OpCapability("simd_ballot")
    execution_mesh = layout.LANE_GROUPS

    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.shared(expr.DType.u32))
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.lanes(expr.DType.bool))


@typeinfer.register(Ballot)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


@access.register_relation(Ballot)
def _(call, ctx):
    dst = access.Relation(isl.map("{ [g, l] -> [g, 0] }"))
    return access.iterating(ctx.type_of(call.args[1]).shape, (dst, access.identity(2), dst))


@evaluator.register(Ballot)
def _(ctx):
    bits = ctx.args[1].data.to(utils.torch.int64)
    return evaluation.write(ctx, (bits << utils.torch.arange(bits.shape[1])).sum(1, keepdim=True))


__all__ = ["Ballot"]
