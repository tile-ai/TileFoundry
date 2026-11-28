"""dst = lhs <kind> rhs on the processing engine (add, sub, mul, min, max)."""

from types import MappingProxyType

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, schedule, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import math
from tilefoundry_apple.ane.ir.tir import evaluation, layout

KINDS = MappingProxyType(
    {
        expr.BinaryKind.ADD: lambda lhs, rhs: lhs + rhs,
        expr.BinaryKind.SUB: lambda lhs, rhs: lhs - rhs,
        expr.BinaryKind.MUL: lambda lhs, rhs: lhs * rhs,
        expr.BinaryKind.MIN: lambda lhs, rhs: lhs.minimum(rhs),
        expr.BinaryKind.MAX: lambda lhs, rhs: lhs.maximum(rhs),
    }
)


@expr.register_op(dialect="T", category="pe", namespace="ane", name="binary")
class Binary(expr.Op):
    """dst = lhs <kind> rhs, rounded to fp16; the three operands have one shape."""

    capability = expr.OpCapability("pe.binary")
    execution_mesh = layout.ENGINE
    hir_matches = (
        schedule.HirMatch(
            math.Binary,
            when={"kind": pattern.Wildcard("kind")},
            operands={"lhs": "lhs", "rhs": "rhs", "dst": "result"},
        ),
    )

    kind = expr.ParamDef(
        kind="attribute", annotation=expr.BinaryKind, pattern=layout.one_of("kind", KINDS)
    )
    patch = expr.ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)
    width_code = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    ocg_size = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    omit_ocg = expr.ParamDef(kind="attribute", annotation=bool, optional=True, default=False)
    lhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.feature("lhs", layout.SMEM_ONLY))
    rhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.feature("rhs", layout.SMEM_ONLY))
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.feature("dst", layout.SMEM_ONLY))
    between = (
        pattern.SameConstraint("shape", "lhs", "dst"),
        pattern.SameConstraint("shape", "rhs", "dst"),
    )


@typeinfer.register(Binary)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


access.register_relation(Binary)(access.identity_relations)


@evaluator.register(Binary)
def _(ctx):
    *sources, dst = ctx.args
    lhs, rhs = (arg.data.to(utils.torch.float32) for arg in sources)
    return evaluation.write(dst, evaluation.f16(KINDS[ctx.op.kind](lhs, rhs)))


__all__ = ["Binary"]
