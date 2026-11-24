"""dst = one affine task with options on the processing engine (abs, square, clamp)."""

from types import MappingProxyType

from tilefoundry import utils
from tilefoundry.interfaces import access, evaluator, schedule, typeinfer, verify
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import math
from tilefoundry_apple.ane.ir.tir import evaluation, layout

OPERANDS = MappingProxyType({"dst": "result", "src": "x"})
CLAMP = pattern.Sequence(
    pattern.Wildcard("clamp_low"),
    pattern.Wildcard("clamp_high"),
    predicates=(pattern.Wildcard("clamp_low") <= pattern.Wildcard("clamp_high"),),
)


@expr.register_op(dialect="T", category="pe", namespace="ane", name="unary")
class Unary(expr.Op):
    """dst = clamp(square(abs(src * pre_scale * scale + bias))), each step rounded to fp16.

    abs, square and clamp apply only when set; clamp is (low, high).
    """

    capability = expr.OpCapability("pe.unary")
    execution_mesh = layout.ENGINE
    hir_matches = (
        schedule.HirMatch(
            math.Unary,
            when={"kind": expr.UnaryKind.ABS},
            attributes={"abs": True},
            operands=OPERANDS,
        ),
        schedule.HirMatch(
            math.Unary,
            when={"kind": expr.UnaryKind.SQUARE},
            attributes={"square": True},
            operands=OPERANDS,
        ),
        schedule.HirMatch(
            math.Unary,
            when={"kind": expr.UnaryKind.NEG},
            attributes={"scale": -1.0},
            operands=OPERANDS,
        ),
    )

    pre_scale = expr.ParamDef(kind="attribute", annotation=float, default=1.0)
    scale = expr.ParamDef(kind="attribute", annotation=float, default=1.0)
    bias = expr.ParamDef(kind="attribute", annotation=float, default=0.0)
    abs = expr.ParamDef(
        kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=False
    )
    square = expr.ParamDef(
        kind="attribute", annotation=bool, pattern=pattern.Or(False, True), default=False
    )
    clamp = expr.ParamDef(
        kind="attribute", annotation=tuple, optional=True, default=None, pattern=CLAMP
    )
    tile_height = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    tile_width = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    patch = expr.ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)
    width_code = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    ocg_size = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    chained_source = expr.ParamDef(kind="attribute", annotation=bool, optional=True, default=False)
    omit_ocg = expr.ParamDef(kind="attribute", annotation=bool, optional=True, default=False)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.feature("dst"))
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=layout.feature("src"))
    between = (pattern.SameConstraint("shape", "dst", "src"),)


@typeinfer.register(Unary)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


access.register_relation(Unary)(access.identity_relations)


@verify.register(Unary)
def _(call, ctx) -> None:
    for name, operand in zip(("dst", "src"), call.args, strict=True):
        try:
            layout.feature_dma_strides(ctx.type_of(operand))
        except ValueError as error:
            ctx.error(call, f"{name}: {error}")


@evaluator.register(Unary)
def _(ctx):
    op = ctx.op
    value = evaluation.f16(ctx.args[1].data.to(utils.torch.float32) * op.pre_scale)
    value = evaluation.f16(value * op.scale)
    value = evaluation.f16(value + op.bias)
    if op.abs:
        value = value.abs()
    if op.square:
        value = evaluation.f16(value * value)
    if op.clamp is not None:
        value = evaluation.f16(value.clamp(*op.clamp))
    return evaluation.write(ctx, value)


__all__ = ["Unary"]
