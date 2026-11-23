"""dst = nonlinear(permute(src)) as one neural-engine pass-through task."""

from itertools import permutations
from types import MappingProxyType

from tilefoundry.interfaces import access, evaluator, schedule, typeinfer, verify
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import nn, tensor
from tilefoundry_apple.ane.ir.tir import evaluation, layout

from .nonlinear import Nonlinear, apply

IDENTITY = (0, 1, 2)
SWAP = (2, 1, 0)
PERMS = tuple(permutations(IDENTITY))
OPERANDS = MappingProxyType({"dst": "result", "src": "x"})
SRC = (layout.feature_layout("src"), layout.matrix_layout("src", layout.ACTIVATION_LIMITS))
DST = (layout.feature_layout("dst"), layout.matrix_layout("dst", layout.ACTIVATION_LIMITS))


@expr.register_op(dialect="T", category="ne", namespace="ane", name="bypass")
class Bypass(expr.Op):
    """dst = nonlinear(src permuted over (channel, height, width) by perm).

    The feature form permutes axes 1 to 3 of [1, C, H, W]. The matrix form relays a
    row-major [M, K] src as a channel-outer dst of the same values; perm is (2, 1, 0).
    """

    capability = expr.OpCapability("ne.bypass")
    execution_mesh = layout.ENGINE
    hir_matches = (
        schedule.HirMatch(nn.ReLU, attributes={"nonlinear": Nonlinear.RELU}, operands=OPERANDS),
        *(
            schedule.HirMatch(
                tensor.Transpose,
                when={"perm": (0, *(1 + axis for axis in perm))},
                attributes={"perm": perm},
                operands=OPERANDS,
            )
            for perm in PERMS[1:]
        ),
        schedule.HirMatch(
            tensor.Transpose, when={"perm": pattern.Wildcard("perm")}, operands=OPERANDS
        ),
    )

    perm = expr.ParamDef(
        kind="attribute", annotation=tuple, pattern=layout.one_of("perm", PERMS), default=IDENTITY
    )
    nonlinear = expr.ParamDef(
        kind="attribute",
        annotation=Nonlinear,
        pattern=pattern.Wildcard("nonlinear"),
        default=Nonlinear.NONE,
    )
    tile_height = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    tile_width = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    patch = expr.ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)
    width_code = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    dst = expr.ParamDef(
        kind="input", effect=expr.MemoryEffect.WRITE, pattern=layout.forms("dst", DST)
    )
    src = expr.ParamDef(
        kind="input", effect=expr.MemoryEffect.READ, pattern=layout.forms("src", SRC)
    )


@typeinfer.register(Bypass)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


def axes(op, rank: int) -> tuple:
    """The axis order of a feature map, or of a matrix, whose values keep their coordinates."""
    return (0, 1) if rank == 2 else (0, *(1 + axis for axis in op.perm))


@access.register_relation(Bypass)
def _(call, ctx):
    src = ctx.type_of(call.args[1])
    hir = tensor.Transpose(perm=axes(call.target, len(src.shape)))
    read, written = access.relations_of(
        expr.Call(hir, call.args[1:], type=types.TupleType.Void), ctx
    )
    return written, read, written


@verify.register(Bypass)
def _(call, ctx) -> None:
    op = call.target
    dst, src = (ctx.type_of(arg) for arg in call.args)
    if len(dst.shape) != len(src.shape):
        ctx.error(
            call, f"dst shape {tuple(dst.shape)} and src shape {tuple(src.shape)} differ in rank"
        )
    if len(src.shape) == 2:
        if tuple(op.perm) != SWAP:
            ctx.error(call, f"perm {tuple(op.perm)} must be {SWAP} for a matrix")
        rows, columns = src.shape
        layout.require_strides(call, ctx, "src", src, (columns, 1))
        layout.require_strides(call, ctx, "dst", dst, (1, rows))
    expected = tuple(src.shape[axis] for axis in axes(op, len(src.shape)))
    if tuple(dst.shape) != expected:
        ctx.error(call, f"dst shape {tuple(dst.shape)} differs from the permuted shape {expected}")


@evaluator.register(Bypass)
def _(ctx):
    op = ctx.op
    src = ctx.args[1].data
    return evaluation.write(ctx, apply(op.nonlinear, src.permute(*axes(op, src.dim()))))


__all__ = ["Bypass"]
