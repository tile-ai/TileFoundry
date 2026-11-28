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
OPERANDS = MappingProxyType({"src": "x", "dst": "result"})
SRC = layout.activation("src", layout.SMEM_ONLY)
DST = layout.activation("dst", layout.SMEM_ONLY)


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
    patch = expr.ParamDef(kind="attribute", annotation=tuple, optional=True, default=None)
    width_code = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    unicast_channels = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    ocg_size = expr.ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    omit_ocg = expr.ParamDef(kind="attribute", annotation=bool, optional=True, default=False)
    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=SRC)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=DST)


@typeinfer.register(Bypass)
def _(call, ctx) -> types.TupleType:
    return types.TupleType.Void


def axes(op, rank: int) -> tuple:
    """The axis order of a feature map, or of a matrix, whose values keep their coordinates."""
    return (0, 1) if rank == 2 else (0, *(1 + axis for axis in op.perm))


@access.register_relation(Bypass)
def _(call, ctx):
    src = ctx.type_of(call.args[0])
    hir = tensor.Transpose(perm=axes(call.target, len(src.shape)))
    read, written = access.relations_of(
        expr.Call(hir, call.args[:1], type=types.TupleType.Void), ctx
    )
    return read, written, written


@verify.register(Bypass)
def _(call, ctx) -> None:
    op = call.target
    src, dst = (ctx.type_of(arg) for arg in call.args)
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
    src, dst = ctx.args
    return evaluation.write(dst, apply(op.nonlinear, src.data.permute(*axes(op, src.data.dim()))))


__all__ = ["Bypass"]
