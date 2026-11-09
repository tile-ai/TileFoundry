"""CUDA tiled matrix-multiply-accumulate operation."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, evaluator, typeinfer, verify
from tilefoundry.ir import expr, types

from .mma_atom import AtomPattern, FromAtom, MmaAtom
from .sm80_mma import Mma
from .sm89_mma import Mma as sm89_mma_Mma
from .wgmma import Wgmma

FP_ACC_WIDEN = frozenset({
    (expr.DType.fp8e4m3, expr.DType.f32),
    (expr.DType.f16, expr.DType.f32),
    (expr.DType.bf16, expr.DType.f32),
    (expr.DType.f16, expr.DType.f16),
    (expr.DType.bf16, expr.DType.bf16),
    (expr.DType.f32, expr.DType.f32),
})


@expr.register_op(dialect="T", category="nn", name="tiled_mma")
class TiledMma(expr.Op):
    """Execute one tiled MMA; the atom declares its operand contracts."""

    capability = (
        expr.OpCapability(
            Wgmma.capability,
            declaration=Wgmma,
            attribute="atom",
        ),
        expr.OpCapability(
            Mma.capability,
            declaration=Mma,
            attribute="atom",
        ),
        expr.OpCapability(
            sm89_mma_Mma.capability,
            declaration=sm89_mma_Mma,
            attribute="atom",
        ),
    )

    @property
    def resource(self):
        return self.atom.resource

    acc = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ | expr.MemoryEffect.WRITE,
        pattern=FromAtom("C"),
    )
    lhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=FromAtom("A"))
    rhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=FromAtom("B"))
    atom = expr.ParamDef(
        kind="attribute",
        annotation=MmaAtom,
        pattern=AtomPattern(Wgmma, Mma, sm89_mma_Mma),
    )
    execution_mesh = expr.ParamDef(
        kind="attribute",
        annotation=types.Mesh,
        optional=True,
        default=None,
    )

    def execution_mesh_pattern(self):
        """The atom's lane frame, preceded by independent group modes."""
        return self.atom.execution_mesh_pattern(groups=True)


@typeinfer.register(TiledMma)
def _(call: "expr.Call", ctx: "typeinfer.TypeInferContext") -> types.UnitType:
    return types.UnitType()


@access.register_relation(TiledMma)
def tiled_mma_access_relation(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    acc, lhs, rhs = (ctx.type_of(arg) for arg in call.args)
    *operands, accumulated = access.matmul_relations(lhs.shape, rhs.shape, (-2, -1, -1, -2))
    return (accumulated, *operands, accumulated)


@evaluator.register(TiledMma)
def eval_scheduled_mma(ctx):
    acc, lhs, rhs = (arg.data for arg in ctx.args)
    return evaluator.TensorValue(
        core.ShardTensor(
            ctx.result_type,
            acc + utils.torch.matmul(lhs.to(acc.dtype), rhs.to(acc.dtype)),
        )
    )


@verify.register(TiledMma)
def verify_operand_shapes(call: "expr.Call", ctx: "verify.Context") -> None:
    """Check A (M,K), B (K,N), and C (M,N) shapes and dtypes."""
    acc_ty, lhs_ty, rhs_ty = (ctx.type_of(arg) for arg in call.args[:3])
    if len(lhs_ty.shape) == 2 and len(rhs_ty.shape) == 2 and len(acc_ty.shape) == 2:
        m, k_l = lhs_ty.shape[-2], lhs_ty.shape[-1]
        k_r, n = rhs_ty.shape[-2], rhs_ty.shape[-1]
        if k_l != k_r:
            ctx.error(call, f"Mma K-dim mismatch: {k_l} vs {k_r}")
        if acc_ty.shape[-2] != m or acc_ty.shape[-1] != n:
            ctx.error(
                call,
                f"Mma acc shape mismatch: expected (...,{m},{n}), got "
                f"(...,{acc_ty.shape[-2]},{acc_ty.shape[-1]})",
            )
    if lhs_ty.dtype != rhs_ty.dtype:
        ctx.error(call, f"Mma lhs/rhs dtype mismatch: {lhs_ty.dtype} vs {rhs_ty.dtype}")
    if (lhs_ty.dtype, acc_ty.dtype) not in FP_ACC_WIDEN:
        ctx.error(
            call,
            f"Mma unsupported dtype combo: input {lhs_ty.dtype} acc {acc_ty.dtype}",
        )


__all__ = ["TiledMma", "verify_operand_shapes"]
