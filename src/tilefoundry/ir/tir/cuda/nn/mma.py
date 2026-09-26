"""CUDA tiled matrix-multiply-accumulate operation."""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    CapturePattern,
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    MultipleOfPattern,
    OrPattern,
)
from tilefoundry.ir.pattern import (
    predicates as P,
)
from tilefoundry.ir.tir.verify import input_params
from tilefoundry.ir.types import DType, Mesh, UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

from .mma_atom import AtomPattern, FromAtom, MmaAtom, physical_frames_match, read_on
from .sm80_mma import Mma as _Sm80Mma
from .wgmma import Wgmma

_FP_ACC_WIDEN = {
    (DType.f16, DType.f32),
    (DType.bf16, DType.f32),
    (DType.f16, DType.f16),
    (DType.bf16, DType.bf16),
    (DType.f32, DType.f32),
}


def _warp_layout_pattern() -> LayoutPattern:
    return LayoutPattern(
        ((CapturePattern("n", MultipleOfPattern(32)),),),
        ((1,),),
        predicates=(P.Forward(per_mode=True), P.Injective(per_mode=True)),
    )


_WARP_ALIGNED = OrPattern(
    ComposedLayoutPattern(
        offset=CapturePattern("p0", MultipleOfPattern(32)),
        outer=_warp_layout_pattern(),
    ),
    _warp_layout_pattern(),
)


@register_op(category="nn", name="tiled_mma")
class TiledMma(Op):
    """Execute one tiled MMA; the atom declares its operand contracts."""

    @property
    def capability(self):
        return self.atom.capability

    acc = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=FromAtom("C"),
    )
    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=FromAtom("A"))
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=FromAtom("B"))
    atom = ParamDef(
        kind="attribute",
        annotation=MmaAtom,
        pattern=AtomPattern(Wgmma, _Sm80Mma),
    )
    scope = ParamDef(
        kind="attribute",
        annotation=Mesh,
        pattern=MeshPattern(("thread",), _WARP_ALIGNED),
        optional=True,
        default=None,
    )


@register_typeinfer(TiledMma)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(TiledMma)
def verify_mma(call: "Call", ctx: "VerifyContext") -> None:
    """Check each operand against its atom and the active physical frame."""
    op = call.target
    atom = op.atom
    held = tuple(ctx.type_of(arg) for arg in call.args)
    for param, value in zip(input_params(type(op)), held):
        pattern = read_on(param.pattern, op)
        if pattern.match(value) is None:
            ctx.error(
                call,
                f"MMA {param.name} is not one {atom.reference} reads; "
                f"it reads {pattern.describe()}",
            )
    if ctx.scope is not None and ctx.scope.module is not None:
        capabilities = ctx.scope.module.target.architecture.instruction_capabilities
        if op.capability not in capabilities:
            ctx.error(call, f"target does not support {op.capability}")
    if not ctx.mesh_scope:
        ctx.error(call, "MMA requires an active physical mesh scope")
    current = ctx.mesh_scope[-1]
    participation = atom.scope_pattern()
    if participation.match(current) is None:
        ctx.error(
            call,
            "MMA enclosing mesh violates declared instruction participation, "
            f"which is {participation.describe()}",
        )
    if atom.mesh is not None and not physical_frames_match(atom.mesh, current):
        ctx.error(call, "MMA atom frame differs from active mesh scope")
    verify_operand_shapes(call, ctx)


def verify_operand_shapes(call: "Call", ctx: "VerifyContext") -> None:
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
    if (lhs_ty.dtype, acc_ty.dtype) not in _FP_ACC_WIDEN:
        ctx.error(
            call,
            f"Mma unsupported dtype combo: input {lhs_ty.dtype} acc {acc_ty.dtype}",
        )


__all__ = ["TiledMma", "verify_mma", "verify_operand_shapes"]
