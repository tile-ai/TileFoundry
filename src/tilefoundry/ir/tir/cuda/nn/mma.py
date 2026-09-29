"""CUDA tiled matrix-multiply-accumulate operation."""

from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Call, Op, OpCapability, Var
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    WildcardPattern,
)
from tilefoundry.ir.pattern import (
    predicates as P,
)
from tilefoundry.ir.types import DType, Mesh, TensorType, UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    matmul_relations,
    register_access_relation,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .mma_atom import AtomPattern, FromAtom, MmaAtom, physical_frames_match
from .sm80_mma import Mma as _Sm80Mma
from .wgmma import Wgmma

_FP_ACC_WIDEN = {
    (DType.f16, DType.f32),
    (DType.bf16, DType.f32),
    (DType.f16, DType.f16),
    (DType.bf16, DType.bf16),
    (DType.f32, DType.f32),
}


_WARP_ALIGNED = ComposedLayoutPattern(
    inner=None,
    offset=WildcardPattern("p0"),
    outer=LayoutPattern(
        ((WildcardPattern("n"),),),
        ((1,),),
        predicates=(
            WildcardPattern("n") % 32 == 0,
            P.Forward(per_mode=True),
            P.Injective(per_mode=True),
        ),
    ),
    predicates=(WildcardPattern("p0") % 32 == 0,),
)


@register_op(category="nn", name="tiled_mma")
class TiledMma(Op):
    """Execute one tiled MMA; the atom declares its operand contracts."""

    capability = (
        OpCapability(
            Wgmma.capability,
            declaration=Wgmma,
            attribute="atom",
        ),
        OpCapability(
            _Sm80Mma.capability,
            declaration=_Sm80Mma,
            attribute="atom",
        ),
    )

    @property
    def resource(self):
        return self.atom.resource

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
    execution_mesh = ParamDef(
        kind="attribute",
        annotation=Mesh,
        pattern=MeshPattern(("thread",), _WARP_ALIGNED),
        optional=True,
        default=None,
    )


@register_typeinfer(TiledMma)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_access_relation(TiledMma)
def _tiled_mma_access_relation(call: "Call", ctx) -> AccessRelations:
    acc, lhs, rhs = (ctx.type_of(arg) for arg in call.args)
    contraction = matmul_relations(lhs.shape, rhs.shape, (-2, -1, -1, -2))
    return AccessRelations(
        inputs=(contraction.outputs[0], *contraction.inputs),
        outputs=(contraction.outputs[0],),
    )


def operand_relations(
    op: TiledMma, operand_types: tuple[TensorType, ...]
) -> AccessRelations:
    """Return the registered operand relations for these concrete types."""
    args = tuple(
        Var(name=f"operand{index}", type=type_)
        for index, type_ in enumerate(operand_types)
    )
    call = Call(target=op, args=args, type=UnitType())
    try:
        relations = relations_of(call, TypeInferContext())
    except ValueError as error:
        raise ValueError(
            f"{op.atom.reference_name} has no registered operand access relation: {error}"
        ) from error
    return relations


@register_schedule_eval(TiledMma)
def _eval_scheduled_mma(ctx):
    acc, lhs, rhs = (arg.data for arg in ctx.args)
    return TensorValue(data=acc + torch.matmul(lhs, rhs), type=ctx.result_type)


@register_verify_stmt(TiledMma)
def verify_mma(call: "Call", ctx: "VerifyContext") -> None:
    """Check each operand against its atom and the active physical frame."""
    op = call.target
    atom = op.atom
    if ctx.scope is not None and ctx.scope.module is not None:
        capabilities = ctx.scope.module.target.architecture.capabilities
        if atom.capability not in capabilities:
            ctx.error(call, f"target does not support {atom.capability}")
    if not ctx.mesh_scope:
        ctx.error(call, "MMA requires an active physical mesh scope")
    current = ctx.mesh_scope[-1]
    participation = atom.execution_mesh_pattern()
    if participation.match(current) is None:
        ctx.error(
            call,
            "MMA enclosing mesh violates declared instruction participation, "
            f"which is {participation!r}",
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


__all__ = ["TiledMma", "operand_relations", "verify_mma", "verify_operand_shapes"]
