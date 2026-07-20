"""HIR matrix-multiply-accumulate ops (per-shape op classes).


- ``Mma`` is an abstract marker base class only — ``isinstance``
  dispatch entry point for cost model / lowering.
- Concrete classes encode arch + shape in the class name
  (``Mma_SM80_16x8x16``, ``Wgmma_SM90_64x128x16``); dtype / layout /
  transpose ride as ParamDef attributes.
- All classes are **SSA value ops** — they take ``a`` / ``b`` and
  return a fragment tensor. Accumulation is expressed at HIR via
  outer ``add(acc, Mma_*(a=, b=))``; the in-place accumulator
  operand only appears at TIR / codegen lowering time.
- ``mma`` is warp-level (32 threads collaborate per call);
  ``wgmma`` is cluster-level (4 warps in a CTA cluster collaborate).
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry import register_typeinfer


class Mma(Op):
    """Abstract marker for the family of matrix-multiply-accumulate ops.

    Concrete subclasses (`Mma_SM80_*`, `Wgmma_SM90_*`) own the
    ParamDef contract, typeinfer rule, and cost-model attributes.
    The base class exists so callers can dispatch with
    ``isinstance(call.target, Mma)`` regardless of fragment shape.
    """

# ── PTX SM80 mma family ─────────────────────────────────────────────────

@register_op(category="nn")
class Mma_SM80_16x8x16(Mma):
    """PTX ``mma.sync.aligned.m16n8k16``.

    Warp-level (32 threads). ``a`` is M=16 × K=16; ``b`` is K=16 ×
    N=8; the returned fragment is M=16 × N=8 in ``dtype_acc``.
    """
    a = ParamDef(kind="input", pattern=Tensor)
    b = ParamDef(kind="input", pattern=Tensor)
    dtype_a = ParamDef(kind="attribute", annotation=DType)
    dtype_b = ParamDef(kind="attribute", annotation=DType)
    dtype_acc = ParamDef(kind="attribute", annotation=DType)
    a_layout = ParamDef(kind="attribute", annotation=str, default="T")
    b_layout = ParamDef(kind="attribute", annotation=str, default="N")

@register_typeinfer(Mma_SM80_16x8x16)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    op = call.target
    a_ty = ctx.type_of(call.args[0])
    b_ty = ctx.type_of(call.args[1])
    return TensorType(
        shape=(16, 8),
        dtype=op.dtype_acc,
        layout=a_ty.layout,
        storage=resolve_anchor_storage(ctx, call, a_ty.storage, b_ty.storage),
    )

# ── PTX SM90 wgmma family ───────────────────────────────────────────────

@register_op(category="nn")
class Wgmma_SM90_64x128x16(Mma):
    """PTX ``wgmma.mma_async.sync.aligned.m64n128k16``.

    Cluster-level (4 warps in a CTA cluster). Returned fragment is
    M=64 × N=128 in ``dtype_acc``.
    """
    a = ParamDef(kind="input", pattern=Tensor)
    b = ParamDef(kind="input", pattern=Tensor)
    dtype_a = ParamDef(kind="attribute", annotation=DType)
    dtype_b = ParamDef(kind="attribute", annotation=DType)
    dtype_acc = ParamDef(kind="attribute", annotation=DType)
    a_layout = ParamDef(kind="attribute", annotation=str, default="T")
    b_layout = ParamDef(kind="attribute", annotation=str, default="N")

@register_typeinfer(Wgmma_SM90_64x128x16)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    op = call.target
    a_ty = ctx.type_of(call.args[0])
    b_ty = ctx.type_of(call.args[1])
    return TensorType(
        shape=(64, 128),
        dtype=op.dtype_acc,
        layout=a_ty.layout,
        storage=resolve_anchor_storage(ctx, call, a_ty.storage, b_ty.storage),
    )

__all__ = ["Mma", "Mma_SM80_16x8x16", "Wgmma_SM90_64x128x16"]
