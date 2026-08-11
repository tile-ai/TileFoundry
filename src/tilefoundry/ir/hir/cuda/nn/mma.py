"""Define shape-specific HIR matrix-multiply-accumulate value Ops.

``Mma`` is a dispatch marker. Concrete class names encode architecture and
shape while attributes encode dtype and orientation. HIR returns ``A @ B`` as
a value; lowering introduces and zero-initialises the in-place accumulator.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from tilefoundry.evaluator import TensorValue, register_eval, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.tir.cuda.nn.mma import (
    SM80_16x8x16_F32BF16BF16F32_TN,
    make_atom,
)
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Broadcast, Layout, ShardLayout, try_c_order_strides
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer

_FLOAT_ACCUMULATOR_COMBINATIONS = frozenset(
    {
        (DType.f16, DType.f16, DType.f16),
        (DType.bf16, DType.bf16, DType.bf16),
        (DType.f16, DType.f16, DType.f32),
        (DType.bf16, DType.bf16, DType.f32),
        (DType.f32, DType.f32, DType.f32),
    }
)
_SM80_ATOM = make_atom(SM80_16x8x16_F32BF16BF16F32_TN)


class Mma(Op):
    """Abstract marker for the family of matrix multiply value Ops."""


@register_op(category="nn")
class Mma_SM80_16x8x16(Mma):
    """PTX ``mma.sync.aligned.m16n8k16`` logical value ``A @ B``."""

    a = ParamDef(kind="input", pattern=Tensor)
    b = ParamDef(kind="input", pattern=Tensor)
    dtype_a = ParamDef(kind="attribute", annotation=DType)
    dtype_b = ParamDef(kind="attribute", annotation=DType)
    dtype_acc = ParamDef(kind="attribute", annotation=DType)
    a_layout = ParamDef(kind="attribute", annotation=str, default="T")
    b_layout = ParamDef(kind="attribute", annotation=str, default="N")


@register_op(category="nn")
class Wgmma_SM90_64x128x16(Mma):
    """SM90 ``wgmma.mma_async.sync.aligned.m64n128k16`` logical ``A @ B``."""

    a = ParamDef(kind="input", pattern=Tensor)
    b = ParamDef(kind="input", pattern=Tensor)
    dtype_a = ParamDef(kind="attribute", annotation=DType)
    dtype_b = ParamDef(kind="attribute", annotation=DType)
    dtype_acc = ParamDef(kind="attribute", annotation=DType)
    a_layout = ParamDef(kind="attribute", annotation=str, default="T")
    b_layout = ParamDef(kind="attribute", annotation=str, default="N")


def _is_lowerable_sm80_target(op: Mma) -> bool:
    return (
        isinstance(op, Mma_SM80_16x8x16)
        and op.dtype_a == DType.bf16
        and op.dtype_b == DType.bf16
        and op.dtype_acc == DType.f32
        and op.a_layout == "T"
        and op.b_layout == "N"
    )


def _is_genuinely_sharded(layout) -> bool:
    return isinstance(layout, ShardLayout) and any(
        not isinstance(attr, Broadcast) for attr in layout.attrs
    )


def _matches_fragment_layout(actual: ShardLayout, expected: ShardLayout) -> bool:
    return (
        actual.layout == expected.layout
        and actual.attrs == expected.attrs
        and actual.mesh.topologies == expected.mesh.topologies
        and actual.mesh.layout == expected.mesh.layout
    )


def _derive_sm80_fragment_layout(a_ty: TensorType, b_ty: TensorType) -> ShardLayout:
    """Return the known C fragment or reject a non-instruction A/B claim."""
    for index, ty, expected, role in (
        (0, a_ty, _SM80_ATOM.A, "A"),
        (1, b_ty, _SM80_ATOM.B, "B"),
    ):
        if not isinstance(ty.layout, ShardLayout):
            raise ValueError(
                f"input {index} does not carry the known SM80 {role} fragment layout; "
                "use an explicit Reshard to that layout and materialize-to-RMEM"
            )
        if ty.storage is not StorageKind.RMEM:
            raise ValueError(
                f"input {index} SM80 {role} fragment is in {ty.storage}, not RMEM; "
                "use an explicit Reshard and materialize-to-RMEM"
            )
        if not _matches_fragment_layout(ty.layout, expected):
            raise ValueError(
                f"input {index} does not match the known SM80 {role} fragment layout; "
                "use an explicit Reshard to that layout and materialize-to-RMEM"
            )
    if (
        a_ty.layout.mesh.topologies != b_ty.layout.mesh.topologies
        or a_ty.layout.mesh.layout != b_ty.layout.mesh.layout
    ):
        raise ValueError(
            "input 1 SM80 B fragment references a different physical mesh from input 0; "
            "use an explicit Reshard to the common fragment mesh and materialize-to-RMEM"
        )
    return replace(_SM80_ATOM.C, mesh=a_ty.layout.mesh)


def _validate_contract(call: "Call", ctx: "TypeInferContext", a_shape, b_shape) -> tuple:
    op = call.target
    a_ty = ctx.type_of(call.args[0])
    b_ty = ctx.type_of(call.args[1])
    if a_ty.shape != a_shape:
        ctx.error(call, f"a shape must be {a_shape}, got {a_ty.shape}")
    if b_ty.shape != b_shape:
        ctx.error(call, f"b shape must be {b_shape}, got {b_ty.shape}")
    if a_ty.dtype != op.dtype_a:
        ctx.error(
            call,
            f"dtype_a={op.dtype_a.name} disagrees with input a dtype {a_ty.dtype.name}",
        )
    if b_ty.dtype != op.dtype_b:
        ctx.error(
            call,
            f"dtype_b={op.dtype_b.name} disagrees with input b dtype {b_ty.dtype.name}",
        )
    combo = (op.dtype_a, op.dtype_b, op.dtype_acc)
    if combo not in _FLOAT_ACCUMULATOR_COMBINATIONS:
        ctx.error(
            call,
            "dtype_acc combination "
            f"(dtype_a={op.dtype_a.name}, dtype_b={op.dtype_b.name}, "
            f"dtype_acc={op.dtype_acc.name}) is unsupported",
        )
    for field in ("a_layout", "b_layout"):
        value = getattr(op, field)
        if value not in ("N", "T"):
            ctx.error(call, f"{field} must be 'N' or 'T', got {value!r}")
    return a_ty, b_ty


def _infer_mma(call, ctx, *, a_shape, b_shape, out_shape) -> TensorType:
    a_ty, b_ty = _validate_contract(call, ctx, a_shape, b_shape)
    genuine = [
        index
        for index, ty in enumerate((a_ty, b_ty))
        if _is_genuinely_sharded(ty.layout)
    ]
    if genuine:
        if isinstance(call.target, Wgmma_SM90_64x128x16):
            ctx.error(
                call,
                f"input {genuine[0]} carries an unrepresentable WGMMA ShardLayout; "
                "use an explicit Reshard to a plain logical layout before WGMMA",
            )
        if not _is_lowerable_sm80_target(call.target):
            ctx.error(
                call,
                f"input {genuine[0]} claims an SM80 fragment, but only the "
                "BF16/BF16/F32 TN fragment contract is representable; use an "
                "explicit Reshard to a plain logical layout or materialize-to-RMEM",
            )
        try:
            out_layout = _derive_sm80_fragment_layout(a_ty, b_ty)
        except ValueError as error:
            ctx.error(call, str(error))
    elif any(isinstance(ty.layout, ShardLayout) for ty in (a_ty, b_ty)):
        out_layout = None
    elif a_ty.layout is None and b_ty.layout is None:
        out_layout = None
    else:
        out_layout = Layout(shape=out_shape, strides=try_c_order_strides(out_shape))
    return TensorType(
        shape=out_shape,
        dtype=call.target.dtype_acc,
        layout=out_layout,
        storage=resolve_anchor_storage(ctx, call, a_ty.storage, b_ty.storage),
    )


@register_typeinfer(Mma_SM80_16x8x16)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return _infer_mma(
        call,
        ctx,
        a_shape=(16, 16),
        b_shape=(16, 8),
        out_shape=(16, 8),
    )


@register_typeinfer(Wgmma_SM90_64x128x16)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return _infer_mma(
        call,
        ctx,
        a_shape=(64, 16),
        b_shape=(16, 128),
        out_shape=(64, 128),
    )


@register_eval(Mma_SM80_16x8x16)
@register_eval(Wgmma_SM90_64x128x16)
def _eval_mma(ctx):
    dtype_acc = to_torch_dtype(ctx.op.dtype_acc)
    a = ctx.args[0].data.to(dtype_acc)
    b = ctx.args[1].data.to(dtype_acc)
    return TensorValue(data=torch.matmul(a, b), type=ctx.result_type)


__all__ = ["Mma", "Mma_SM80_16x8x16", "Wgmma_SM90_64x128x16"]
