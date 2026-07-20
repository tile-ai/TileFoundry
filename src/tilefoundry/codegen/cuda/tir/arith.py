"""Codegen for generic Binary and Unary TIR effect-form Ops — tag-dispatched."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.arith import Binary, BinaryKind, Unary, UnaryKind
from tilefoundry.ir.types.shape_helpers import (
    shape_numel_upper_bound,
    shape_runtime_total,
    shape_upper_bound,
)
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, shard_layout_local_shape

_BINARY_TAG = {
    BinaryKind.MUL: "tilefoundry::ops::mul_op",
    BinaryKind.ADD: "tilefoundry::ops::add_op",
    BinaryKind.SUB: "tilefoundry::ops::sub_op",
    BinaryKind.DIV: "tilefoundry::ops::div_op",
}

_UNARY_TAG = {
    UnaryKind.RSQRT: "tilefoundry::ops::rsqrt_op",
    UnaryKind.NEG: "tilefoundry::ops::neg_op",
    UnaryKind.RELU: "tilefoundry::ops::relu_op",
    UnaryKind.SQUARE: "tilefoundry::ops::square_op",
}




def _materialised_shape(ty) -> tuple:
    """Return the cute-side materialised shape for *ty*.

    Sharded Tensors are materialised at the ``ShardLayout``'s
    per-mesh-position local shape, not the logical
    ``TensorType.shape``. Elementwise helpers must iterate by the
    local count or they walk past the per-thread / per-CTA
    allocation.
    """
    layout = getattr(ty, "layout", None)
    if isinstance(layout, ShardLayout):
        # spec §7: ``layout.shape`` is global; derive per-thread local.
        return shard_layout_local_shape(layout)
    return shape_upper_bound(ty.shape)


def _materialised_shape_dyn(ty) -> tuple:
    """Like ``_materialised_shape`` but preserves ``DimVar`` entries so
    the caller can ask for a runtime (string) iteration count."""
    layout = getattr(ty, "layout", None)
    if isinstance(layout, ShardLayout):
        return shard_layout_local_shape(layout)
    return tuple(ty.shape)


def _runtime_total(ty, ctx: CodegenContext) -> object:
    """Runtime element count for *ty* — int or C++ expression string.

    Uses the kernel's DimVar runtime scalars registered by the
    PrimFunction emitter so dynamic dims drive loop counts at launch
    time instead of the static envelope upper bound.
    """
    return shape_runtime_total(_materialised_shape_dyn(ty), ctx._dim_var_runtime)


def _tensor_expr(var, ctx: CodegenContext) -> str:
    """Kernel-param tensor operands are accessed through the cute wrap
    (``<name>_tensor``) the PrimFunction emitter materialises at the
    top of the body; non-param vars are referenced directly."""
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


@register_codegen_cuda(Binary)
def _emit_binary(call, ctx: CodegenContext) -> None:
    lhs, rhs, dst = call.args
    op = call.target
    lhs_n = _tensor_expr(lhs, ctx)
    rhs_n = _tensor_expr(rhs, ctx)
    dst_n = _tensor_expr(dst, ctx)
    op_tag = _BINARY_TAG[op.kind]
    # Iterate over the materialised per-thread shape, not the
    # logical ``TensorType.shape`` — reg+ShardLayout operands hold
    # only the local view in registers.
    l_shape = _materialised_shape(lhs.type)
    r_shape = _materialised_shape(rhs.type)

    if not r_shape or shape_numel_upper_bound(r_shape) == 1:
        N = _runtime_total(lhs.type, ctx)
        ctx.emit(f"tilefoundry::ops::binary_bcast_scalar({lhs_n}, {rhs_n}, {dst_n}, {N}, {op_tag}{{}});")
    elif len(l_shape) == 2 and len(r_shape) == 2 and r_shape[-1] == 1 and l_shape[0] == r_shape[0]:
        M, K = int(l_shape[0]), int(l_shape[1])
        ctx.emit(f"tilefoundry::ops::binary_bcast_col({lhs_n}, {rhs_n}, {dst_n}, {M}, {K}, {op_tag}{{}});")
    elif len(l_shape) == 2 and len(r_shape) == 1 and l_shape[-1] == r_shape[0]:
        M, K = int(l_shape[0]), int(l_shape[1])
        ctx.emit(f"tilefoundry::ops::binary_bcast_row({lhs_n}, {rhs_n}, {dst_n}, {M}, {K}, {op_tag}{{}});")
    elif (
        shape_numel_upper_bound(l_shape) % shape_numel_upper_bound(r_shape) == 0
        and shape_numel_upper_bound(l_shape) > shape_numel_upper_bound(r_shape)
    ):
        # Multi-cell broadcast: lhs is ``n_dst`` cells of ``step`` lanes,
        # rhs is a per-cell scalar (e.g. ``(1,3,4) op (1,3,1)`` after the
        # reduce keepdim path).
        n_dst = shape_numel_upper_bound(r_shape)
        step = shape_numel_upper_bound(l_shape) // n_dst
        ctx.emit(
            f"tilefoundry::ops::binary_cell_bcast({lhs_n}, {rhs_n}, {dst_n}, "
            f"{n_dst}, {step}, {op_tag}{{}});"
        )
    else:
        N = _runtime_total(dst.type, ctx)
        ctx.emit(f"tilefoundry::ops::binary({lhs_n}, {rhs_n}, {dst_n}, {N}, {op_tag}{{}});")


@register_codegen_cuda(Unary)
def _emit_unary(call, ctx: CodegenContext) -> None:
    src, dst = call.args
    op = call.target
    src_n = _tensor_expr(src, ctx)
    dst_n = _tensor_expr(dst, ctx)
    # Iterate over local view length — runtime when a DimVar is in play.
    N = _runtime_total(dst.type, ctx)

    if op.kind == UnaryKind.CAST:
        ctx.emit(f"tilefoundry::ops::cast({src_n}, {dst_n}, {N});")
    else:
        op_tag = _UNARY_TAG[op.kind]
        ctx.emit(f"tilefoundry::ops::unary({src_n}, {dst_n}, {N}, {op_tag}{{}});")
