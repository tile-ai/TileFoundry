"""Emitter for ``tir.nn.ReLU`` (stmt-form pointwise ReLU).

Emits a single call to the runtime's uniform unary entry
``tilefoundry::ops::unary(src, dst, N, relu_op{})`` — the same explicit
element-count convention ``tir.arith.Unary`` uses for ``UnaryKind.RELU``
(see ``codegen/cuda/tir/arith.py``); the element-wise loop semantics live in
the runtime header. The destination tensor must already have been
materialised by a preceding ``LetStmt`` on an ``AllocTensor`` Expr Op.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.nn import ReLU
from tilefoundry.ir.types.shape_helpers import shape_runtime_total
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, shard_layout_local_shape


def _materialised_shape_dyn(ty) -> tuple:
    """Per-thread materialised shape, preserving ``DimVar`` entries so a
    runtime element count can be derived via ``shape_runtime_total``."""
    layout = getattr(ty, "layout", None)
    if isinstance(layout, ShardLayout):
        # spec §7: ``layout.shape`` is global; derive per-thread local.
        return shard_layout_local_shape(layout)
    return tuple(ty.shape)


@register_codegen_cuda(ReLU)
def _emit(call, ctx: CodegenContext) -> None:
    src, dst = call.args[0], call.args[1]
    if not isinstance(src, Var) or not isinstance(dst, Var):
        raise RuntimeError("tir.nn.ReLU: demo path expects Var operands on both sides")
    src_name = ctx.name_for(src)
    dst_name = ctx.name_for(dst)
    N = shape_runtime_total(_materialised_shape_dyn(dst.type), ctx._dim_var_runtime)
    ctx.emit(
        f"tilefoundry::ops::unary({src_name}, {dst_name}, {N}, "
        f"tilefoundry::ops::relu_op{{}});"
    )
