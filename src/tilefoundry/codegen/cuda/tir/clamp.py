"""Codegen for TIR Clamp — emits ``tilefoundry::ops::clamp(src, dst, N, min, max)``."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.clamp import Clamp
from tilefoundry.ir.types.shape_helpers import shape_runtime_total
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, shard_layout_local_shape


def _materialised_shape_dyn(ty) -> tuple:
    """Per-thread materialised shape, preserving ``DimVar`` entries so a
    runtime element count can be derived via ``shape_runtime_total``."""
    layout = getattr(ty, "layout", None)
    if isinstance(layout, ShardLayout):
        # [shard §7.1.1](docs/spec/shard.md#711-layoutshape): ``layout.shape`` is global; derive per-thread local.
        return shard_layout_local_shape(layout)
    return tuple(ty.shape)


@register_codegen_cuda(Clamp)
def _emit(call, ctx: CodegenContext) -> None:
    src, dst = call.args
    op = call.target
    src_n = ctx.name_for(src)
    dst_n = ctx.name_for(dst)
    # Runtime element count — for DimVar-bearing tensors this resolves
    # to a C++ expression (e.g. ``x_shape_0``) so the kernel loop
    # follows the actual extent, not the envelope upper bound.
    N = shape_runtime_total(_materialised_shape_dyn(dst.type), ctx._dim_var_runtime)
    ctx.emit(
        f"tilefoundry::ops::clamp({src_n}, {dst_n}, {N}, "
        f"{float(op.min_val)}f, {float(op.max_val)}f);"
    )
