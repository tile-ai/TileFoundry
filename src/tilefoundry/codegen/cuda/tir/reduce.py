"""Codegen for the Reduce TIR stmt — emits the uniform runtime reduce call."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.reduce import Reduce, ReduceKind

_REDUCE_TAG = {
    ReduceKind.MEAN: "tilefoundry::ops::mean_op",
    ReduceKind.SUM: "tilefoundry::ops::sum_op",
    ReduceKind.ABS_MAX: "tilefoundry::ops::absmax_op",
}


def _axes_pack_typename(axes: tuple) -> str:
    """Render the HIR ``axes`` tuple as a ``cute::tuple<cute::Int<i>...>``
    template type for the runtime entry point. Using cute's native
    tuple keeps the reduce dispatch idiomatic with the rest of the
    codegen."""
    args = ", ".join(f"cute::Int<{int(a)}>" for a in axes)
    return f"cute::tuple<{args}>"


@register_codegen_cuda(Reduce)
def _emit(call, ctx: CodegenContext) -> None:
    src, dst = call.args[0], call.args[1]
    src_n = ctx.name_for(src)
    dst_n = ctx.name_for(dst)
    op_tag = _REDUCE_TAG[call.target.kind]
    axes_t = _axes_pack_typename(call.target.axes)

    # Single public entry for every case. The runtime ``reduce`` derives the
    # tier (sharded) or the extents (plain, non-sharded) from the operands;
    # a workspace arg is passed only when the lowering sized one.
    if len(call.args) >= 3:
        ws_n = ctx.name_for(call.args[2])
        ctx.emit(
            f"tilefoundry::ops::reduce<{op_tag}, {axes_t}>"
            f"({src_n}, {dst_n}, {ws_n});"
        )
    else:
        ctx.emit(
            f"tilefoundry::ops::reduce<{op_tag}, {axes_t}>"
            f"({src_n}, {dst_n});"
        )
