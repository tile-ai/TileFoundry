"""Codegen for the Reduce TIR stmt — emits the uniform runtime reduce call."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.reduce import Reduce, ReduceKind
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen

REDUCE_TAG = {
    ReduceKind.MEAN: "tilefoundry::ops::mean_op",
    ReduceKind.SUM: "tilefoundry::primitive::add_op",
    ReduceKind.ABS_MAX: "tilefoundry::ops::absmax_op",
    ReduceKind.MAX: "tilefoundry::primitive::max_op",
    ReduceKind.MIN: "tilefoundry::primitive::min_op",
}


def _axes_pack_typename(axes: tuple) -> str:
    """Axes pack typename.

    Render the HIR ``axes`` tuple as a ``cute::tuple<cute::Int<i>...>``
    template type for the runtime entry point. Using cute's native
    tuple keeps the reduce dispatch idiomatic with the rest of the
    codegen.
    """
    args = ", ".join(f"cute::Int<{int(a)}>" for a in axes)
    return f"cute::tuple<{args}>"


@register_codegen(CudaTarget, Role.EMIT, Reduce)
def _emit(call, ctx: CudaCodegenContext) -> None:
    src, dst = call.args[0], call.args[1]
    src_n = ctx.name_for(src)
    dst_n = ctx.name_for(dst)
    op_tag = REDUCE_TAG[call.target.kind]
    axes_t = _axes_pack_typename(call.target.axes)




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
