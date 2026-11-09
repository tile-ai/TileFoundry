"""Codegen for the Reduce TIR stmt — emits the uniform runtime reduce call."""

from __future__ import annotations

from types import MappingProxyType

from tilefoundry import codegen
from tilefoundry.ir import expr, tir
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context

REDUCE_TAG = MappingProxyType({
    expr.ReduceKind.MEAN: "tilefoundry::ops::mean_op",
    expr.ReduceKind.SUM: "tilefoundry::primitive::add_op",
    expr.ReduceKind.ABS_MAX: "tilefoundry::ops::absmax_op",
    expr.ReduceKind.MAX: "tilefoundry::primitive::max_op",
    expr.ReduceKind.MIN: "tilefoundry::primitive::min_op",
})


def axes_pack_typename(axes: tuple) -> str:
    """Axes pack typename.

    Render the HIR ``axes`` tuple as a ``cute::tuple<cute::Int<i>...>``
    template type for the runtime entry point. Using cute's native
    tuple keeps the reduce dispatch idiomatic with the rest of the
    codegen.
    """
    args = ", ".join(f"cute::Int<{int(a)}>" for a in axes)
    return f"cute::tuple<{args}>"


@codegen.register(target.Target, codegen.Role.EMIT, tir.reduce.Reduce)
def emit(call, ctx: context.Context) -> None:
    src, dst = call.args[0], call.args[1]
    src_n = ctx.name_for(src)
    dst_n = ctx.name_for(dst)
    op_tag = REDUCE_TAG[call.target.kind]
    axes_t = axes_pack_typename(call.target.axes)




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


__all__ = ["REDUCE_TAG"]
