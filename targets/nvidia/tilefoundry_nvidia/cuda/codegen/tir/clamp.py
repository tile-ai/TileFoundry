"""Codegen for TIR Clamp — the pointwise entry with ``clamp_op`` as its ``fn``."""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry.ir import tir
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context


@codegen.register(target.Target, codegen.Role.EMIT, tir.clamp.Clamp)
def emit(call, ctx: context.Context) -> None:
    src, dst = call.args
    op = call.target
    src_n = ctx.name_for(src)
    dst_n = ctx.name_for(dst)
    ctx.emit(
        f"tilefoundry::ops::elementwise({dst_n}, tilefoundry::primitive::clamp_op{{"
        f"{float(op.min_val)}f, {float(op.max_val)}f}}, {src_n});"
    )


__all__: list[str] = []
