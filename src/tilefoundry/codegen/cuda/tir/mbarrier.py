"""Emitters for the mbarrier TIR ops — each writes its instruction inline."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.cuda.tir.stmts.scalar_expr import render_scalar_expr
from tilefoundry.ir.tir.cuda.sync.mbarrier import (
    MBarrierArriveExpectTx,
    MBarrierInit,
    MBarrierInvalidate,
    MBarrierWaitParity,
)
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


def barrier_word(var, ctx: CudaCodegenContext) -> str:
    """The address of the barrier word itself, as a generic pointer."""
    base = ctx.name_for(var)
    if ctx.is_kernel_param(var):
        base = f"{base}_tensor"
    if isinstance(getattr(getattr(var, "type", None), "layout", None), ShardLayout):
        base = f"tilefoundry::local({base})"
    return f"&{base}(0)"


def _smem_addr(var, ctx: CudaCodegenContext) -> str:
    """The barrier's shared-window address, which every ``mbarrier.*`` takes."""
    return f"static_cast<unsigned>(__cvta_generic_to_shared({barrier_word(var, ctx)}))"


@register_codegen(CudaTarget, Role.EMIT, MBarrierInit)
def _emit_init(call, ctx: CudaCodegenContext) -> None:
    addr = _smem_addr(call.args[0], ctx)
    count = int(call.target.arrive_count)
    ctx.emit(
        f'asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" '
        f':: "r"({addr}), "r"({count}u));'
    )


@register_codegen(CudaTarget, Role.EMIT, MBarrierArriveExpectTx)
def _emit_arrive_expect_tx(call, ctx: CudaCodegenContext) -> None:
    addr = _smem_addr(call.args[0], ctx)
    tx = int(call.target.tx_bytes)
    body = (
        "{\\n  .reg .b64 state;\\n"
        "  mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\\n}\\n"
    )
    ctx.emit(f'asm volatile("{body}" :: "r"({addr}), "r"({tx}u));')


@register_codegen(CudaTarget, Role.EMIT, MBarrierWaitParity)
def _emit_wait_parity(call, ctx: CudaCodegenContext) -> None:
    addr = _smem_addr(call.args[0], ctx)
    phase = render_scalar_expr(call.args[1], ctx)
    ctx.emit("{")
    ctx.indent()
    ctx.emit("unsigned tilefoundry_mbarrier_ready = 0u;")
    ctx.emit("while (!tilefoundry_mbarrier_ready) {")
    ctx.indent()
    body = (
        "{\\n  .reg .pred complete;\\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 complete, [%1], %2;\\n"
        "  selp.b32 %0, 1, 0, complete;\\n}\\n"
    )
    ctx.emit(
        f'asm volatile("{body}" : "=r"(tilefoundry_mbarrier_ready) '
        f': "r"({addr}), "r"(unsigned({phase})));'
    )
    ctx.dedent()
    ctx.emit("}")
    ctx.dedent()
    ctx.emit("}")


@register_codegen(CudaTarget, Role.EMIT, MBarrierInvalidate)
def _emit_invalidate(call, ctx: CudaCodegenContext) -> None:
    addr = _smem_addr(call.args[0], ctx)
    ctx.emit(f'asm volatile("mbarrier.inval.shared::cta.b64 [%0];\\n" :: "r"({addr}));')


__all__ = ["barrier_word"]
