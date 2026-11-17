"""Emitter for `tir.memory.Copy` — emits ``tilefoundry::ops::copy``.

One entry whatever the operands are: ``ops::copy`` reads width, alignment and
*count* off the two layouts, so a run-time extent is the op's business and not a
form it lacks. What is missing is on this side -- a plain operand is wrapped at
its envelope bound -- so a dynamic pair is handed a view over the same data
whose layout is the run-time length. That is what the retired
``ops::copy_n(src, dst, N)`` said with a third argument.
"""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.tir import memory
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context

from .tensor_view import emit_paired, tensor_reference


def is_shard(var) -> bool:
    return isinstance(getattr(var.type, "layout", None), types.ShardLayout)


def has_dyn_shape(var) -> bool:
    shape = getattr(getattr(var, "type", None), "shape", ())
    return expr.shape_has_dim_var(shape)


@codegen.register(target.Target, codegen.Role.EMIT, memory.Copy)
def emit(call, ctx: context.Context) -> None:
    source, destination = call.args[0], call.args[1]
    src_shard = is_shard(source)
    dst_shard = is_shard(destination)
    dyn = has_dyn_shape(source) or has_dyn_shape(destination)
    if dyn and not src_shard and not dst_shard:
        src = tensor_reference(source, ctx)
        dst = tensor_reference(destination, ctx)
        n = utils.shape_runtime_total(destination.type.shape, ctx.dynamic_extents)


        ctx.emit("{")
        ctx.indent()
        ctx.emit(f"auto tilefoundry_copy_n = cute::make_layout({n});")
        ctx.emit(f"auto tilefoundry_copy_src = cute::make_tensor({src}.data(), tilefoundry_copy_n);")
        ctx.emit(f"auto tilefoundry_copy_dst = cute::make_tensor({dst}.data(), tilefoundry_copy_n);")
        ctx.emit("tilefoundry::ops::copy(tilefoundry_copy_src, tilefoundry_copy_dst);")
        ctx.dedent()
        ctx.emit("}")
        return
    emit_paired("tilefoundry::ops::copy", (source, destination), ctx)


__all__: list[str] = []
