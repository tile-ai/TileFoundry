"""Emitter for `tir.memory.Copy` — emits ``tilefoundry::ops::copy``.

One entry whatever the operands are: ``ops::copy`` reads width, alignment and
*count* off the two layouts, so a run-time extent is the op's business and not a
form it lacks. What is missing is on this side -- a plain operand is wrapped at
its envelope bound -- so a dynamic pair is handed views over the same data whose
layouts are their own at the run-time extents. That is what the retired
``ops::copy_n(src, dst, N)`` said with a third argument.
"""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.tir import memory
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context

from .tensor_view import emit_paired, render_layout_value, tensor_reference


def is_shard(var) -> bool:
    layout = context.materialised_layout(getattr(var.type, "layout", None))
    return isinstance(layout, types.ShardLayout)


def has_dyn_shape(var) -> bool:
    shape = getattr(getattr(var, "type", None), "shape", ())
    return expr.shape_has_dim_var(shape)


def run_time_value(value, ctx: context.Context) -> str | None:
    """*value*, a layout entry, as a C++ integer at run time; None when it cannot be stated.

    A bound dimension variable reads its extent, and dimension arithmetic over
    such entries, a stride of a row-major tile for one, is written out.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, expr.Constant) and isinstance(value.value, int):
        return str(value.value)
    if isinstance(value, expr.DimVar):
        return ctx.dynamic_extents.get(value.name)
    if isinstance(value, expr.Call):
        operands = [run_time_value(arg, ctx) for arg in value.args]
        if None in operands:
            return None
        written = iter(operands)
        return codegen.render_dim_call(value, lambda _: next(written))
    return None


def run_time_layout(var, n, ctx: context.Context) -> str:
    """The layout *var* is copied through: its own layout at the run-time extents.

    A strided or swizzled window then copies the elements it names. A value with
    no stated layout is read as *n* contiguous elements. A layout whose affine
    part has an entry ``run_time_value`` cannot state fails.
    """
    layout = context.materialised_layout(getattr(var.type, "layout", None))
    if layout is None:
        return f"cute::make_layout({n})"
    affine = types.get_nonswizzle_portion(layout)

    def known(value) -> bool:
        if isinstance(value, tuple):
            return all(known(entry) for entry in value)
        return run_time_value(value, ctx) is not None

    stated = isinstance(affine, types.Layout) and known(affine.shape) and known(affine.strides)
    if not stated or (affine is not layout and not isinstance(layout.offset, int)):
        raise NotImplementedError(f"dynamic copy: no run-time layout for {layout!r}")

    def entry(value) -> str:
        if isinstance(value, int):
            return f"cute::Int<{value}>{{}}"
        return run_time_value(value, ctx)

    return render_layout_value(layout, entry, entry)


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
        src_layout = run_time_layout(source, n, ctx)
        dst_layout = run_time_layout(destination, n, ctx)
        ctx.emit("{")
        ctx.indent()
        ctx.emit(f"auto tilefoundry_copy_src = cute::make_tensor({src}.data(), {src_layout});")
        ctx.emit(f"auto tilefoundry_copy_dst = cute::make_tensor({dst}.data(), {dst_layout});")
        ctx.emit("tilefoundry::ops::copy(tilefoundry_copy_src, tilefoundry_copy_dst);")
        ctx.dedent()
        ctx.emit("}")
        return
    emit_paired("tilefoundry::ops::copy", (source, destination), ctx)


__all__: list[str] = []
