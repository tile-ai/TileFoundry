"""Emit the body of one CUDA kernel: its buffers, then its statements.

A kernel is handed pointers and the extents its parameter types leave open, so
the body starts by wrapping each pointer in the cute tensor the ops index. A
specialization prototype has no body of its own: its variants are branches of
the one kernel, chosen on the extent the pattern ranges over, because every
kernel in a translation unit runs at the geometry that unit states once.
"""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry.ir import core, pattern, types, utils
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context

from .memory import render_shard_layout_value


def wrap_buffer(signature: codegen.TensorSignature, ctx: context.Context) -> None:
    """Wrap one parameter's pointer in the tensor the body indexes it through.

    The layout is the envelope the whole dispatch range fits in, so one kernel
    covers every shape its type admits; a sharded parameter carries which part
    of that envelope this instance holds.
    """
    name = signature.name
    total = utils.shape_numel_upper_bound(signature.type.shape)
    layout = f"cute::make_layout(cute::Shape<cute::Int<{total}>>{{}})"
    pointer = f"cute::make_tensor(cute::make_gmem_ptr({name}), {layout})"
    shard = signature.type.layout
    if not isinstance(shard, types.ShardLayout):
        ctx.emit(f"auto {name}_tensor = {pointer};")
        return
    preamble, value = render_shard_layout_value(
        f"{name}_tensor", shard, ctx.dynamic_extents, signature.type.storage, ctx
    )
    for line in preamble:
        ctx.emit(line)
    ctx.emit(f"auto {name}_tensor = tilefoundry::make_shard_tensor({pointer}, {layout}, {value});")


def dispatch(fn: core.PrimFunction, ctx: context.Context) -> None:
    """Write the variants of *fn* as branches on the extent they range over.

    The extent is already a parameter of this kernel, so the choice costs one
    comparison inside the launch the host would otherwise have had to make
    before it -- and the host has nothing to gain by making it, since all the
    variants run at one geometry anyway.
    """
    ctx.bind_extents(codegen.tensor_signature_of(param) for param in fn.params)
    emit_range_dispatch(fn, ctx, ctx.emit_node)


def emit_range_dispatch(fn, ctx: context.Context, emit_branch) -> None:
    """Emit the common range selection, delegating each selected variant body."""
    for index, variant in enumerate(fn.variants):
        declared = range_over_one_dimension(fn, variant)
        subject = dispatch_subject(fn, declared, ctx)
        opening = "if" if index == 0 else "} else if"
        ctx.emit(f"{opening} (({declared.lo} <= {subject}) && ({subject} <= {declared.hi})) {{")
        ctx.indent()
        emit_branch(variant)
        ctx.dedent()
    ctx.emit("} else {")
    ctx.indent()
    ctx.emit("__trap();")
    ctx.dedent()
    ctx.emit("}")


def range_over_one_dimension(fn: core.PrimFunction, variant: core.PrimFunction) -> pattern.Range:
    """The pattern selecting *variant*, which a branch can only be a range."""
    declared = variant.specializations[0]
    if not isinstance(declared, pattern.Range):
        raise NotImplementedError(
            f"CUDA dispatch: {fn.name!r} selects {variant.name!r} by "
            f"{type(declared).__name__}; only a range over one dimension is written"
        )
    return declared


def dispatch_subject(fn: core.PrimFunction, declared: pattern.Range, ctx: context.Context) -> str:
    """Where this kernel reads the dimension *pattern* ranges over."""
    subject = ctx.dynamic_extents.get(declared.dim_var)
    if subject is None:
        raise ValueError(
            f"CUDA dispatch: {fn.name!r} selects on {declared.dim_var!r}, which "
            f"no parameter's type leaves open, so the kernel is never told it"
        )
    return subject


@codegen.register(target.Target, codegen.Role.EMIT, core.PrimFunction)
def emit(fn: core.PrimFunction, ctx: context.Context) -> None:
    """Write what runs inside one ``__global__``: its buffers, then its statements."""
    if fn.variants:
        dispatch(fn, ctx)
        return
    params = tuple(codegen.tensor_signature_of(param) for param in fn.params)
    for param in fn.params:
        ctx.register_kernel_param(param)
    ctx.bind_extents(params)
    for signature in params:
        wrap_buffer(signature, ctx)
    if ctx.codegen_context is None:
        raise ValueError("CUDA function emission requires a CodegenContext")
    if ctx.codegen_context.options.emit == "skeleton":
        ctx.emit(f"// TODO: {fn.name}")
    else:
        codegen.CppEmitter(context=ctx).visit(fn.body)


__all__ = [
    "emit_range_dispatch",
    "wrap_buffer",
]
