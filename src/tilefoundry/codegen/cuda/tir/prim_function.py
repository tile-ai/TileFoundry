"""Emit the body of one CUDA kernel: its buffers, then its statements.

A kernel is handed pointers and the extents its parameter types leave open, so
the body starts by wrapping each pointer in the cute tensor the ops index. A
specialization prototype has no body of its own: its variants are branches of
the one kernel, chosen on the extent the pattern ranges over, because every
kernel in a translation unit runs at the geometry that unit states once.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.cuda.tir.memory.tensor_view import render_shard_layout_value
from tilefoundry.codegen.emitter import CudaEmitter
from tilefoundry.codegen.signature import TensorSignature, tensor_signature_of
from tilefoundry.ir.pattern import RangePattern
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.ir.types.utils import shape_numel_upper_bound
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


def _wrap_buffer(signature: TensorSignature, ctx: CudaCodegenContext) -> None:
    """Wrap one parameter's pointer in the tensor the body indexes it through.

    The layout is the envelope the whole dispatch range fits in, so one kernel
    covers every shape its type admits; a sharded parameter carries which part
    of that envelope this instance holds.
    """
    name = signature.name
    total = shape_numel_upper_bound(signature.type.shape)
    layout = f"cute::make_layout(cute::Shape<cute::Int<{total}>>{{}})"
    pointer = f"cute::make_tensor(cute::make_gmem_ptr({name}), {layout})"
    shard = signature.type.layout
    if not isinstance(shard, ShardLayout):
        ctx.emit(f"auto {name}_tensor = {pointer};")
        return
    preamble, value = render_shard_layout_value(
        f"{name}_tensor", shard, ctx.dynamic_extents, signature.type.storage, ctx
    )
    for line in preamble:
        ctx.emit(line)
    ctx.emit(f"auto {name}_tensor = tilefoundry::make_shard_tensor({pointer}, {layout}, {value});")


def _dispatch(fn: PrimFunction, ctx: CudaCodegenContext) -> None:
    """Write the variants of *fn* as branches on the extent they range over.

    The extent is already a parameter of this kernel, so the choice costs one
    comparison inside the launch the host would otherwise have had to make
    before it -- and the host has nothing to gain by making it, since all the
    variants run at one geometry anyway.
    """
    ctx.bind_extents(tensor_signature_of(param) for param in fn.params)
    for index, variant in enumerate(fn.variants):
        pattern = _range_over_one_dimension(fn, variant)
        subject = _subject(fn, pattern, ctx)
        opening = "if" if index == 0 else "} else if"
        ctx.emit(f"{opening} (({pattern.lo} <= {subject}) && ({subject} <= {pattern.hi})) {{")
        ctx.indent()
        ctx.emit_node(variant)
        ctx.dedent()
    ctx.emit("} else {")
    ctx.indent()
    ctx.emit("__trap();")
    ctx.dedent()
    ctx.emit("}")


def _range_over_one_dimension(fn: PrimFunction, variant: PrimFunction) -> RangePattern:
    """The pattern selecting *variant*, which a branch can only be a range."""
    pattern = variant.specializations[0]
    if not isinstance(pattern, RangePattern):
        raise NotImplementedError(
            f"CUDA dispatch: {fn.name!r} selects {variant.name!r} by "
            f"{type(pattern).__name__}; only a range over one dimension is written"
        )
    return pattern


def _subject(fn: PrimFunction, pattern: RangePattern, ctx: CudaCodegenContext) -> str:
    """Where this kernel reads the dimension *pattern* ranges over."""
    subject = ctx.dynamic_extents.get(pattern.dim_var)
    if subject is None:
        raise ValueError(
            f"CUDA dispatch: {fn.name!r} selects on {pattern.dim_var!r}, which "
            f"no parameter's type leaves open, so the kernel is never told it"
        )
    return subject


@register_codegen(CudaTarget, Role.EMIT, PrimFunction)
def _emit(fn: PrimFunction, ctx: CudaCodegenContext) -> None:
    """Write what runs inside one ``__global__``: its buffers, then its statements."""
    ctx.reset_smem_base()
    if fn.variants:
        _dispatch(fn, ctx)
        return
    params = tuple(tensor_signature_of(param) for param in fn.params)
    for param in fn.params:
        ctx.register_kernel_param(param)
    ctx.bind_extents(params)
    for signature in params:
        _wrap_buffer(signature, ctx)
    CudaEmitter(context=ctx).visit(fn.body)
