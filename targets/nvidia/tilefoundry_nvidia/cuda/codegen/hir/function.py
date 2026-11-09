"""HIR skeleton device bodies and the kernel body that calls them."""

from tilefoundry import codegen
from tilefoundry.ir import core
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context, tir


def device_function(fn: core.Function, ctx: context.Context) -> codegen.LinkableFunction:
    """One editable device function receiving the kernel's cute parameter views."""
    resolved = ctx.codegen_context.symbols.by_function[id(fn)]
    params = ", ".join(f"auto {param.name}_tensor" for param in resolved.callable.params)
    definition = f"__device__ void {resolved.symbol}({params}) {{\n  // TODO: {fn.name}\n}}"
    return codegen.LinkableFunction(name=fn.name, declaration="", definition=definition)


@codegen.register(target.Target, codegen.Role.EMIT, core.Function)
def emit(fn: core.Function, ctx: context.Context) -> None:
    """Wrap incoming buffers, then select the device body on the runtime extent."""
    params = ctx.signature_of(fn).params
    ctx.dynamic_extents.clear()
    ctx.bind_extents(params)
    for signature in params:
        tir.wrap_buffer(signature, ctx)
    arguments = ", ".join(f"{param.name}_tensor" for param in params)
    symbols = ctx.codegen_context.symbols.by_function
    if not fn.variants:
        ctx.emit(f"{symbols[id(fn)].symbol}({arguments});")
        return
    tir.emit_range_dispatch(
        fn, ctx, lambda variant: ctx.emit(f"{symbols[id(variant)].symbol}({arguments});")
    )


def device_functions(kernels, ctx) -> tuple[codegen.LinkableFunction, ...]:
    """Emit each HIR function and variant, in declaration order."""
    hir_functions = tuple(fn for fn in kernels if isinstance(fn, core.Function))
    device_bodies = tuple(
        fn for prototype in hir_functions for fn in (prototype, *prototype.variants)
    )
    return tuple(device_function(fn, ctx) for fn in device_bodies)


__all__ = ["device_functions"]
