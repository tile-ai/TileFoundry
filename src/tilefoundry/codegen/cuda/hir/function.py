"""HIR skeleton device bodies and the kernel body that calls them."""

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.cuda.tir.prim_function import emit_range_dispatch, wrap_buffer
from tilefoundry.codegen.linkable_function import LinkableFunction
from tilefoundry.ir.core.module import function_scopes, reachable_functions
from tilefoundry.ir.hir.function import Function
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


def hir_device_function(fn: Function, ctx: CudaCodegenContext) -> LinkableFunction:
    """One editable device function receiving the kernel's cute parameter views."""
    resolved = ctx.codegen_context.symbols.by_function[id(fn)]
    params = ", ".join(f"auto {param.name}_tensor" for param in resolved.callable.params)
    definition = f"__device__ void {resolved.symbol}({params}) {{\n  // TODO: {fn.name}\n}}"
    return LinkableFunction(name=fn.name, declaration="", definition=definition)


@register_codegen(CudaTarget, Role.EMIT, Function)
def _emit(fn: Function, ctx: CudaCodegenContext) -> None:
    """Wrap incoming buffers, then select the device body on the runtime extent."""
    params = ctx.signature_of(fn).params
    ctx.dynamic_extents.clear()
    ctx.bind_extents(params)
    for signature in params:
        wrap_buffer(signature, ctx)
    arguments = ", ".join(f"{param.name}_tensor" for param in params)
    symbols = ctx.codegen_context.symbols.by_function
    if not fn.variants:
        ctx.emit(f"{symbols[id(fn)].symbol}({arguments});")
        return
    emit_range_dispatch(
        fn, ctx, lambda variant: ctx.emit(f"{symbols[id(variant)].symbol}({arguments});")
    )


def device_functions(kernels, ctx) -> tuple[LinkableFunction, ...]:
    """Emit each HIR function and variant, with callees before callers."""
    hir_functions = tuple(fn for fn in kernels if isinstance(fn, Function))
    device_bodies = tuple(
        fn for prototype in hir_functions for fn in (prototype, *prototype.variants)
    )
    scopes = {
        id(each): scope
        for _selector, scope in function_scopes(ctx.codegen_context.module)
        for each in (scope.function, *scope.function.variants)
    }
    device_bodies = tuple(
        sorted(device_bodies, key=lambda fn: len(reachable_functions(fn, scopes[id(fn)])))
    )
    return tuple(hir_device_function(fn, ctx) for fn in device_bodies)
