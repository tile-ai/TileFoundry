"""HIR skeleton device bodies and the kernel body that calls them."""

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.cuda.tir.prim_function import emit_range_dispatch, wrap_buffer
from tilefoundry.codegen.linkable_function import LinkableFunction
from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.core.module import ModuleList
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.visitor import ExprWalker
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.contexts import CallFrame
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
    frames = {
        id(each): frame
        for frame in list_call_paths(ctx.codegen_context.module).values()
        for each in (frame.function, *frame.function.variants)
    }

    class Dependencies(ExprWalker):
        def __init__(self):
            super().__init__()
            self.functions = set()

        def function(self, fn, frame):
            if id(fn) in self.functions:
                return
            self.functions.add(id(fn))
            if fn.body is not None:
                self.visit(fn.body, frame)

        def visit_Call(self, call, frame):
            self.visit_operands(call, frame)
            target = call.target
            if isinstance(target, Function):
                self.function(target, frame)
                return
            slot = target if isinstance(target, ModuleList) else (
                target.args[0] if isinstance(target, Call) and isinstance(target.target, TupleGetItem)
                and isinstance(target.args[0], ModuleList) else None
            )
            if slot is not None:
                slot = frame.module.children[slot.key]
                index = target.args[1] if isinstance(target, Call) else None
                index = index.value if isinstance(index, Constant) else 0
                child = slot.elements[index]
                callee = child.entry_function()
                self.function(callee, CallFrame(
                    child, callee, child.target if child.target is not None else frame.target,
                    child.topologies if child.topologies is not None else frame.topologies,
                ))

    sizes = {}
    for fn in device_bodies:
        walk = Dependencies()
        walk.function(fn, frames[id(fn)])
        sizes[id(fn)] = len(walk.functions)
    device_bodies = tuple(sorted(device_bodies, key=lambda fn: sizes[id(fn)]))
    return tuple(hir_device_function(fn, ctx) for fn in device_bodies)
