"""CUDA value emission for tuple indexing."""

from __future__ import annotations

from types import MappingProxyType

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry.ir.hir import tensor
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context

DIM_SYMBOLS = MappingProxyType({
    expr.DimAdd: "+",
    expr.DimSub: "-",
    expr.DimMul: "*",
    expr.DimFloorDiv: "/",
    expr.DimMod: "%",
})


def value(node, ctx: context.Context) -> str:
    if isinstance(node, expr.Constant):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        return str(node.value)
    name = ctx.name_for(node)
    return f"{name}_tensor" if ctx.is_kernel_param(node) else name


def scalar_index(node, ctx: context.Context) -> str:
    if isinstance(node, expr.Constant) and isinstance(node.value, int):
        return str(node.value)
    if isinstance(node, expr.Var):
        type_ = getattr(node, "type", None)
        if getattr(type_, "storage", None) in (None, expr.StorageKind.UMAT):
            return ctx.name_for(node)
        return context.scalar_read(node, ctx)
    if isinstance(node, expr.Call):
        for op_type, symbol in DIM_SYMBOLS.items():
            if isinstance(node.target, op_type):
                lhs, rhs = node.args
                return f"({scalar_index(lhs, ctx)} {symbol} {scalar_index(rhs, ctx)})"
    raise NotImplementedError(
        f"CUDA TupleGetItem index {type(node).__name__} is not scalar dimension arithmetic"
    )


def constant_index(index: expr.Constant) -> int:
    """The element a constant TupleGetItem index selects."""
    selected = index.value
    if not isinstance(selected, int) or isinstance(selected, bool):
        raise ValueError("CUDA TupleGetItem constant index must be an integer")
    return selected


@codegen.register(target.Target, codegen.Role.EMIT, expr.Tuple)
def bind_tuple(let_stmt, ctx: context.Context) -> None:
    """Declare one C++ variable for every tuple binding."""
    elements = let_stmt.value.elements
    rendered = tuple(value(element, ctx) for element in elements)
    name, values = ctx.name_for(let_stmt.var), ", ".join(rendered)
    if elements and all(element.type == elements[0].type for element in elements[1:]):
        ctx.emit(f"cute::array<decltype({rendered[0]}), {len(elements)}> {name}{{{values}}};")
    else:
        ctx.emit(f"auto {name} = cute::make_tuple({values});")


@codegen.register(target.Target, codegen.Role.EMIT, tensor.TupleGetItem)
def emit(let_stmt, ctx: context.Context) -> None:
    tuple_var, index = let_stmt.value.args
    held, result = ctx.name_for(tuple_var), ctx.name_for(let_stmt.var)
    if isinstance(index, expr.Constant):
        ctx.emit(f"auto {result} = cute::get<{constant_index(index)}>({held});")
        return
    ctx.emit(f"auto {result} = {held}[{scalar_index(index, ctx)}];")


__all__: list[str] = []
