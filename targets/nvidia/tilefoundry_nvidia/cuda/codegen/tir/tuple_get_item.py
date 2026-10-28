"""CUDA value emission for tuple indexing."""

from __future__ import annotations

from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimMod, DimMul, DimSub
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_nvidia.cuda import CudaTarget
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext, scalar_read

_DIM_SYMBOLS = {
    DimAdd: "+",
    DimSub: "-",
    DimMul: "*",
    DimFloorDiv: "/",
    DimMod: "%",
}


def _value(expr, ctx: CudaCodegenContext) -> str:
    if isinstance(expr, Constant):
        if isinstance(expr.value, bool):
            return "true" if expr.value else "false"
        return str(expr.value)
    name = ctx.name_for(expr)
    return f"{name}_tensor" if ctx.is_kernel_param(expr) else name


def _index(expr, ctx: CudaCodegenContext) -> str:
    if isinstance(expr, Constant) and isinstance(expr.value, int):
        return str(expr.value)
    if isinstance(expr, Var):
        type_ = getattr(expr, "type", None)
        if getattr(type_, "storage", None) in (None, StorageKind.UMAT):
            return ctx.name_for(expr)
        return scalar_read(expr, ctx)
    if isinstance(expr, Call):
        for op_type, symbol in _DIM_SYMBOLS.items():
            if isinstance(expr.target, op_type):
                lhs, rhs = expr.args
                return f"({_index(lhs, ctx)} {symbol} {_index(rhs, ctx)})"
    raise NotImplementedError(
        f"CUDA TupleGetItem index {type(expr).__name__} is not scalar dimension arithmetic"
    )


def _constant_index(index: Constant) -> int:
    """The element a constant TupleGetItem index selects."""
    selected = index.value
    if not isinstance(selected, int) or isinstance(selected, bool):
        raise ValueError("CUDA TupleGetItem constant index must be an integer")
    return selected


@register_codegen(CudaTarget, Role.EMIT, Tuple)
def _bind_tuple(let_stmt, ctx: CudaCodegenContext) -> None:
    """Declare one C++ variable for every tuple binding."""
    elements = let_stmt.value.elements
    rendered = tuple(_value(element, ctx) for element in elements)
    name, values = ctx.name_for(let_stmt.var), ", ".join(rendered)
    if elements and all(element.type == elements[0].type for element in elements[1:]):
        ctx.emit(f"cute::array<decltype({rendered[0]}), {len(elements)}> {name}{{{values}}};")
    else:
        ctx.emit(f"auto {name} = cute::make_tuple({values});")


@register_codegen(CudaTarget, Role.EMIT, TupleGetItem)
def _emit(let_stmt, ctx: CudaCodegenContext) -> None:
    tuple_var, index = let_stmt.value.args
    held, result = ctx.name_for(tuple_var), ctx.name_for(let_stmt.var)
    if isinstance(index, Constant):
        ctx.emit(f"auto {result} = cute::get<{_constant_index(index)}>({held});")
        return
    ctx.emit(f"auto {result} = {held}[{_index(index, ctx)}];")
