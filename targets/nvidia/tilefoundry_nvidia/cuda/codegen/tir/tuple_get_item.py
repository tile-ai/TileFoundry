"""CUDA value emission for tuple indexing."""

from __future__ import annotations

from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimMod, DimMul, DimSub
from tilefoundry.ir.types.dtype import IntegerDType
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
    return ctx.name_for(expr)


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


def constant_index(index: Constant) -> int:
    """The element a constant TupleGetItem index selects."""
    selected = index.value
    if not isinstance(selected, int) or isinstance(selected, bool):
        raise ValueError("CUDA TupleGetItem constant index must be an integer")
    return selected


@register_codegen(CudaTarget, Role.EMIT, Tuple)
def _bind_tuple(let_stmt, ctx: CudaCodegenContext) -> None:
    """Declare one array for a tuple whose elements share a type."""
    elements = let_stmt.value.elements
    if not elements or any(element.type != elements[0].type for element in elements[1:]):
        return
    rendered = tuple(_value(element, ctx) for element in elements)
    type_ = elements[0].type
    name = ctx.name_for(let_stmt.var)
    values = ", ".join(rendered)
    if isinstance(type_, TensorType) and type_.shape == () and isinstance(type_.dtype, IntegerDType):
        ctx.emit(f"int64_t {name}[{len(elements)}] = {{{values}}};")
    else:
        ctx.emit(f"cute::array<decltype({rendered[0]}), {len(elements)}> {name}{{{values}}};")


@register_codegen(CudaTarget, Role.EMIT, TupleGetItem)
def _emit(let_stmt, ctx: CudaCodegenContext) -> None:
    call = let_stmt.value
    tuple_var, index = call.args
    held = tuple_var
    if isinstance(held, Var):
        held = ctx.tuple_for(held)
    if not isinstance(held, Tuple):
        raise ValueError("CUDA TupleGetItem requires a tuple value")
    result = ctx.name_for(let_stmt.var)
    if isinstance(index, Constant):
        ctx.emit(f"auto {result} = {_value(held.elements[constant_index(index)], ctx)};")
        return
    ctx.emit(f"auto {result} = {ctx.name_for(tuple_var)}[{_index(index, ctx)}];")
