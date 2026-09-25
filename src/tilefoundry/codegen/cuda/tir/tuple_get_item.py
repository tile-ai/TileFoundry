"""CUDA value emission for tuple indexing."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimMod, DimMul, DimSub
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen

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
        return ctx.name_for(expr)
    if isinstance(expr, Call):
        for op_type, symbol in _DIM_SYMBOLS.items():
            if isinstance(expr.target, op_type):
                lhs, rhs = expr.args
                return f"({_index(lhs, ctx)} {symbol} {_index(rhs, ctx)})"
    raise NotImplementedError(
        f"CUDA TupleGetItem index {type(expr).__name__} is not scalar dimension arithmetic"
    )


@register_codegen(CudaTarget, Role.EMIT, TupleGetItem)
def _emit(let_stmt, ctx: CudaCodegenContext) -> None:
    call = let_stmt.value
    held, index = call.args
    if isinstance(held, Var):
        held = ctx.tuple_for(held)
    if not isinstance(held, Tuple):
        raise ValueError("CUDA TupleGetItem requires a tuple value")
    result = ctx.name_for(let_stmt.var)
    if isinstance(index, Constant):
        selected = index.value
        if not isinstance(selected, int) or isinstance(selected, bool):
            raise ValueError("CUDA TupleGetItem constant index must be an integer")
        ctx.emit(f"auto {result} = {_value(held.elements[selected], ctx)};")
        return
    elements = ", ".join(_value(element, ctx) for element in held.elements)
    ctx.emit(f"auto {result} = cute::array{{{elements}}}[{_index(index, ctx)}];")
