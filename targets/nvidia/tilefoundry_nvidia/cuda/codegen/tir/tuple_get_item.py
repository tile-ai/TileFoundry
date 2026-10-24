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


def constant_index(index: Constant) -> int:
    """The element a constant TupleGetItem index selects."""
    selected = index.value
    if not isinstance(selected, int) or isinstance(selected, bool):
        raise ValueError("CUDA TupleGetItem constant index must be an integer")
    return selected


def render_tuple_choice(rendered: tuple[str, ...], index, ctx: CudaCodegenContext) -> str:
    """One runtime choice among *rendered* elements, subscripted by *index*.

    The choice is decided on the card: one ``cute::array`` of the elements,
    subscripted by the index this function renders. ``cute::array`` has no
    deduction guide, so the element type and count come from the first
    element's spelling.
    """
    array = f"cute::array<decltype({rendered[0]}), {len(rendered)}>{{{', '.join(rendered)}}}"
    return f"{array}[{_index(index, ctx)}]"


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
        ctx.emit(f"auto {result} = {_value(held.elements[constant_index(index)], ctx)};")
        return
    rendered = tuple(_value(element, ctx) for element in held.elements)
    ctx.emit(f"auto {result} = {render_tuple_choice(rendered, index, ctx)};")
