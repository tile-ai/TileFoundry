"""CUDA emitter for ``tir.DispatchCall`` — nested ``if`` chain over cases.

One positive case: a 2-arm dispatch over ``ShapeOf(x, 1)`` produces
two predicate clauses in source order plus an ``else { assert(false); }``
fallback. ``assert(false)`` works in both host and device contexts —
``__trap()`` is ``__device__``-only and the dispatch op now emits
from host-wrapper context too (entry of a dispatch group).
"""

from __future__ import annotations

import tilefoundry.codegen.cuda  # noqa: F401  — trigger emitter autodiscovery
from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.ir.core import Var
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import Abort, Sequential
from tilefoundry.ir.tir.symbol_ref import symbol_call
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.storage import StorageKind


def _tensor_var(name: str, shape: tuple[int, ...]) -> Var:
    return Var(
        name=name,
        type=TensorType(shape=shape, dtype=DType.f32, layout=None, storage=StorageKind.GMEM),
    )


def _shape_scalar(name: str) -> Var:
    return Var(name=name, type=TensorType.scalar(dtype=DType.i32))


def _mangled_prim_func(name: str, x: Var) -> PrimFunction:
    return PrimFunction(name=name, params=(x,), body=Sequential(body=()))


def test_dispatch_call_emits_if_chain_and_trap_fallback() -> None:
    x = _tensor_var("x", (16, 4))
    x_shape_1 = _shape_scalar("x_shape_1")

    pf_lo = _mangled_prim_func("main$S$1_3", x)
    pf_hi = _mangled_prim_func("main$S$4_7", x)

    dispatch = DispatchCall(
        callee_name="main",
        subjects=(
            ShapeOf(param=x, axis=1, type=TensorType.scalar(dtype=DType.i32)),
        ),
        case_patterns=(
            (DimVarRangePat(dim_var="S", lo=1, hi=3),),
            (DimVarRangePat(dim_var="S", lo=4, hi=7),),
        ),
        case_calls=(
            symbol_call(pf_lo, (x, x_shape_1)),
            symbol_call(pf_hi, (x, x_shape_1)),
        ),
        fallback=Sequential(body=(Abort(),)),
    )

    ctx = CodegenContext()
    # Simulate the enclosing PrimFunction having registered its params,
    # so the dispatch emitter renders args under their bare names.
    ctx.register_kernel_param(x)
    ctx.register_kernel_param(x_shape_1)
    ctx.emit_node(dispatch)
    src = ctx.source()

    assert "if (((1 <= (x_shape_1)) && ((x_shape_1) < 3))) {" in src
    assert "} else if (((4 <= (x_shape_1)) && ((x_shape_1) < 7))) {" in src
    # Dispatch invokes the callees' internal C++ wrapper symbols, not
    # their user-facing names (which may collide with ``::main`` or use
    # ``$`` characters that are only valid via GCC extension).
    assert "__tilefoundry_main__S__1_3_host(x, x_shape_1);" in src
    assert "__tilefoundry_main__S__4_7_host(x, x_shape_1);" in src
    assert "} else {" in src
    assert "assert(false);" in src
