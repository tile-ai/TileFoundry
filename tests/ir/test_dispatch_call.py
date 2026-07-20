"""Verify ``tir.DispatchCall`` IR op + verifier rules + viewer rendering."""
from __future__ import annotations

import pytest

from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.core.pattern import DimVarRangePat, ScalarPat
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import Abort, Evaluate, Return, Sequential
from tilefoundry.ir.tir.symbol_ref import SymbolRef, symbol_call
from tilefoundry.ir.tir.verify import verify_module
from tilefoundry.ir.types import DType, TensorType, callable_type_for_prim_function


def _x_type() -> TensorType:
    return TensorType(shape=(4,), dtype=DType.f32, layout=None, storage="gmem")


def _scalar_i32() -> TensorType:
    return TensorType.scalar(dtype=DType.i32)


def _build_module(
    *,
    subjects=None,
    case_patterns=None,
    fallback=None,
    case_calls=None,
    callee_count: int = 2,
):
    """Construct a module: entry PrimFunction with DispatchCall body + N callees."""
    x_entry = Var(type=_x_type(), name="x")
    callees = []
    for i in range(callee_count):
        x_callee = Var(type=_x_type(), name="x")
        pf = PrimFunction(
            name=f"main$S$variant_{i}",
            params=(x_callee,),
            body=Sequential(body=(Return(),)),
        )
        callees.append(pf)
    so = ShapeOf(type=_scalar_i32(), param=x_entry, axis=0)
    if subjects is None:
        subjects = (so,)
    if case_patterns is None:
        case_patterns = tuple(
            # Non-overlapping closed ranges: [1,3], [4,6], [7,9], ...
            (DimVarRangePat(dim_var="S", lo=1 + 3 * i, hi=3 + 3 * i),)
            for i in range(callee_count)
        )
    if case_calls is None:
        case_calls = tuple(
            symbol_call(callees[i], (x_entry,))
            for i in range(callee_count)
        )
    if fallback is None:
        fallback = Sequential(body=(Abort(),))
    dc = DispatchCall(
        callee_name="main",
        subjects=subjects,
        case_patterns=case_patterns,
        case_calls=case_calls,
        fallback=fallback,
    )
    entry = PrimFunction(
        name="main",
        params=(x_entry,),
        body=Sequential(body=(dc,)),
    )
    return [entry, *callees], dc


def test_symbol_call_rejects_nonempty_nested() -> None:
    """verify rejects an ``Evaluate(SymbolRef)`` whose ``nested`` is non-empty
    (tir.md §9: nested MUST be empty under the top-level-only module)."""
    x_callee = Var(type=_x_type(), name="x")
    callee = PrimFunction(
        name="callee", params=(x_callee,), body=Sequential(body=(Return(),))
    )
    x_entry = Var(type=_x_type(), name="x")
    bad = Evaluate(
        callable=SymbolRef(
            name="callee",
            nested=("bad",),
            type=callable_type_for_prim_function(callee),
        ),
        args=(x_entry,),
    )
    entry = PrimFunction(
        name="main", params=(x_entry,), body=Sequential(body=(bad,))
    )
    with pytest.raises(VerifyError, match="nested"):
        verify_module([entry, callee])


def test_dispatch_call_positive():
    fns, _ = _build_module()
    verify_module(fns)


def test_dispatch_call_rejects_non_shapeof_subject():
    x_entry = Var(type=_x_type(), name="x")
    fns, _ = _build_module(subjects=(x_entry,))
    with pytest.raises(VerifyError, match="ShapeOf"):
        verify_module(fns)


def test_dispatch_call_rejects_non_dimvarrangepat():
    fns, _ = _build_module(
        case_patterns=(
            (DimVarRangePat(dim_var="S", lo=1, hi=4),),
            (ScalarPat(),),
        ),
    )
    with pytest.raises(VerifyError, match="DimVarRangePat"):
        verify_module(fns)


def test_dispatch_call_rejects_length_mismatch():
    fns, _ = _build_module(
        case_patterns=(
            (DimVarRangePat(dim_var="S", lo=1, hi=4),),
        ),
    )
    with pytest.raises(VerifyError, match="len\\(case_patterns\\)"):
        verify_module(fns)


def test_dispatch_call_rejects_multi_axis():
    x_entry = Var(type=_x_type(), name="x")
    so1 = ShapeOf(type=_scalar_i32(), param=x_entry, axis=0)
    so2 = ShapeOf(type=_scalar_i32(), param=x_entry, axis=1)
    fns, _ = _build_module(
        subjects=(so1, so2),
        case_patterns=(
            (
                DimVarRangePat(dim_var="S", lo=1, hi=4),
                DimVarRangePat(dim_var="T", lo=1, hi=4),
            ),
            (
                DimVarRangePat(dim_var="S", lo=4, hi=7),
                DimVarRangePat(dim_var="T", lo=4, hi=7),
            ),
        ),
    )
    with pytest.raises(VerifyError, match="len\\(subjects\\) == 1"):
        verify_module(fns)


def test_dispatch_call_rejects_empty_fallback():
    fns, _ = _build_module(fallback=Sequential(body=()))
    with pytest.raises(VerifyError, match="Sequential\\(\\(Abort"):
        verify_module(fns)


def test_dispatch_call_rejects_non_abort_fallback():
    fns, _ = _build_module(fallback=Sequential(body=(Return(),)))
    with pytest.raises(VerifyError, match="Sequential\\(\\(Abort"):
        verify_module(fns)


def test_dispatch_call_rejects_non_param_shape_of_subject():
    """ShapeOf.param must be one of the enclosing PrimFunction's params."""
    stranger = Var(type=_x_type(), name="stranger")
    bad_subject = ShapeOf(type=_scalar_i32(), param=stranger, axis=0)
    fns, _ = _build_module(subjects=(bad_subject,))
    with pytest.raises(VerifyError, match="not one of the enclosing"):
        verify_module(fns)


def test_dispatch_call_rejects_out_of_rank_shape_of_axis():
    """ShapeOf.axis must satisfy 0 <= axis < len(param.type.shape).

    The check is contextual against the enclosing PrimFunction, so we
    build the module manually to thread the same Var identity through
    both the entry's params and the ShapeOf subject.
    """
    x_entry = Var(type=_x_type(), name="x")
    bad_subject = ShapeOf(type=_scalar_i32(), param=x_entry, axis=5)
    x_callee = Var(type=_x_type(), name="x")
    callee = PrimFunction(
        name="main$S$variant_0",
        params=(x_callee,),
        body=Sequential(body=(Return(),)),
    )
    dc = DispatchCall(
        callee_name="main",
        subjects=(bad_subject,),
        case_patterns=((DimVarRangePat(dim_var="S", lo=1, hi=4),),),
        case_calls=(symbol_call(callee, (x_entry,)),),
        fallback=Sequential(body=(Abort(),)),
    )
    entry = PrimFunction(
        name="main",
        params=(x_entry,),
        body=Sequential(body=(dc,)),
    )
    with pytest.raises(VerifyError, match="out of\\s+rank"):
        verify_module([entry, callee])
