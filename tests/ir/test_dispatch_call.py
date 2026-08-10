"""``tir.DispatchCall`` verifier rules — the diagnosis for a malformed dispatch.

A dispatch that verifies wrongly still compiles: it reads a shape the kernel was
never given, or selects an arm by a pattern that is not a range, and the failure
surfaces as a wrong result at runtime. Each rule below is the message that
localises one such construction; the well-formed path is a runtime witness
(``tests/e2e/test_dynamic_shape_dispatch.py``) and a lowering witness
(``tests/ir/test_specialization_lowering.py``).
"""

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
    callee_count: int = 2,
):
    """Construct a module: entry PrimFunction with DispatchCall body + N callees."""
    x_entry = Var(type=_x_type(), name="x")
    callees = [
        PrimFunction(
            name=f"main$S$variant_{i}",
            params=(Var(type=_x_type(), name="x"),),
            body=Sequential(body=(Return(),)),
        )
        for i in range(callee_count)
    ]
    if subjects is None:
        subjects = (ShapeOf(type=_scalar_i32(), param=x_entry, axis=0),)
    if case_patterns is None:
        case_patterns = tuple(
            (DimVarRangePat(dim_var="S", lo=1 + 3 * i, hi=3 + 3 * i),) for i in range(callee_count)
        )
    dc = DispatchCall(
        callee_name="main",
        subjects=subjects,
        case_patterns=case_patterns,
        case_calls=tuple(symbol_call(c, (x_entry,)) for c in callees),
        fallback=fallback if fallback is not None else Sequential(body=(Abort(),)),
    )
    entry = PrimFunction(
        name="main",
        params=(x_entry,),
        body=Sequential(body=(dc,)),
    )
    return [entry, *callees]


def test_subject_must_be_a_shape_of_an_enclosing_param_axis() -> None:
    """The subject is the one value read at runtime to pick an arm.

    The subject is the one value read at runtime to pick an arm. It must be a
    ``ShapeOf``, of a param the enclosing PrimFunction actually declares (so the
    kernel is passed that extent), at an axis that param has. A stranger Var or an
    out-of-rank axis would read memory the launch never bound.
    """
    x_entry = Var(type=_x_type(), name="x")
    with pytest.raises(VerifyError, match="ShapeOf"):
        verify_module(_build_module(subjects=(x_entry,)))

    stranger = Var(type=_x_type(), name="stranger")
    with pytest.raises(VerifyError, match="not one of the enclosing"):
        verify_module(
            _build_module(
                subjects=(ShapeOf(type=_scalar_i32(), param=stranger, axis=0),),
            )
        )

    callee = PrimFunction(
        name="main$S$variant_0",
        params=(Var(type=_x_type(), name="x"),),
        body=Sequential(body=(Return(),)),
    )
    dc = DispatchCall(
        callee_name="main",
        subjects=(ShapeOf(type=_scalar_i32(), param=x_entry, axis=5),),
        case_patterns=((DimVarRangePat(dim_var="S", lo=1, hi=4),),),
        case_calls=(symbol_call(callee, (x_entry,)),),
        fallback=Sequential(body=(Abort(),)),
    )
    entry = PrimFunction(name="main", params=(x_entry,), body=Sequential(body=(dc,)))
    with pytest.raises(VerifyError, match="out of\\s+rank"):
        verify_module([entry, callee])


def test_case_patterns_must_be_one_range_per_arm() -> None:
    """Every arm is selected by a ``DimVarRangePat``.

    Every arm is selected by a ``DimVarRangePat``, and there are exactly as many
    pattern tuples as calls: a shorter list silently drops an arm, and a
    non-range pattern has no runtime comparison to lower to.
    """
    with pytest.raises(VerifyError, match="DimVarRangePat"):
        verify_module(
            _build_module(
                case_patterns=(
                    (DimVarRangePat(dim_var="S", lo=1, hi=4),),
                    (ScalarPat(),),
                )
            )
        )

    with pytest.raises(VerifyError, match="len\\(case_patterns\\)"):
        verify_module(_build_module(case_patterns=((DimVarRangePat(dim_var="S", lo=1, hi=4),),)))


def test_dispatch_call_rejects_multi_axis() -> None:
    """Dispatch selects on a single extent.

    Dispatch selects on a single extent; a second subject would need a product
    of ranges no lowering produces.
    """
    x_entry = Var(type=_x_type(), name="x")
    with pytest.raises(VerifyError, match="len\\(subjects\\) == 1"):
        verify_module(
            _build_module(
                subjects=(
                    ShapeOf(type=_scalar_i32(), param=x_entry, axis=0),
                    ShapeOf(type=_scalar_i32(), param=x_entry, axis=1),
                ),
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
        )


def test_dispatch_call_rejects_non_abort_fallback() -> None:
    """An unmatched extent must abort.

    An unmatched extent must abort. Returning instead would leave the output
    buffer untouched and look like a numerical bug.
    """
    with pytest.raises(VerifyError, match="Sequential\\(\\(Abort"):
        verify_module(_build_module(fallback=Sequential(body=(Return(),))))


def test_symbol_call_rejects_nonempty_nested() -> None:
    """Verify rejects an ``Evaluate`` whose ``nested`` is non-empty.

    Verify rejects an ``Evaluate(SymbolRef)`` whose ``nested`` is non-empty
    (nested MUST be empty under the top-level-only module).
    """
    x_callee = Var(type=_x_type(), name="x")
    callee = PrimFunction(name="callee", params=(x_callee,), body=Sequential(body=(Return(),)))
    x_entry = Var(type=_x_type(), name="x")
    bad = Evaluate(
        callable=SymbolRef(
            name="callee",
            nested=("bad",),
            type=callable_type_for_prim_function(callee),
        ),
        args=(x_entry,),
    )
    entry = PrimFunction(name="main", params=(x_entry,), body=Sequential(body=(bad,)))
    with pytest.raises(VerifyError, match="nested"):
        verify_module([entry, callee])
