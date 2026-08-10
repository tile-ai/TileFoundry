"""Verify hard constraints on LetStmt.

All three are conditions a lowering pass can produce while emitting perfectly
plausible TIR: a rebound Var makes two distinct buffers one name, a declared type
that disagrees with its value places a tensor in the wrong memory, and an
allocation nested inside another expression has no defined lifetime.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Call, Var, VerifyError
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.memory.ptr_of import PtrOf
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import LetStmt, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, TensorType, make_tensor_type


def _alloc_call(t: TensorType) -> Call:
    return Call(type=t, target=AllocTensor(tensor_type=t), args=())


def _let(var: Var, value: Call) -> LetStmt:
    return LetStmt(var=var, value=value, body=Sequential(body=(Return(),)))


def _pf(*stmts) -> PrimFunction:
    return PrimFunction(name="fn", params=(), body=Sequential(body=stmts))


def test_letstmt_requires_a_fresh_var_anywhere_in_the_function():
    """Test letstmt requires a fresh var anywhere in the function.

    Binding the same Var object twice must raise whether the second binding is
    nested inside the first or a sibling of it — fresh-Var applies across the whole
    function, not merely within the current lexical scope.
    """
    rmem = make_tensor_type((4,), storage="rmem")
    v = Var(type=rmem, name="v")

    nested = LetStmt(
        var=v,
        value=_alloc_call(rmem),
        body=Sequential(body=(_let(v, _alloc_call(rmem)),)),
    )
    with pytest.raises(VerifyError, match="fresh Var"):
        verify_prim_function(_pf(nested))

    with pytest.raises(VerifyError, match="fresh Var"):
        verify_prim_function(_pf(_let(v, _alloc_call(rmem)), _let(v, _alloc_call(rmem))))


def test_letstmt_rejects_type_mismatch():
    """var.type must equal type_of(value)."""
    t_reg = make_tensor_type((4,), storage="rmem")
    t_shared = make_tensor_type((4,), DType.f32, storage="smem")
    v = Var(type=t_shared, name="v")
    with pytest.raises(VerifyError, match="!= value.type"):
        verify_prim_function(_pf(_let(v, _alloc_call(t_reg))))


def test_letstmt_rejects_alloc_nested_in_other_expr():
    """Call may only appear directly as LetStmt.value, never nested inside another Expr operand.

    Call(AllocTensor, ...) may only appear directly as
    LetStmt.value, never nested inside another Expr operand.
    """
    t_scalar = TensorType.scalar(DType.f32)

    nested = Call(
        type=t_scalar,
        target=PtrOf(),
        args=(_alloc_call(t_scalar),),
    )
    with pytest.raises(VerifyError, match="AllocTensor"):
        verify_prim_function(_pf(_let(Var(type=t_scalar, name="v"), nested)))
