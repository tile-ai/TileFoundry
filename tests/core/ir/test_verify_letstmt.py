"""Verify hard constraints on LetStmt.

All three are conditions a lowering pass can produce while emitting perfectly
plausible TIR: a rebound Var makes two distinct buffers one name, a declared type
that disagrees with its value places a tensor in the wrong memory, and an
allocation nested inside another expression has no defined lifetime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import load_program
from tilefoundry.ir.core import Call, Constant, Var, VerifyError
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.memory.ptr_of import PtrOf
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import For, If, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import DType, Layout, Mesh, TensorType, Topology, make_tensor_type
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.passes.transforms.convert_hir_to_tir import View
from tilefoundry.visitor_registry.verify import verify_prim_function


def _alloc_call(t: TensorType) -> Call:
    return Call(type=t, target=AllocTensor(tensor_type=t), args=())


def _let(var: Var, value: Call) -> LetStmt:
    return LetStmt(var=var, value=value)


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

    nested = Sequential(body=(
        _let(v, _alloc_call(rmem)),
        Sequential(body=(_let(v, _alloc_call(rmem)),)),
    ))
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
    t_scalar = TensorType.scalar(DType.f32, storage=StorageKind.RMEM)

    nested = Call(
        type=t_scalar,
        target=PtrOf(),
        args=(_alloc_call(t_scalar),),
    )
    with pytest.raises(VerifyError, match="AllocTensor"):
        verify_prim_function(_pf(_let(Var(type=t_scalar, name="v"), nested)))


@pytest.mark.parametrize("scope", ("let", "if", "for", "mesh"))
def test_var_use_must_be_inside_its_binding_scope(scope):
    """A sibling statement cannot consume a variable held only inside a scope."""
    scalar = make_tensor_type((), DType.i32, storage="rmem")
    local = Var(type=scalar, name="local")
    zero = Constant(type=scalar, value=0)
    one = Constant(type=scalar, value=1)
    empty = Sequential(body=())
    if scope == "let":
        binding = Sequential(body=(LetStmt(var=local, value=zero),))
    elif scope == "if":
        condition = Constant(
            type=make_tensor_type((), DType.bool, storage="rmem"), value=True
        )
        binding = If(
            cond=condition,
            then_body=Sequential(body=(LetStmt(var=local, value=zero),)),
            else_body=empty,
        )
    elif scope == "for":
        binding = For(induction_var=local, start=zero, stop=one, step=one, body=empty)
    else:
        mesh = Mesh((Topology("thread", 1),), Layout((1,), (1,)))
        binding = MeshScope(mesh=mesh, binding=local, body=empty)
    consumer = LetStmt(var=Var(type=scalar, name="escaped"), value=local)
    with pytest.raises(VerifyError, match="local.*used outside the scope that binds it"):
        verify_prim_function(_pf(binding, consumer))


@pytest.mark.parametrize("self_reference", (False, True))
def test_letstmt_value_cannot_use_an_unbound_var(self_reference):
    """A value cannot read its own binding or a later statement's binding."""
    scalar = make_tensor_type((), DType.i32, storage="rmem")
    local = Var(type=scalar, name="local")
    consumer = LetStmt(
        var=local if self_reference else Var(type=scalar, name="early"),
        value=local,
    )
    binding = LetStmt(var=local, value=Constant(type=scalar, value=0))
    with pytest.raises(VerifyError, match="local.*used outside the scope that binds it"):
        verify_prim_function(_pf(consumer, binding))


def test_verify_rejects_a_non_ir_node():
    """Lowering recipes must be materialized before entering TIR expression fields."""
    tensor = make_tensor_type((4,), storage="rmem")
    source = Var(type=tensor, name="source")
    recipe = View(base=source, starts=(), sizes=(4,), type=tensor)
    function = PrimFunction(
        name="recipe",
        params=(source,),
        body=Sequential(body=(
            LetStmt(
                var=Var(type=tensor, name="leaked"),
                value=recipe,
            ),
        )),
    )
    with pytest.raises(VerifyError, match="non-IR node View"):
        verify_prim_function(function)


def test_variant_cannot_use_its_parents_parameter():
    """Each specialization owns its parameters; the parent's Var is invisible."""
    program = load_program(Path(__file__).parents[1] / "fixtures/tir/parent_parameter.py")
    function = program.entry_function()
    variant = function.variants[0]
    assert variant.params[0] is not function.params[0]
    variant.body = Sequential(body=(LetStmt(
        var=Var(type=function.params[0].type, name="escaped"),
        value=function.params[0],
    ),))
    with pytest.raises(VerifyError, match="x.*used outside the scope that binds it"):
        verify_prim_function(function)
