"""Verify hard constraints on LetStmt.

All three are conditions a lowering pass can produce while emitting perfectly
plausible TIR: a rebound Var makes two distinct buffers one name, a declared type
that disagrees with its value places a tensor in the wrong memory, and an
allocation nested inside another expression has no defined lifetime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry import cli, verifier
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.tir import memory
from tilefoundry.passes import transforms


def _alloc_call(t: types.TensorType) -> expr.Call:
    return expr.Call(type=t, target=memory.AllocTensor(tensor_type=t), args=())


def _let(var: expr.Var, value: expr.Call) -> core.LetStmt:
    return core.LetStmt(var=var, value=value)


def _pf(*stmts) -> core.PrimFunction:
    return core.PrimFunction(name="fn", params=(), body=core.Sequential(body=stmts))


def test_letstmt_requires_fresh_var():
    """A LetStmt binds a fresh Var anywhere in the function.

    Binding the same Var object twice must raise whether the second binding is
    nested inside the first or a sibling of it — fresh-Var applies across the whole
    function, not merely within the current lexical scope.
    """
    rmem = types.TensorType.from_shape((4,), storage="rmem")
    v = expr.Var(type=rmem, name="v")

    nested = core.Sequential(body=(
        _let(v, _alloc_call(rmem)),
        core.Sequential(body=(_let(v, _alloc_call(rmem)),)),
    ))
    with pytest.raises(expr.VerifyError, match="fresh Var"):
        verifier.verify_prim_function(_pf(nested))

    with pytest.raises(expr.VerifyError, match="fresh Var"):
        verifier.verify_prim_function(_pf(_let(v, _alloc_call(rmem)), _let(v, _alloc_call(rmem))))


def test_letstmt_rejects_type_mismatch():
    """var.type must equal type_of(value)."""
    t_reg = types.TensorType.from_shape((4,), storage="rmem")
    t_shared = types.TensorType.from_shape((4,), expr.DType.f32, storage="smem")
    v = expr.Var(type=t_shared, name="v")
    with pytest.raises(expr.VerifyError, match="!= value.type"):
        verifier.verify_prim_function(_pf(_let(v, _alloc_call(t_reg))))


def test_letstmt_rejects_nested_alloc():
    """Call may only appear directly as LetStmt.value, never nested inside another Expr operand.

    Call(AllocTensor, ...) may only appear directly as
    LetStmt.value, never nested inside another Expr operand.
    """
    t_scalar = types.TensorType.from_scalar(expr.DType.f32, storage=expr.StorageKind.RMEM)

    nested = expr.Call(
        type=t_scalar,
        target=memory.PtrOf(),
        args=(_alloc_call(t_scalar),),
    )
    with pytest.raises(expr.VerifyError, match="AllocTensor"):
        verifier.verify_prim_function(_pf(_let(expr.Var(type=t_scalar, name="v"), nested)))


@pytest.mark.parametrize("scope", ("let", "if", "for", "mesh"))
def test_var_escape_from_scope_refused(scope):
    """A sibling statement cannot consume a variable held only inside a scope."""
    scalar = types.TensorType.from_shape((), expr.DType.i32, storage="rmem")
    local = expr.Var(type=scalar, name="local")
    zero = expr.Constant(type=scalar, value=0)
    one = expr.Constant(type=scalar, value=1)
    empty = core.Sequential(body=())
    if scope == "let":
        binding = core.Sequential(body=(core.LetStmt(var=local, value=zero),))
    elif scope == "if":
        condition = expr.Constant(
            type=types.TensorType.from_shape((), expr.DType.bool, storage="rmem"), value=True
        )
        binding = core.If(
            cond=condition,
            then_body=core.Sequential(body=(core.LetStmt(var=local, value=zero),)),
            else_body=empty,
        )
    elif scope == "for":
        binding = core.For(induction_var=local, start=zero, stop=one, step=one, body=empty)
    else:
        mesh = types.Mesh((types.Topology("thread", 1),), types.Layout((1,), (1,)))
        binding = core.MeshScope(mesh=mesh, binding=local, body=empty)
    consumer = core.LetStmt(var=expr.Var(type=scalar, name="escaped"), value=local)
    with pytest.raises(expr.VerifyError, match="local.*used outside the scope that binds it"):
        verifier.verify_prim_function(_pf(binding, consumer))


@pytest.mark.parametrize("self_reference", (False, True))
def test_letstmt_value_refuses_unbound_var(self_reference):
    """A value cannot read its own binding or a later statement's binding."""
    scalar = types.TensorType.from_shape((), expr.DType.i32, storage="rmem")
    local = expr.Var(type=scalar, name="local")
    consumer = core.LetStmt(
        var=local if self_reference else expr.Var(type=scalar, name="early"),
        value=local,
    )
    binding = core.LetStmt(var=local, value=expr.Constant(type=scalar, value=0))
    with pytest.raises(expr.VerifyError, match="local.*used outside the scope that binds it"):
        verifier.verify_prim_function(_pf(consumer, binding))


def test_verify_rejects_non_ir_node():
    """Lowering recipes must be materialized before entering TIR expression fields."""
    tensor = types.TensorType.from_shape((4,), storage="rmem")
    source = expr.Var(type=tensor, name="source")
    recipe = transforms.View(base=source, starts=(), sizes=(4,), type=tensor)
    function = core.PrimFunction(
        name="recipe",
        params=(source,),
        body=core.Sequential(body=(
            core.LetStmt(
                var=expr.Var(type=tensor, name="leaked"),
                value=recipe,
            ),
        )),
    )
    with pytest.raises(expr.VerifyError, match="non-IR node View"):
        verifier.verify_prim_function(function)


def test_variant_refuses_parent_parameter():
    """Each specialization owns its parameters; the parent's Var is invisible."""
    program = cli.load_authored_ir(str(Path(__file__).parents[1] / "fixtures/tir/parent_parameter.py"))
    function = program.entry_function()
    variant = function.variants[0]
    assert variant.params[0] is not function.params[0]
    variant.body = core.Sequential(body=(core.LetStmt(
        var=expr.Var(type=function.params[0].type, name="escaped"),
        value=function.params[0],
    ),))
    with pytest.raises(expr.VerifyError, match="x.*used outside the scope that binds it"):
        verifier.verify_prim_function(function)
