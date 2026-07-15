"""hir Function call typeinfer: the callee body is re-derived under the actual
argument types.

A parameter declared without sharding (``layout is None``) is a layout-
unconstrained logical tensor: an argument of any layout flows in and propagates
through the body, so the same callee specializes per call site. A parameter
that declares an explicit ``ShardLayout`` constrains its argument (mismatch
fails at the boundary).
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import infer_call, mesh, ten
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types import DType, make_shard_tensor_type
from tilefoundry.ir.types.shard.shard_layout import Split
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_F = DType.f32
_M = mesh((4,))


def _add_callee(param_type):
    """A callee ``f(x) = x + x``; the body's output layout is whatever the
    Binary engine derives from the actual ``x`` type."""
    x = Var(type=param_type, name="x")
    body = Call(type=param_type, target=Binary(kind=BinaryKind.ADD), args=(x, x))
    return Function.build(
        name="f", params=(x,), body=body, return_type=ten((4, 8), _F)
    )


def test_plain_formal_accepts_plain_actual():
    f = _add_callee(ten((4, 8), _F))
    out = infer_call(f, ten((4, 8), _F))
    assert out == ten((4, 8), _F)


def test_plain_formal_accepts_split_actual_and_propagates():
    # A split argument flows into the layout-unconstrained parameter; the body
    # re-derives, so the result carries the split (specialization per caller).
    f = _add_callee(ten((4, 8), _F))
    out = infer_call(f, make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)))
    assert out == make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),))


def test_same_callee_specializes_per_call_site():
    # The same callee object returns different types for plain vs split actuals.
    f = _add_callee(ten((4, 8), _F))
    plain_out = infer_call(f, ten((4, 8), _F))
    split_out = infer_call(f, make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)))
    assert plain_out.layout is None
    assert split_out.layout == make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)).layout


def _carry_callee(param_type):
    """A callee whose body is a single-carry loop-phi ``GridRegionExpr``:
    ``acc = x + x`` before the loop, ``acc = acc + x`` inside it."""
    x = Var(type=param_type, name="x")
    init = Call(type=param_type, target=Binary(kind=BinaryKind.ADD), args=(x, x))
    phi = Var(type=param_type, name="acc")
    iv = Var(type=ten((), DType.i64), name="i")
    body = Call(type=param_type, target=Binary(kind=BinaryKind.ADD), args=(phi, x))
    grid = GridRegionExpr(
        type=param_type, induction_var=iv, carried_args=(phi,),
        init_args=(init,), body=body, yield_values=(body,),
        extent=8, step=1,
    )
    return Function.build(name="carry", params=(x,), body=grid, return_type=param_type)


def test_carrying_loop_propagates_split():
    # The loop-phi's own type must re-derive from the elaborated init value
    # (hir.md §1.2), not stay at the callee's parse-time unsharded type.
    f = _carry_callee(ten((8,), _F))
    out = infer_call(f, make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),)))
    assert out == make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),))


def test_explicit_sharded_formal_accepts_matching_actual():
    f = _add_callee(make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)))
    out = infer_call(f, make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)))
    assert out == make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),))


def test_explicit_sharded_formal_rejects_plain_actual():
    # An explicit sharded parameter is a layout constraint: a plain actual is a
    # boundary mismatch, not silently accepted.
    f = _add_callee(make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)))
    with pytest.raises(VerifyError, match="type mismatch"):
        infer_call(f, ten((4, 8), _F))


def test_explicit_sharded_formal_rejects_wrong_split_actual():
    f = _add_callee(make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)))
    with pytest.raises(VerifyError, match="type mismatch"):
        infer_call(f, make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(1),)))


def test_plain_formal_rejects_shape_mismatch():
    f = _add_callee(ten((4, 8), _F))
    with pytest.raises(VerifyError, match="shape/dtype mismatch"):
        infer_call(f, ten((4, 16), _F))


def test_plain_formal_rejects_dtype_mismatch():
    f = _add_callee(ten((4, 8), _F))
    with pytest.raises(VerifyError, match="shape/dtype mismatch"):
        infer_call(f, ten((4, 8), DType.bf16))


def test_bind_error_reports_call_site_loc():
    # A bind-mismatch VerifyError must report the *call's* loc, not the
    # callee's own (always-None) .loc.
    f = _add_callee(make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)))
    arg = Var(type=ten((4, 8), _F), name="x_arg")
    call = Call(type=f.return_type, target=f, args=(arg,), loc="y")
    with pytest.raises(VerifyError, match="at y"):
        TypeInferVisitor(TypeInferContext()).visit(call)
