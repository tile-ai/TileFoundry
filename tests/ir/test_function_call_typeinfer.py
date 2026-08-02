"""hir Function call typeinfer: the callee body is re-derived under the actual
argument types.

A parameter declared without sharding (``layout is None``) is a layout-
unconstrained logical tensor: an argument of any layout flows in and propagates
through the body, so the same callee specializes per call site. A parameter
that declares an explicit ``ShardLayout`` constrains its argument (mismatch
fails at the boundary). A model call graph exercises the plain case only, so what
is kept here is the layout-carrying propagation and every boundary rejection —
the ones whose message has to name the call site to be actionable.
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import infer_call
from tilefoundry.ir.core import BindingMetadata, Call, Tuple, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types import DType, TupleType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, Split
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_F = DType.f32
_M = make_mesh((4,))
_PLAIN = make_tensor_type((4, 8), _F)
_SPLIT0 = make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),))


def _add_callee(param_type):
    """A callee ``f(x) = x + x``; the body's output layout is whatever the
    Binary engine derives from the actual ``x`` type."""
    x = Var(type=param_type, name="x")
    body = Call(type=param_type, target=Binary(kind=BinaryKind.ADD), args=(x, x))
    return Function.build(
        name="f", params=(x,), body=body, return_type=make_tensor_type((4, 8), _F)
    )


def test_plain_formal_specializes_per_call_site():
    # A split argument flows into the layout-unconstrained parameter; the body
    # re-derives, so the result carries the split and the same callee object
    # returns different types for plain vs split actuals.
    f = _add_callee(_PLAIN)
    assert infer_call(f, _PLAIN) == _PLAIN
    assert infer_call(f, _SPLIT0) == _SPLIT0
    assert infer_call(f, _PLAIN).layout is None


def test_carrying_loop_propagates_split():
    """A callee whose body is a single-carry loop-phi ``GridRegionExpr``
    (``acc = x + x`` before the loop, ``acc = acc + x`` inside it): the loop-phi's
    own type must re-derive from the elaborated init value ([hir §1.2](docs/spec/hir.md#12-gridregionexpr)), not stay
    at the callee's parse-time unsharded type."""
    param_type = make_tensor_type((8,), _F)
    x = Var(type=param_type, name="x")
    init = Call(type=param_type, target=Binary(kind=BinaryKind.ADD), args=(x, x))
    phi = Var(type=param_type, name="acc")
    iv = Var(type=make_tensor_type((), DType.i64), name="i")
    body = Call(type=param_type, target=Binary(kind=BinaryKind.ADD), args=(phi, x))
    grid = GridRegionExpr(
        type=param_type, induction_var=iv, carried_args=(phi,),
        init_args=(init,), body=body, yield_values=(body,),
        extent=8, step=1,
    )
    f = Function.build(name="carry", params=(x,), body=grid, return_type=param_type)

    split = make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),))
    assert infer_call(f, split) == split


def test_explicit_sharded_formal_constrains_its_actual():
    # An explicit sharded parameter is a layout constraint: only the matching
    # actual flows in. A plain actual, or a differently split one, is a boundary
    # mismatch rather than something silently accepted and re-derived.
    f = _add_callee(_SPLIT0)
    assert infer_call(f, _SPLIT0) == _SPLIT0
    with pytest.raises(VerifyError, match="type mismatch"):
        infer_call(f, _PLAIN)
    with pytest.raises(VerifyError, match="type mismatch"):
        infer_call(f, make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(1),)))


def test_plain_formal_rejects_shape_or_dtype_mismatch():
    # Layout is unconstrained; shape and dtype never are.
    f = _add_callee(_PLAIN)
    with pytest.raises(VerifyError, match="shape/dtype mismatch"):
        infer_call(f, make_tensor_type((4, 16), _F))
    with pytest.raises(VerifyError, match="shape/dtype mismatch"):
        infer_call(f, make_tensor_type((4, 8), DType.bf16))


def test_function_call_preserves_partial_in_tuple_return():
    mesh_ab = make_mesh((2, 4), ("a", "b"))
    partial = make_shard_tensor_type(
        (4, 8), mesh=mesh_ab, attrs=(Broadcast(), Partial("max"))
    )
    param = Var(type=_PLAIN, name="x")
    return_type = TupleType(fields=(_PLAIN, _PLAIN))
    body = Tuple(type=return_type, elements=(param, param))
    callee = Function.build(
        name="partial_pair", params=(param,), body=body, return_type=return_type
    )
    arg = Var(type=partial, name="arg")
    call = Call(type=return_type, target=callee, args=(arg,))

    result = TypeInferVisitor(TypeInferContext()).visit(call)

    assert result == TupleType(fields=(partial, partial))
    assert result.fields[0].layout.mesh == mesh_ab
    assert result.fields[0].layout.attrs == (Broadcast(), Partial("max"))


def test_bind_error_reports_call_site_binding():
    # A bind-mismatch VerifyError must report the call-site binding metadata.
    f = _add_callee(_SPLIT0)
    arg = Var(type=_PLAIN, name="x_arg")
    call = Call(
        type=f.return_type,
        target=f,
        args=(arg,),
        metadata=(BindingMetadata("y"),),
    )
    with pytest.raises(VerifyError, match="at y"):
        TypeInferVisitor(TypeInferContext()).visit(call)
