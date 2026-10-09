"""Cover HIR call elaboration under concrete argument types.

Layout-free parameters accept and propagate caller layouts; explicit layouts
remain constraints. Boundary failures must identify the call site.

See [hir §1.1](docs/spec/hir.md#11-function).
"""

from __future__ import annotations

import pytest

from tests.ops.ir.typeinfer_utils import infer_call
from tilefoundry.ir.core import BindingMetadata, Call, Constant, Tuple, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    Topology,
    TupleType,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.dim import DimMul, DimVar, simplify_dim
from tilefoundry.ir.types.shard_layout import Broadcast, Partial, Split
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor

_F = DType.f32
_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
_PLAIN = make_tensor_type((4, 8), _F)
_SPLIT0 = make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),))


def _add_callee(param_type):
    """A callee ``f(x) = x + x``.

    A callee ``f(x) = x + x``; the body's output layout is whatever the
    Binary engine derives from the actual ``x`` type.
    """
    x = Var(type=param_type, name="x")
    body = Call(type=param_type, target=Binary(kind=BinaryKind.ADD), args=(x, x))
    return Function.build(
        name="f", params=(x,), body=body, return_type=make_tensor_type((4, 8), _F)
    )


def test_plain_formal_specializes_per_call_site():

    f = _add_callee(_PLAIN)
    assert infer_call(f, _PLAIN) == _PLAIN
    assert infer_call(f, _SPLIT0) == _SPLIT0
    assert infer_call(f, _PLAIN).layout is None


_PLAIN8 = make_tensor_type((8,), _F)
_SPLIT8 = make_shard_tensor_type((8,), mesh=_M, attrs=(Split(0),))
_UMAT8 = make_tensor_type((8,), storage="umat")
_RMEM8 = make_tensor_type((8,), storage="rmem")


@pytest.mark.parametrize(
    ("formal", "actual", "yielded", "extent", "yields", "error"),
    [
        (_PLAIN8, _SPLIT8, None, 8, 1, None),
        (_UMAT8, _UMAT8, _RMEM8, 8, 1, None),
        (_UMAT8, _UMAT8, _RMEM8, 0, 1, None),
        (
            _RMEM8,
            _RMEM8,
            make_tensor_type((8,), storage="smem"),
            8,
            1,
            "LoopRegion yield 0 type mismatch for param 'acc'",
        ),
        (_PLAIN8, _PLAIN8, make_tensor_type((8,), DType.f16), 8, 1, "yield 0 type mismatch"),
        (_PLAIN8, _PLAIN8, make_tensor_type((4,)), 8, 1, "yield 0 type mismatch"),
        (
            _SPLIT8,
            _SPLIT8,
            make_shard_tensor_type((8,), mesh=_M, attrs=(Broadcast(),)),
            8,
            1,
            "yield 0 type mismatch",
        ),
        (_PLAIN8, _PLAIN8, _PLAIN8, 8, 5, "yields 5 values but has 4 params"),
    ],
)
def test_carrying_loop_propagates_split(formal, actual, yielded, extent, yields, error):
    """Test carrying loop propagates split.

    A loop-phi ``acc`` starts at ``x + x`` and adds the captured ``x``, or
    carries the captured ``y`` when ``yielded`` is given. Region parameters take
    the current entry type, not their parse-time stamp, so a split actual for
    the layout-free formal ``x`` reaches the phi and the call's result; no
    stamp, the entry's included, is read as an input. The entry type is the
    result even with no iteration; a yield must fit the entry and does not
    narrow it ([hir §1.2](docs/spec/hir.md#12-loopregion)).
    """
    stamped = make_tensor_type((8,), DType.i32)
    y_formal = formal if yielded is None else yielded
    x = Var(type=formal, name="x")
    y = Var(type=y_formal, name="y")
    init = Call(type=stamped, target=Binary(kind=BinaryKind.ADD), args=(x, x))
    acc = Var(type=stamped, name="acc")
    captured_x = Var(type=stamped, name="x")
    captured_y = Var(type=stamped, name="y")
    unused = Var(type=stamped, name="unused")
    if yielded is None:
        body = Call(type=stamped, target=Binary(kind=BinaryKind.ADD), args=(acc, captured_x))
        yield_values = (body,)
    else:
        body = captured_y
        yield_values = (captured_y,) * yields
    grid = LoopRegion(
        type=stamped,
        induction_var=Var(type=make_tensor_type((), DType.i64), name="i"),
        params=(acc, captured_x, captured_y, unused),
        args=(init, x, y, x),
        body=body,
        yield_values=yield_values,
        extent=extent,
        step=1,
    )
    f = Function.build(name="carry", params=(x, y), body=grid, return_type=formal)
    region = (init, acc, captured_x, captured_y, unused, grid)
    actual_y = actual if yielded is None else yielded

    if error is not None:
        with pytest.raises(VerifyError, match=error):
            infer_call(f, actual, actual_y)
        assert all(expr.type is stamped for expr in region)
        return
    assert infer_call(f, actual, actual_y) == actual
    assert all(expr.type is stamped for expr in region)
    TypeInferVisitor().visit(f, TypeInferContext())
    assert [expr.type for expr in region] == [formal, formal, formal, y_formal, formal, formal]


def test_explicit_sharded_formal_constrains_its_actual():

    f = _add_callee(_SPLIT0)
    assert infer_call(f, _SPLIT0) == _SPLIT0
    with pytest.raises(VerifyError, match="type mismatch"):
        infer_call(f, _PLAIN)
    with pytest.raises(VerifyError, match="type mismatch"):
        infer_call(f, make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(1),)))


def test_broadcast_formal_accepts_reshaped_runtime_slice():
    packed = make_shard_tensor_type((4, 8, 16), _F, mesh=_M, attrs=(Broadcast(),))
    sliced_type = make_shard_tensor_type((1, 8, 16), _F, mesh=_M, attrs=(Broadcast(),))
    formal = make_shard_tensor_type((8, 16), _F, mesh=_M, attrs=(Broadcast(),))
    packed_var = Var(type=packed, name="packed")
    layer = Var(type=make_tensor_type((), DType.i64, storage="umat"), name="layer")
    zero = Constant(type=make_tensor_type((), DType.i64), value=0)
    starts = Tuple(
        type=TupleType(fields=(layer.type, zero.type, zero.type)),
        elements=(layer, zero, zero),
    )
    sliced = Call(
        type=sliced_type,
        target=Slice(sizes=(1, 8, 16), strides=(1, 1, 1)),
        args=(packed_var, starts),
    )
    reshaped = Call(type=formal, target=Reshape(new_shape=(8, 16)), args=(sliced,))
    w = Var(type=formal, name="w")
    callee = Function.build(name="consume", params=(w,), body=w, return_type=formal)
    call = Call(type=formal, target=callee, args=(reshaped,))

    assert TypeInferVisitor().visit(call, TypeInferContext()) == formal


def test_symbolic_arithmetic_signature_matches_inferred_argument():
    seq = DimVar("call_seq", 1, 4096)
    authored_dim = simplify_dim(DimMul, (seq, 2))
    authored_type = make_tensor_type((authored_dim, 8), _F)
    x = Var(type=authored_type, name="x")
    stage = Function.build(
        name="stage",
        params=(x,),
        body=x,
        return_type=authored_type,
    )
    y = Var(type=authored_type, name="y")
    call = Call(type=authored_type, target=stage, args=(y,))
    expected_dim = simplify_dim(DimMul, (2, seq))

    assert stage.params[0].type.shape == (expected_dim, 8)
    assert TypeInferVisitor().visit(call, TypeInferContext()) == stage.return_type


def test_plain_formal_rejects_shape_or_dtype_mismatch():

    f = _add_callee(_PLAIN)
    with pytest.raises(VerifyError, match="type mismatch"):
        infer_call(f, make_tensor_type((4, 16), _F))
    with pytest.raises(VerifyError, match="type mismatch"):
        infer_call(f, make_tensor_type((4, 8), DType.bf16))


def test_function_call_preserves_partial_in_tuple_return():
    mesh_ab = Mesh((Topology("gpu", 8),), Layout((2, 4), (4, 1)), ("a", "b"))
    partial = make_shard_tensor_type((4, 8), mesh=mesh_ab, attrs=(Broadcast(), Partial("max")))
    param = Var(type=_PLAIN, name="x")
    return_type = TupleType(fields=(_PLAIN, _PLAIN))
    body = Tuple(type=return_type, elements=(param, param))
    callee = Function.build(
        name="partial_pair", params=(param,), body=body, return_type=return_type
    )
    arg = Var(type=partial, name="arg")
    call = Call(type=return_type, target=callee, args=(arg,))

    result = TypeInferVisitor().visit(call, TypeInferContext())

    assert result == TupleType(fields=(partial, partial))
    assert result.fields[0].layout.mesh == mesh_ab
    assert result.fields[0].layout.attrs == (Broadcast(), Partial("max"))


def test_bind_error_reports_call_site_binding():

    f = _add_callee(_SPLIT0)
    arg = Var(type=_PLAIN, name="x_arg")
    call = Call(
        type=f.return_type,
        target=f,
        args=(arg,),
        metadata=(BindingMetadata("y"),),
    )
    with pytest.raises(VerifyError, match="at y"):
        TypeInferVisitor().visit(call, TypeInferContext())
