"""Cover HIR call elaboration under concrete argument types.

Layout-free parameters accept and propagate caller layouts; explicit layouts
remain constraints. Boundary failures must identify the call site.

See [hir §1.1](docs/spec/hir.md#11-function).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tilefoundry import services
from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import core, expr, types, utils
from tilefoundry.ir.hir import math, tensor

_F = expr.DType.f32
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
_PLAIN = types.TensorType.from_shape((4, 8), _F)
_SPLIT0 = utils.make_shard_tensor_type((4, 8), mesh=_M, attrs=(types.Split(0),))


def call_in_module(callee, args, return_type, metadata=()):
    """A call naming *callee* by reference, and the context of a Module that declares it."""
    call = expr.Call(
        type=return_type, target=core.FunctionRef((), callee.name), args=args, metadata=metadata
    )
    module = core.Module(name="Calls", functions=(callee,))
    return call, typeinfer.TypeInferContext(module=module)


def infer_call(callee, *input_types):
    """Infer one call of *callee* over typed variables."""
    args = tuple(expr.Var(type=t, name=f"x{i}") for i, t in enumerate(input_types))
    call, context = call_in_module(callee, args, input_types[0])
    return typeinfer.TypeInferVisitor().visit(call, context)


def _add_callee(param_type):
    """A callee ``f(x) = x + x``.

    A callee ``f(x) = x + x``; the body's output layout is whatever the
    Binary engine derives from the actual ``x`` type.
    """
    x = expr.Var(
        type=param_type,
        annotation=replace(param_type, layout=None) if param_type is _PLAIN else param_type,
        name="x",
    )
    body = expr.Call(type=param_type, target=math.Binary(kind=expr.BinaryKind.ADD), args=(x, x))
    return core.Function.from_parts(
        name="f", params=(x,), body=body, return_type=types.TensorType.from_shape((4, 8), _F)
    )


def test_plain_formal_specializes_per_call():

    f = _add_callee(_PLAIN)
    assert infer_call(f, _PLAIN) == _PLAIN
    assert infer_call(f, _SPLIT0) == _SPLIT0
    assert infer_call(f, _PLAIN).layout == _PLAIN.layout


_PLAIN8 = types.TensorType.from_shape((8,), _F)
_SPLIT8 = utils.make_shard_tensor_type((8,), mesh=_M, attrs=(types.Split(0),))
_UMAT8 = types.TensorType.from_shape((8,), storage="umat")
_RMEM8 = types.TensorType.from_shape((8,), storage="rmem")


@pytest.mark.parametrize(
    ("formal", "actual", "yielded", "extent", "yields", "error"),
    [
        (_PLAIN8, _SPLIT8, None, 8, 1, None),
        (_UMAT8, _UMAT8, _RMEM8, 8, 1, None),
        (_UMAT8, _UMAT8, _RMEM8, 0, 1, None),
        (
            _RMEM8,
            _RMEM8,
            types.TensorType.from_shape((8,), storage="smem"),
            8,
            1,
            "LoopRegion yield 0 type mismatch for param 'acc'",
        ),
        (_PLAIN8, _PLAIN8, types.TensorType.from_shape((8,), expr.DType.f16), 8, 1, "yield 0 type mismatch"),
        (_PLAIN8, _PLAIN8, types.TensorType.from_shape((4,)), 8, 1, "yield 0 type mismatch"),
        (
            _SPLIT8,
            _SPLIT8,
            utils.make_shard_tensor_type((8,), mesh=_M, attrs=(types.Broadcast(),)),
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
    stamped = types.TensorType.from_shape((8,), expr.DType.i32)
    y_formal = formal if yielded is None else yielded
    x = expr.Var(
        type=formal,
        annotation=replace(formal, layout=None) if isinstance(formal.layout, types.Layout) else formal,
        name="x",
    )
    y = expr.Var(
        type=y_formal,
        annotation=replace(y_formal, layout=None)
        if isinstance(y_formal.layout, types.Layout)
        else y_formal,
        name="y",
    )
    init = expr.Call(type=stamped, target=math.Binary(kind=expr.BinaryKind.ADD), args=(x, x))
    acc = expr.Var(type=stamped, name="acc")
    captured_x = expr.Var(type=stamped, name="x")
    captured_y = expr.Var(type=stamped, name="y")
    unused = expr.Var(type=stamped, name="unused")
    if yielded is None:
        body = expr.Call(type=stamped, target=math.Binary(kind=expr.BinaryKind.ADD), args=(acc, captured_x))
        yield_values = (body,)
    else:
        body = captured_y
        yield_values = (captured_y,) * yields
    grid = core.LoopRegion(
        type=stamped,
        induction_var=expr.Var(type=types.TensorType.from_shape((), expr.DType.i64), name="i"),
        params=(acc, captured_x, captured_y, unused),
        args=(init, x, y, x),
        body=body,
        yield_values=yield_values,
        extent=extent,
        step=1,
    )
    f = core.Function.from_parts(name="carry", params=(x, y), body=grid, return_type=formal)
    region = (init, acc, captured_x, captured_y, unused, grid)
    actual_y = actual if yielded is None else yielded

    if error is not None:
        with pytest.raises(expr.VerifyError, match=error):
            infer_call(f, actual, actual_y)
        assert all(node.type is stamped for node in region)
        return
    assert infer_call(f, actual, actual_y) == actual
    assert all(node.type is stamped for node in region)
    typeinfer.TypeInferVisitor().visit(f, typeinfer.TypeInferContext())
    assert [node.type for node in region] == [formal, formal, formal, y_formal, formal, formal]


def test_sharded_formal_constrains_actual():

    f = _add_callee(_SPLIT0)
    assert infer_call(f, _SPLIT0) == _SPLIT0
    with pytest.raises(expr.VerifyError, match="type mismatch"):
        infer_call(f, _PLAIN)
    with pytest.raises(expr.VerifyError, match="type mismatch"):
        infer_call(f, utils.make_shard_tensor_type((4, 8), mesh=_M, attrs=(types.Split(1),)))


def test_broadcast_formal_accepts_reshaped_slice():
    packed = utils.make_shard_tensor_type((4, 8, 16), _F, mesh=_M, attrs=(types.Broadcast(),))
    sliced_type = utils.make_shard_tensor_type((1, 8, 16), _F, mesh=_M, attrs=(types.Broadcast(),))
    formal = utils.make_shard_tensor_type((8, 16), _F, mesh=_M, attrs=(types.Broadcast(),))
    packed_var = expr.Var(type=packed, name="packed")
    layer = expr.Var(type=types.TensorType.from_shape((), expr.DType.i64, storage="umat"), name="layer")
    zero = expr.Constant(type=types.TensorType.from_shape((), expr.DType.i64), value=0)
    starts = expr.Tuple(
        type=types.TupleType(fields=(layer.type, zero.type, zero.type)),
        elements=(layer, zero, zero),
    )
    sliced = expr.Call(
        type=sliced_type,
        target=tensor.Slice(sizes=(1, 8, 16), strides=(1, 1, 1)),
        args=(packed_var, starts),
    )
    reshaped = expr.Call(type=formal, target=tensor.Reshape(new_shape=(8, 16)), args=(sliced,))
    w = expr.Var(type=formal, name="w")
    callee = core.Function.from_parts(name="consume", params=(w,), body=w, return_type=formal)
    call, context = call_in_module(callee, (reshaped,), formal)

    assert typeinfer.TypeInferVisitor().visit(call, context) == formal


def test_symbolic_signature_matches_argument():
    seq = expr.DimVar("call_seq", 1, 4096)
    authored_dim = services.simplify_dim("mul", (seq, 2))
    authored_type = types.TensorType.from_shape((authored_dim, 8), _F)
    x = expr.Var(type=authored_type, name="x")
    stage = core.Function.from_parts(
        name="stage",
        params=(x,),
        body=x,
        return_type=authored_type,
    )
    y = expr.Var(type=authored_type, name="y")
    call, context = call_in_module(stage, (y,), authored_type)
    expected_dim = services.simplify_dim("mul", (2, seq))

    assert stage.params[0].type.shape == (expected_dim, 8)
    assert typeinfer.TypeInferVisitor().visit(call, context) == stage.return_type


def test_plain_formal_rejects_mismatch():

    f = _add_callee(_PLAIN)
    with pytest.raises(expr.VerifyError, match="type mismatch"):
        infer_call(f, types.TensorType.from_shape((4, 16), _F))
    with pytest.raises(expr.VerifyError, match="type mismatch"):
        infer_call(f, types.TensorType.from_shape((4, 8), expr.DType.bf16))


def test_call_preserves_partial_tuple_return():
    mesh_ab = types.Mesh((types.Topology("gpu", 8),), types.Layout((2, 4), (4, 1)), ("a", "b"))
    partial = utils.make_shard_tensor_type((4, 8), mesh=mesh_ab, attrs=(types.Broadcast(), types.Partial("max")))
    param = expr.Var(type=_PLAIN, annotation=replace(_PLAIN, layout=None), name="x")
    return_type = types.TupleType(fields=(_PLAIN, _PLAIN))
    body = expr.Tuple(type=return_type, elements=(param, param))
    callee = core.Function.from_parts(
        name="partial_pair", params=(param,), body=body, return_type=return_type
    )
    arg = expr.Var(type=partial, name="arg")
    call, context = call_in_module(callee, (arg,), return_type)

    result = typeinfer.TypeInferVisitor().visit(call, context)

    assert result == types.TupleType(fields=(partial, partial))
    assert result.fields[0].layout.mesh == mesh_ab
    assert result.fields[0].layout.attrs == (types.Broadcast(), types.Partial("max"))


def test_bind_error_names_call_binding():

    f = _add_callee(_SPLIT0)
    arg = expr.Var(type=_PLAIN, name="x_arg")
    call, context = call_in_module(f, (arg,), f.return_type, (expr.BindingMetadata("y"),))
    with pytest.raises(expr.VerifyError, match="at y"):
        typeinfer.TypeInferVisitor().visit(call, context)
