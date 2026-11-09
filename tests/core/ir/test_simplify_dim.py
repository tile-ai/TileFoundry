"""Construction and IR-boundary normalization for dimension arithmetic."""

from __future__ import annotations

import copy

import pytest

from tilefoundry import services
from tilefoundry.interfaces import access, typeinfer
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import math, tensor


def _i64(v: int) -> expr.Constant:
    return expr.Constant(type=types.TensorType.from_scalar(expr.DType.i64), value=v)


def _sym(name: str) -> expr.Call:
    """A symbolic dim Expr — wrap a DimVar in a Call so its presence breaks all-Constant folding.

    A symbolic dim Expr — wrap a DimVar in a Call so its
    presence breaks all-Constant folding.
    """
    return expr.Call(
        type=types.TensorType.from_scalar(expr.DType.i64),
        target=expr.DimVar(name=name, lo=1, hi=1023),
        args=(),
    )


def test_simplify_dim_only_constructs_constant_arithmetic() -> None:
    table = [
        (expr.DimAdd, "add", 3, 4),
        (expr.DimSub, "sub", 10, 4),
        (expr.DimMul, "mul", 3, 4),
        (expr.DimFloorDiv, "floordiv", 17, 4),
        (expr.DimMod, "mod", 17, 5),
        (expr.DimMin, "min", 7, 3),
        (expr.DimMax, "max", 7, 3),
        (expr.DimFloorDiv, "floordiv", -7, 2),
    ]
    for op_cls, kind, a, b in table:
        result = services.simplify_dim(kind, (_i64(a), _i64(b)))
        assert isinstance(result, expr.Call)
        assert isinstance(result.target, op_cls)
        assert result.args == (_i64(a), _i64(b))
        assert result.type == types.TensorType.from_umat_scalar()


def test_simplify_dim_constructs_symbolic_and_invalid_arithmetic() -> None:
    sym = _sym("M")
    for kind, op_cls in (("add", expr.DimAdd), ("floordiv", expr.DimFloorDiv)):
        for args in ((sym, _i64(0)), (_i64(0), sym)):
            result = services.simplify_dim(kind, args)
            assert isinstance(result, expr.Call)
            assert isinstance(result.target, op_cls)
            assert result.args == args

    div_zero = services.simplify_dim("floordiv", (_i64(10), _i64(0)))
    assert isinstance(div_zero, expr.Call)
    assert isinstance(div_zero.target, expr.DimFloorDiv)
    assert div_zero.args == (_i64(10), _i64(0))

    b_true = expr.Constant(type=types.TensorType.from_scalar(expr.DType.i64), value=True)
    bool_arg = services.simplify_dim("add", (b_true, _i64(1)))
    assert isinstance(bool_arg, expr.Call)
    assert isinstance(bool_arg.target, expr.DimAdd)


def test_dim_call_arithmetic_is_pure_construction_in_both_directions() -> None:
    seq = expr.DimVar("S_chain", 1, 1023)
    base = seq - 1

    expressions = (
        base + 8,
        8 + base,
        base - 8,
        8 - base,
        base * 8,
        8 * base,
        base // 8,
        8 // base,
        base % 8,
        8 % base,
    )

    assert all(isinstance(node, expr.Call) for node in expressions)
    assert [type(node.target) for node in expressions] == [
        expr.DimAdd,
        expr.DimAdd,
        expr.DimSub,
        expr.DimSub,
        expr.DimMul,
        expr.DimMul,
        expr.DimFloorDiv,
        expr.DimFloorDiv,
        expr.DimMod,
        expr.DimMod,
    ]


def test_non_dim_calls_do_not_gain_dimension_arithmetic() -> None:
    tensor_value = types.TensorType.from_umat_tensor((8,), expr.DType.f32)
    value = expr.Var(type=tensor_value, name="value")
    ordinary = expr.Call(
        type=tensor_value,
        target=math.Unary(kind=expr.UnaryKind.NEG),
        args=(value,),
    )
    operations = (
        lambda: ordinary + 1,
        lambda: 1 + ordinary,
        lambda: ordinary - 1,
        lambda: 1 - ordinary,
        lambda: ordinary * 1,
        lambda: 1 * ordinary,
        lambda: ordinary // 1,
        lambda: 1 // ordinary,
        lambda: ordinary % 1,
        lambda: 1 % ordinary,
    )

    for operation in operations:
        with pytest.raises(TypeError):
            operation()


def test_typeinfer_canonicalizes_equivalent_symbolic_shapes() -> None:
    seq = expr.DimVar("S_canonical", 1, 8192)
    verbose = services.simplify_dim(
        "floordiv",
        (
            services.simplify_dim(
                "add",
                (services.simplify_dim("sub", (services.simplify_dim("add", (seq, 0)), 0)), 0),
            ),
            1,
        ),
    )
    def layout(dim):
        return types.ComposedLayout(
            inner=types.Layout(shape=(dim,), strides=(dim,)),
            offset=dim,
            outer=types.ShardLayout(
                layout=types.Layout(shape=(dim,), strides=(dim,)),
                attrs=(types.Broadcast(),),
                mesh=types.Mesh(
                    topologies=(types.Topology("cta", dim),),
                    layout=types.Layout(shape=(dim,), strides=(1,)),
                ),
            ),
        )

    verbose_type = types.TensorType(
        shape=(verbose, 128), dtype=expr.DType.f32, layout=layout(verbose), storage="gmem"
    )
    direct_type = types.TensorType(
        shape=(seq, 128), dtype=expr.DType.f32, layout=layout(seq), storage="gmem"
    )

    inferred = typeinfer.TypeInferVisitor().visit(
        expr.Var(type=verbose_type, name="verbose"), typeinfer.TypeInferContext()
    )

    assert inferred == direct_type
    assert inferred.shape[0] is seq


def test_static_typeinfer_does_not_enter_dim_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static = types.TensorType(
        shape=(8, 128), dtype=expr.DType.f32, layout=types.Layout((8, 128), (128, 1)), storage="gmem"
    )

    def fail_if_called(_):
        raise AssertionError("static types must not enter isl canonicalization")

    monkeypatch.setattr("tilefoundry.ir.dim.substitute.normalize", fail_if_called)

    assert (
        typeinfer.TypeInferVisitor().visit(expr.Var(type=static, name="static"), typeinfer.TypeInferContext())
        is static
    )


def test_a_fully_static_dim_has_one_canonical_int_representation() -> None:
    """Test a fully static dim has one canonical int representation.

    ``TensorType`` folds integer-valued ``Constant`` shape dims to plain ``int``,
    leaving ``DimVar`` / dynamic dims alone, and a typeinfer result built through
    ``simplify_dim`` arrives the same way.

    Regression: a ``Slice`` output used to carry flat ``Constant`` dims, so
    ``Binary`` broadcast failed comparing ``(1,4,32,Constant(128))`` against a
    param's ``(1,4,32,128)`` — the rotate-half bug.
    """
    ty = types.TensorType(
        shape=(_i64(1), _i64(4), _i64(32), _i64(128)),
        dtype=expr.DType.bf16,
        layout=None,
        storage="gmem",
    )
    assert ty.shape == (1, 4, 32, 128)
    assert all(isinstance(d, int) and not isinstance(d, bool) for d in ty.shape)

    s = expr.DimVar(name="S_ti", lo=1, hi=7)
    mixed = types.TensorType(shape=(s, _i64(128)), dtype=expr.DType.f32, layout=None, storage="gmem")
    assert mixed.shape[0] is s
    assert mixed.shape[1] == 128 and isinstance(mixed.shape[1], int)

    param = types.TensorType(
        shape=(1, 4, 32, 128),
        dtype=expr.DType.bf16,
        layout=types.Layout((1, 4, 32, 128)),
        storage="gmem",
    )
    x = expr.Constant(type=param, value=None)
    starts = expr.Tuple(
        type=types.TupleType(fields=(types.TensorType.from_scalar(expr.DType.i64),) * 4),
        elements=(_i64(0), _i64(0), _i64(0), _i64(0)),
    )
    sliced = typeinfer.TypeInferVisitor().visit(
        expr.Call(
            type=types.TensorType.from_scalar(expr.DType.bf16),
            target=tensor.Slice(sizes=(1, 4, 32, 128), strides=(1, 1, 1, 1)),
            args=(x, starts),
        ),
        typeinfer.TypeInferContext(),
    )
    for dim in sliced.shape:
        assert isinstance(dim, int) and not isinstance(dim, bool), (
            f"expected canonical int static dim, got {type(dim).__name__}: {dim}"
        )
    assert sliced.shape == param.shape == (1, 4, 32, 128)
    assert access.broadcast_shapes(sliced.shape, param.shape) == (1, 4, 32, 128)


def test_constant_dim_arithmetic_normalizes_when_stored_as_an_op_attribute() -> None:
    constructed = services.simplify_dim("mul", (4, 2))
    assert isinstance(constructed, expr.Call)

    reshape = tensor.Reshape(new_shape=(constructed, 16))

    assert reshape.new_shape == (8, 16)


def test_static_op_attributes_do_not_enter_dim_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(_):
        raise AssertionError("static attributes must not enter isl normalization")

    monkeypatch.setattr("tilefoundry.ir.dim.isl_interop.normalize", fail_if_called)

    assert tensor.Reshape(new_shape=(8, 16)).new_shape == (8, 16)


def test_unary_propagates_dim_var_in_shape() -> None:
    """Test unary propagates dim var in shape.

    Unary(NEG) on a tensor whose first axis is a ``DimVar`` keeps
    that ``DimVar`` (with the same bounds) on the result type — the
    dynamic dim is not collapsed to a concrete int.
    """
    s = expr.DimVar(name="S_ti", lo=1, hi=7)
    in_ty = types.TensorType(shape=(s, 8), dtype=expr.DType.f32, layout=None, storage="gmem")
    x = expr.Var(type=in_ty, name="x")
    call = expr.Call(type=in_ty, target=math.Unary(kind=expr.UnaryKind.NEG), args=(x,))
    out_ty = typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    assert out_ty.shape == (s, 8)

    assert out_ty.shape[0] is s
    assert (out_ty.shape[0].lo, out_ty.shape[0].hi) == (1, 7)


def test_a_copied_dim_var_is_the_canonical_one() -> None:
    """A deep copy of a shape keeps its ``DimVar`` identity.

    Test-side deep copies must preserve the canonical ``DimVar`` so a copied
    signature still binds against the dimension its caller wrote.
    """
    s = expr.DimVar(name="S_copy", lo=1, hi=7)
    in_ty = types.TensorType(shape=(s, 8), dtype=expr.DType.f32, layout=None, storage="gmem")

    assert copy.deepcopy(s) is s
    assert copy.deepcopy(in_ty).shape == in_ty.shape
    assert copy.deepcopy([in_ty])[0].shape[0] is s
