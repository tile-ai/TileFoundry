"""Construction and IR-boundary normalization for dimension arithmetic."""

from __future__ import annotations

import copy

import pytest

import tilefoundry.ir.types.dim_isl as dim_isl
import tilefoundry.ir.types.substitute as dim_substitute
from tilefoundry.ir.core import Tuple, TypeInferContext
from tilefoundry.ir.core.expr import Call, Constant, Var
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir._helpers import broadcast_shapes
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
    simplify_dim,
)
from tilefoundry.ir.types.shard import ComposedLayout, Layout, Mesh, ShardLayout, Topology
from tilefoundry.ir.types.shard.shard_layout import Broadcast
from tilefoundry.visitor_registry.visitors import TypeInferVisitor


def _i64(v: int) -> Constant:
    return Constant(type=TensorType.scalar(DType.i64), value=v)


def _sym(name: str) -> Call:
    """A symbolic dim Expr — wrap a DimVar in a Call so its presence breaks all-Constant folding.

    A symbolic dim Expr — wrap a DimVar in a Call so its
    presence breaks all-Constant folding.
    """
    return Call(
        type=TensorType.scalar(DType.i64),
        target=DimVar(name=name, lo=1, hi=1024),
        args=(),
    )


def test_simplify_dim_only_constructs_constant_arithmetic() -> None:
    table = [
        (DimAdd, 3, 4),
        (DimSub, 10, 4),
        (DimMul, 3, 4),
        (DimFloorDiv, 17, 4),
        (DimMod, 17, 5),
        (DimMin, 7, 3),
        (DimMax, 7, 3),
        (DimFloorDiv, -7, 2),
    ]
    for op_cls, a, b in table:
        result = simplify_dim(op_cls, (_i64(a), _i64(b)))
        assert isinstance(result, Call)
        assert isinstance(result.target, op_cls)
        assert result.args == (_i64(a), _i64(b))
        assert result.type == TensorType.umat_scalar()


def test_simplify_dim_constructs_symbolic_and_invalid_arithmetic() -> None:
    sym = _sym("M")
    for op_cls in (DimAdd, DimFloorDiv):
        for args in ((sym, _i64(0)), (_i64(0), sym)):
            result = simplify_dim(op_cls, args)
            assert isinstance(result, Call)
            assert isinstance(result.target, op_cls)
            assert result.args == args

    div_zero = simplify_dim(DimFloorDiv, (_i64(10), _i64(0)))
    assert isinstance(div_zero, Call)
    assert isinstance(div_zero.target, DimFloorDiv)
    assert div_zero.args == (_i64(10), _i64(0))

    b_true = Constant(type=TensorType.scalar(DType.i64), value=True)
    bool_arg = simplify_dim(DimAdd, (b_true, _i64(1)))
    assert isinstance(bool_arg, Call)
    assert isinstance(bool_arg.target, DimAdd)


def test_dim_call_arithmetic_is_pure_construction_in_both_directions() -> None:
    seq = DimVar("S_chain", 1, 1024)
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

    assert all(isinstance(expr, Call) for expr in expressions)
    assert [type(expr.target) for expr in expressions] == [
        DimAdd,
        DimAdd,
        DimSub,
        DimSub,
        DimMul,
        DimMul,
        DimFloorDiv,
        DimFloorDiv,
        DimMod,
        DimMod,
    ]


def test_non_dim_calls_do_not_gain_dimension_arithmetic() -> None:
    tensor = TensorType.umat_tensor((8,), DType.f32)
    value = Var(type=tensor, name="value")
    ordinary = Call(
        type=tensor,
        target=Unary(kind=UnaryKind.NEG),
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
    seq = DimVar("S_canonical", 1, 8193)
    verbose = simplify_dim(
        DimFloorDiv,
        (
            simplify_dim(
                DimAdd,
                (simplify_dim(DimSub, (simplify_dim(DimAdd, (seq, 0)), 0)), 0),
            ),
            1,
        ),
    )
    def layout(dim):
        return ComposedLayout(
            inner=Layout(shape=(dim,), strides=(dim,)),
            offset=dim,
            outer=ShardLayout(
                layout=Layout(shape=(dim,), strides=(dim,)),
                attrs=(Broadcast(),),
                mesh=Mesh(
                    topologies=(Topology("cta", dim),),
                    layout=Layout(shape=(dim,), strides=(1,)),
                ),
            ),
        )

    verbose_type = TensorType(
        shape=(verbose, 128), dtype=DType.f32, layout=layout(verbose), storage="gmem"
    )
    direct_type = TensorType(
        shape=(seq, 128), dtype=DType.f32, layout=layout(seq), storage="gmem"
    )

    inferred = TypeInferVisitor().visit(
        Var(type=verbose_type, name="verbose"), TypeInferContext()
    )

    assert inferred == direct_type
    assert inferred.shape[0] is seq


def test_static_typeinfer_does_not_enter_dim_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static = TensorType(shape=(8, 128), dtype=DType.f32, layout=None, storage="gmem")

    def fail_if_called(_):
        raise AssertionError("static types must not enter isl canonicalization")

    monkeypatch.setattr(dim_substitute, "normalize_dim", fail_if_called)

    assert (
        TypeInferVisitor().visit(Var(type=static, name="static"), TypeInferContext())
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
    ty = TensorType(
        shape=(_i64(1), _i64(4), _i64(32), _i64(128)),
        dtype=DType.bf16,
        layout=None,
        storage="gmem",
    )
    assert ty.shape == (1, 4, 32, 128)
    assert all(isinstance(d, int) and not isinstance(d, bool) for d in ty.shape)

    s = DimVar(name="S_ti", lo=1, hi=8)
    mixed = TensorType(shape=(s, _i64(128)), dtype=DType.f32, layout=None, storage="gmem")
    assert mixed.shape[0] is s
    assert mixed.shape[1] == 128 and isinstance(mixed.shape[1], int)

    param = TensorType(shape=(1, 4, 32, 128), dtype=DType.bf16, layout=None, storage="gmem")
    x = Constant(type=param, value=None)
    starts = Tuple(
        type=TupleType(fields=(TensorType.scalar(DType.i64),) * 4),
        elements=(_i64(0), _i64(0), _i64(0), _i64(0)),
    )
    sliced = TypeInferVisitor().visit(
        Call(
            type=TensorType.scalar(DType.bf16),
            target=Slice(sizes=(1, 4, 32, 128), strides=(1, 1, 1, 1)),
            args=(x, starts),
        ),
        TypeInferContext(),
    )
    for dim in sliced.shape:
        assert isinstance(dim, int) and not isinstance(dim, bool), (
            f"expected canonical int static dim, got {type(dim).__name__}: {dim}"
        )
    assert sliced.shape == param.shape == (1, 4, 32, 128)
    assert broadcast_shapes(sliced.shape, param.shape) == (1, 4, 32, 128)


def test_constant_dim_arithmetic_normalizes_when_stored_as_an_op_attribute() -> None:
    constructed = simplify_dim(DimMul, (4, 2))
    assert isinstance(constructed, Call)

    reshape = Reshape(new_shape=(constructed, 16))

    assert reshape.new_shape == (8, 16)


def test_static_op_attributes_do_not_enter_dim_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(_):
        raise AssertionError("static attributes must not enter isl normalization")

    monkeypatch.setattr(dim_isl, "normalize_dim", fail_if_called)

    assert Reshape(new_shape=(8, 16)).new_shape == (8, 16)


def test_unary_propagates_dim_var_in_shape() -> None:
    """Test unary propagates dim var in shape.

    Unary(NEG) on a tensor whose first axis is a ``DimVar`` keeps
    that ``DimVar`` (with the same bounds) on the result type — the
    dynamic dim is not collapsed to a concrete int.
    """
    s = DimVar(name="S_ti", lo=1, hi=8)
    in_ty = TensorType(shape=(s, 8), dtype=DType.f32, layout=None, storage="gmem")
    x = Var(type=in_ty, name="x")
    call = Call(type=in_ty, target=Unary(kind=UnaryKind.NEG), args=(x,))
    out_ty = TypeInferVisitor().visit(call, TypeInferContext())
    assert out_ty.shape == (s, 8)

    assert out_ty.shape[0] is s
    assert (out_ty.shape[0].lo, out_ty.shape[0].hi) == (1, 8)


def test_a_copied_dim_var_is_the_canonical_one() -> None:
    """A deep copy of a shape keeps its ``DimVar`` identity.

    ``Module.cloned`` copies the IR graph when a child is attached, and a
    signature that came back holding a fresh ``DimVar`` would no longer bind
    against the one its caller wrote.
    """
    s = DimVar(name="S_copy", lo=1, hi=8)
    in_ty = TensorType(shape=(s, 8), dtype=DType.f32, layout=None, storage="gmem")

    assert copy.deepcopy(s) is s
    assert copy.deepcopy(in_ty).shape == in_ty.shape
    assert copy.deepcopy([in_ty])[0].shape[0] is s
