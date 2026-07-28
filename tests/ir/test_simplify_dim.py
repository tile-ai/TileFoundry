"""Construction-time folding for dim arithmetic Calls.

Folding is what makes a static shape one canonical value, so the boundaries are
where it must *not* fold (a symbolic operand, a division by zero, a bool) and
where the folded result must arrive as a plain ``int`` rather than a ``Constant``
or a nested ``Call`` — two shapes that print and compare differently and produced
a real broadcast failure.
"""
from __future__ import annotations

from tilefoundry.ir.core import TypeInferContext
from tilefoundry.ir.core.expr import Call, Constant, Var
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir._helpers import broadcast_shapes
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import DType, TensorType
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


def _i64(v: int) -> Constant:
    return Constant(type=TensorType.scalar(DType.i64), value=v)


def _sym(name: str) -> Call:
    """A symbolic dim Expr — wrap a DimVar in a Call so its
    presence breaks all-Constant folding."""
    return Call(
        type=TensorType.scalar(DType.i64),
        target=DimVar(name=name, lo=1, hi=1024),
        args=(),
    )


def test_simplify_dim_folds_all_constant_args() -> None:
    """When both args are int Constants, simplify_dim returns a folded
    Constant with the canonical i64 dim type. Floor division follows Python
    ``//`` (floor), not C truncation, which is the convention every tiling
    expression is written against."""
    table = [
        (DimAdd, 3, 4, 7),
        (DimSub, 10, 4, 6),
        (DimMul, 3, 4, 12),
        (DimFloorDiv, 17, 4, 4),
        (DimMod, 17, 5, 2),
        (DimMin, 7, 3, 3),
        (DimMax, 7, 3, 7),
        (DimFloorDiv, -7, 2, -4),
    ]
    for op_cls, a, b, expected in table:
        result = simplify_dim(op_cls, (_i64(a), _i64(b)))
        assert isinstance(result, Constant), (
            f"{op_cls.__name__}: expected Constant, got {type(result).__name__}"
        )
        assert result.value == expected
        assert result.type == TensorType.meta_scalar()


def test_simplify_dim_refuses_to_fold_outside_all_int_constants() -> None:
    """Three non-foldable inputs, in either operand position:

    - a non-Constant arg (a symbolic DimVar Call): the Call survives with no
      algebraic identity applied, so ``x + 0`` stays a Call;
    - division by zero: folding to ``Constant(0)`` would mask a real bug, so the
      Call survives for a later verify pass to flag;
    - ``Constant(True)``: a bool is not an int dim value.
    """
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


def test_a_fully_static_dim_has_one_canonical_int_representation() -> None:
    """``TensorType`` folds integer-valued ``Constant`` shape dims to plain ``int``,
    leaving ``DimVar`` / dynamic dims alone, and a typeinfer result built through
    ``simplify_dim`` arrives the same way.

    Regression: a ``Slice`` output used to carry flat ``Constant`` dims, so
    ``Binary`` broadcast failed comparing ``(1,4,32,Constant(128))`` against a
    param's ``(1,4,32,128)`` — the rotate-half bug.
    """
    ty = TensorType(
        shape=(_i64(1), _i64(4), _i64(32), _i64(128)),
        dtype=DType.bf16, layout=None, storage="gmem",
    )
    assert ty.shape == (1, 4, 32, 128)
    assert all(isinstance(d, int) and not isinstance(d, bool) for d in ty.shape)

    s = DimVar(name="S_ti", lo=1, hi=8)
    mixed = TensorType(shape=(s, _i64(128)), dtype=DType.f32, layout=None, storage="gmem")
    assert mixed.shape[0] is s          # symbolic dim untouched
    assert mixed.shape[1] == 128 and isinstance(mixed.shape[1], int)

    # The Slice typeinfer builds its output shape via ``simplify_dim``, collapsing
    # the nested ``DimFloorDiv(DimAdd(DimSub(...)))`` chain to a single value.
    param = TensorType(shape=(1, 4, 32, 128), dtype=DType.bf16, layout=None, storage="gmem")
    x = Constant(type=param, value=None)
    sliced = TypeInferContext().type_of(Call(
        type=TensorType.scalar(DType.bf16),  # ignored: typeinfer fills in
        target=Slice(begin=(0, 0, 0, 0), end=(1, 4, 32, 128), strides=(1, 1, 1, 1)),
        args=(x,),
    ))
    for dim in sliced.shape:
        assert isinstance(dim, int) and not isinstance(dim, bool), (
            f"expected canonical int static dim, got {type(dim).__name__}: {dim}"
        )
    assert sliced.shape == param.shape == (1, 4, 32, 128)
    assert broadcast_shapes(sliced.shape, param.shape) == (1, 4, 32, 128)


def test_unary_propagates_dim_var_in_shape() -> None:
    """Unary(NEG) on a tensor whose first axis is a ``DimVar`` keeps
    that ``DimVar`` (with the same bounds) on the result type — the
    dynamic dim is not collapsed to a concrete int."""
    s = DimVar(name="S_ti", lo=1, hi=8)
    in_ty = TensorType(shape=(s, 8), dtype=DType.f32, layout=None, storage="gmem")
    x = Var(type=in_ty, name="x")
    call = Call(type=in_ty, target=Unary(kind=UnaryKind.NEG), args=(x,))
    out_ty = TypeInferContext().type_of(call)
    assert out_ty.shape == (s, 8)
    # Same (name, lo, hi) DimVar identity (cached).
    assert out_ty.shape[0] is s
    assert (out_ty.shape[0].lo, out_ty.shape[0].hi) == (1, 8)
