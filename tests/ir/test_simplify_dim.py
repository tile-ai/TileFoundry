"""Construction-time folding for dim arithmetic Calls."""
from __future__ import annotations

import pytest

from tests.parser.hir.test_demo_proj_qkv import proj_qkv_with_mma
from tilefoundry.ir.core import TypeInferContext
from tilefoundry.ir.core.expr import Call, Constant, Var
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math._helpers import _broadcast, _shapes_equal
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


@pytest.mark.parametrize(
    "op_cls,a,b,expected",
    [
        (DimAdd, 3, 4, 7),
        (DimSub, 10, 4, 6),
        (DimMul, 3, 4, 12),
        (DimFloorDiv, 17, 4, 4),
        (DimMod, 17, 5, 2),
        (DimMin, 7, 3, 3),
        (DimMax, 7, 3, 7),
        # Floor-div sign convention (Python //, not C truncation).
        (DimFloorDiv, -7, 2, -4),
    ],
)
def test_simplify_dim_folds_all_constant_args(op_cls, a, b, expected) -> None:
    """When both args are int Constants, simplify_dim returns a folded
    Constant with the canonical i64 dim type."""
    result = simplify_dim(op_cls, (_i64(a), _i64(b)))
    assert isinstance(result, Constant), (
        f"{op_cls.__name__}: expected Constant, got {type(result).__name__}"
    )
    assert result.value == expected
    assert result.type == TensorType.scalar(DType.i64)


@pytest.mark.parametrize("op_cls", [DimAdd, DimFloorDiv])
def test_simplify_dim_keeps_call_when_arg_is_symbolic(op_cls) -> None:
    """If any arg is non-Constant (e.g. a symbolic DimVar Call),
    simplify_dim returns a Call with no algebraic identity
    simplification (``x + 0`` stays as the Call)."""
    sym = _sym("M")
    result = simplify_dim(op_cls, (sym, _i64(0)))
    assert isinstance(result, Call)
    assert isinstance(result.target, op_cls)
    assert result.args == (sym, _i64(0))
    # Reverse-order also stays as Call.
    result2 = simplify_dim(op_cls, (_i64(0), sym))
    assert isinstance(result2, Call)
    assert isinstance(result2.target, op_cls)


@pytest.mark.parametrize("op_cls", [DimFloorDiv])
def test_simplify_dim_preserves_call_for_div_by_zero(op_cls) -> None:
    """Division / mod by zero is NOT silently folded; the original
    Call survives so a later verify pass can flag the error.
    Folding to ``Constant(0)`` would mask a real bug."""
    result = simplify_dim(op_cls, (_i64(10), _i64(0)))
    assert isinstance(result, Call)
    assert isinstance(result.target, op_cls)
    assert result.args == (_i64(10), _i64(0))


def test_simplify_dim_does_not_fold_x_plus_zero() -> None:
    """Algebraic identity folding is out of scope.
    ``simplify_dim(DimAdd, (Var, Constant(0)))`` stays as a Call."""
    sym = _sym("N")
    result = simplify_dim(DimAdd, (sym, _i64(0)))
    assert isinstance(result, Call)
    assert isinstance(result.target, DimAdd)


def test_simplify_dim_bool_constant_falls_back_to_call() -> None:
    """Booleans are not int dim values; simplify_dim must not treat
    ``Constant(True)`` as an int and fold it."""
    b_true = Constant(type=TensorType.scalar(DType.i64), value=True)
    result = simplify_dim(DimAdd, (b_true, _i64(1)))
    assert isinstance(result, Call)
    assert isinstance(result.target, DimAdd)


def test_proj_qkv_viewer_collapses_all_constant_dim_subtrees() -> None:
    """Every all-``Constant`` arithmetic sub-tree in the
    ``proj_qkv_with_mma`` demo's reshard tensor types collapses to
    a single ``Constant``. The symbolic skeleton driven by the
    induction var ``ok`` does NOT collapse to ``16`` — algebraic
    identity folding (``(x + 16) - x → 16``) is explicitly out of
    scope.

    This guards against a regression where a producer skips the
    ``simplify_dim`` helper and reintroduces a `DimSub(Constant(a),
    Constant(b))` style chain.
    """

    fn = proj_qkv_with_mma

    dim_op_classes = (
        DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax,
    )

    def _walk(expr) -> None:
        if isinstance(expr, Call) and isinstance(expr.target, dim_op_classes):
            if all(isinstance(a, Constant) for a in expr.args):
                raise AssertionError(
                    f"all-Constant {type(expr.target).__name__} chain survived "
                    f"in viewer model — simplify_dim was bypassed: "
                    f"args={expr.args}"
                )
            for a in expr.args:
                _walk(a)

    # Walk every Expr in the function body, including GridRegion
    # children + Call args + TensorType shape entries.

    seen: set[int] = set()

    def _visit(expr) -> None:
        if id(expr) in seen:
            return
        seen.add(id(expr))
        _walk(expr)
        ty = getattr(expr, "type", None)
        if isinstance(ty, TensorType):
            for dim in ty.shape:
                _visit(dim)
        if isinstance(expr, GridRegionExpr):
            _visit(expr.body)
            for y in expr.yield_values:
                _visit(y)
            for c in expr.carried_args:
                _visit(c)
        elif isinstance(expr, Call):
            for a in expr.args:
                _visit(a)

    _visit(fn.body)


def test_slice_typeinfer_canonicalizes_static_shape_to_int() -> None:
    """End-to-end smoke: the Slice typeinfer constructs its output shape via
    ``simplify_dim`` (collapsing the nested ``DimFloorDiv(DimAdd(DimSub(...)))``
    chain to a single value), and ``TensorType`` then canonicalizes that
    fully-static dim to a plain ``int`` — not a flat ``Constant`` and not a
    nested ``Call`` chain. (Static dims have one canonical representation; see
    ``TensorType.__post_init__``.)"""

    x = Constant(
        type=TensorType(shape=(_i64(16), _i64(2048)),
                        dtype=DType.bf16, layout=None, storage="gmem"),
        value=None,
    )
    call = Call(
        type=TensorType.scalar(DType.bf16),  # ignored: typeinfer fills in
        target=Slice(begin=(0, 0), end=(16, 16), strides=(1, 1)),
        args=(x,),
    )
    ty = TypeInferContext().type_of(call)
    assert isinstance(ty, TensorType)
    for dim in ty.shape:
        assert isinstance(dim, int) and not isinstance(dim, bool), (
            f"expected canonical int static dim, got {type(dim).__name__}: {dim}"
        )
    assert ty.shape == (16, 16)


def test_tensor_type_canonicalizes_constant_dim_to_int() -> None:
    """``TensorType`` folds integer-valued ``Constant`` shape dims to plain
    ``int`` (one canonical static rep), leaving ``DimVar`` / dynamic dims alone."""
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


def test_slice_output_broadcasts_against_param_int_shape() -> None:
    """Regression: a ``Slice`` output (formerly flat ``Constant`` dims) now
    canonicalizes to ``int`` and compares equal to a param-style ``int`` shape,
    so ``Binary`` broadcast no longer fails with
    ``(1,4,32,Constant(128))`` vs ``(1,4,32,128)`` (the rotate-half bug)."""
    x = Constant(
        type=TensorType(shape=(1, 4, 32, 128), dtype=DType.bf16, layout=None, storage="gmem"),
        value=None,
    )
    sliced = TypeInferContext().type_of(Call(
        type=TensorType.scalar(DType.bf16),
        target=Slice(begin=(0, 0, 0, 0), end=(1, 4, 32, 128), strides=(1, 1, 1, 1)),
        args=(x,),
    ))
    param = TensorType(shape=(1, 4, 32, 128), dtype=DType.bf16, layout=None, storage="gmem")
    assert sliced.shape == param.shape == (1, 4, 32, 128)
    assert _shapes_equal(sliced.shape, param.shape)
    assert _broadcast(sliced.shape, param.shape) == (1, 4, 32, 128)


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
    assert out_ty.shape[0].lo == 1
    assert out_ty.shape[0].hi == 8
