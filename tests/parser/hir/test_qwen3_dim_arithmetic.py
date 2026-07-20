"""DimVar arithmetic DSL surface tests.

Covers ``DimVar.__add__`` / ``__radd__`` producing a canonical
``DimAdd`` ``Call``, the structural shape of that ``Call``,
``verify_function`` accepting nested ``DimVar`` through dim
arithmetic in a signature, and ``shape_entry_str`` rendering
``DimAdd(var, 1)`` as ``"<name> + 1"``.
"""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.inspection.python_printer import _shape_tuple, shape_entry_str
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.types.dim import DimAdd

CTX_LEN = DimVar("CTX_LEN", 1, 4097)


def _is_dim_add_of(node, dim_var: DimVar, k: int) -> bool:
    """``True`` iff *node* is a dim-arithmetic Call whose target is
    ``DimAdd`` and whose operands resolve to *dim_var* (canonical
    DimVar instance) and the integer literal *k* (regardless of
    whether the literal is wrapped as ``Constant(i64, k)``,
    ``DimConst(k)`` Call, or anything else)."""
    if not isinstance(node, Call):
        return False
    if not isinstance(node.target, DimAdd):
        return False
    if len(node.args) != 2:
        return False
    has_var = False
    has_k = False
    for arg in node.args:
        if arg is dim_var:
            has_var = True
            continue
        if isinstance(arg, Constant) and arg.value == k:
            has_k = True
            continue
        # Tolerate a ``DimConst`` Op wrapped as ``Call(target=DimConst(value=k))``.
        if isinstance(arg, Call):
            from tilefoundry.ir.types.dim import DimConst  # noqa: PLC0415
            if isinstance(arg.target, DimConst) and arg.target.value == k:
                has_k = True
                continue
    return has_var and has_k


def test_dim_var_plus_int_produces_dim_add_call() -> None:
    expr = CTX_LEN + 1
    assert _is_dim_add_of(expr, CTX_LEN, 1)


def test_int_plus_dim_var_produces_dim_add_call() -> None:
    # ``__radd__`` is commutative with ``__add__``: same canonical Call.
    expr = 1 + CTX_LEN
    assert _is_dim_add_of(expr, CTX_LEN, 1)


def test_dim_var_plus_zero_is_still_a_call() -> None:
    # No algebraic folding for ``x + 0`` per ``simplify_dim`` contract.
    expr = CTX_LEN + 0
    assert _is_dim_add_of(expr, CTX_LEN, 0)


@pytest.mark.parametrize(
    "bad",
    [
        object(),
        # ``bool`` is a subclass of ``int`` in Python — reject explicitly
        # so ``CTX_LEN + True`` does not silently become ``CTX_LEN + 1``.
        True,
    ],
)
def test_dim_var_plus_bad_operand_raises_type_error(bad) -> None:
    with pytest.raises(TypeError):
        _ = CTX_LEN + bad


def test_simplify_dim_rejects_bool_operand() -> None:
    """``simplify_dim`` canonicalises raw ``int`` shape entries to
    ``Constant(i64, value)``; a stray ``bool`` (subclass of int) must
    raise rather than slip into ``Call(args=(True, ...))``.
    """
    from tilefoundry.ir.types.dim import DimAdd, simplify_dim  # noqa: PLC0415

    with pytest.raises(TypeError, match="bool operand"):
        simplify_dim(DimAdd, (True, CTX_LEN))

    with pytest.raises(TypeError, match="bool operand"):
        simplify_dim(DimAdd, (CTX_LEN, False))


def test_dim_var_arithmetic_renders_in_shape_strings() -> None:
    assert shape_entry_str(CTX_LEN) == "CTX_LEN"
    assert shape_entry_str(CTX_LEN + 1) == "CTX_LEN + 1"
    shape = (1, 2, CTX_LEN + 1, 256)
    assert _shape_tuple(shape) == "(1, 2, CTX_LEN + 1, 256)"


@func
def dim_add_shape_fn(
    x: Tensor[(1, 2, CTX_LEN + 1, 256), "bf16"],
):
    return x


def test_dim_var_arithmetic_in_parsed_function_signature() -> None:
    ir = dim_add_shape_fn
    param_ty = ir.params[0].type
    assert _is_dim_add_of(param_ty.shape[2], CTX_LEN, 1)
    assert param_ty.shape[0] == 1
    assert param_ty.shape[1] == 2
    assert param_ty.shape[3] == 256
    ret_ty = ir.return_type
    assert _is_dim_add_of(ret_ty.shape[2], CTX_LEN, 1)


@func
def dim_add_consistency_fn(
    x: Tensor[(CTX_LEN,), "bf16"],
    y: Tensor[(CTX_LEN + 1,), "bf16"],
):
    return x


def test_verify_function_anchors_same_name_through_dim_add() -> None:
    """Two params share the same ``CTX_LEN``, one directly and one
    nested inside ``DimAdd``; the verifier's recurse-into-Call walk
    must anchor both and not flag a consistency error.
    """
    ir = dim_add_consistency_fn
    verify_function(ir)
