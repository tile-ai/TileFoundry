"""Choosing an extent for a dimension that was declared as a range."""

from __future__ import annotations

import pytest

from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.substitute import (
    DimSubstitutionError,
    dim_vars_in,
    has_symbolic_dims,
    substitute_dims,
    substitute_shape_dim,
)
from tilefoundry.ir.types.tensor_type import TupleType

CTX = DimVar("ctx_len", 1, 262145)
SEQ = DimVar("seq_len", 1, 5)


def _tensor(*shape) -> TensorType:
    return TensorType(shape=tuple(shape), dtype=DType.f32, layout=None, storage="gmem")


def test_a_bound_dimension_becomes_the_extent_it_was_given() -> None:
    bound = substitute_dims(_tensor(1, CTX, 4, 128), {"ctx_len": 4096})

    assert bound.shape == (1, 4096, 4, 128)
    assert not has_symbolic_dims(bound)


def test_an_unbound_dimension_stays_a_range() -> None:
    """A context length is chosen while the sequence length stays open.

    A context length is chosen while the sequence length stays open, so
    binding one dimension must not demand the others.
    """
    bound = substitute_dims(_tensor(1, SEQ, CTX, 128), {"seq_len": 1})

    assert bound.shape == (1, 1, CTX, 128)
    assert dim_vars_in(bound) == ("ctx_len",)
    assert has_symbolic_dims(bound)


def test_arithmetic_over_a_bound_dimension_folds_to_its_value() -> None:
    """Test arithmetic over a bound dimension folds to its value.

    A shape written as a division has to become a number, or the analysis
    inherits an expression it cannot count.
    """
    folded = substitute_dims(_tensor(1, CTX // 8), {"ctx_len": 4096})

    assert folded.shape == (1, 512)


def test_arithmetic_over_an_unbound_dimension_is_left_alone() -> None:
    expression = CTX // 8
    assert substitute_shape_dim(expression, {"seq_len": 1}) is expression


def test_a_tuple_substitutes_every_leaf() -> None:
    pair = TupleType(fields=(_tensor(1, CTX), _tensor(CTX, 8)))

    bound = substitute_dims(pair, {"ctx_len": 64})

    assert [field.shape for field in bound.fields] == [(1, 64), (64, 8)]


def test_an_extent_outside_the_declared_range_is_refused() -> None:
    """The declaration states what the model was written to handle.

    The declaration states what the model was written to handle. Accepting
    a length outside it would report an answer for a program nobody wrote.
    """
    with pytest.raises(DimSubstitutionError, match=r"\[1, 262145\) and cannot take"):
        substitute_dims(_tensor(CTX), {"ctx_len": 262145})

    with pytest.raises(DimSubstitutionError, match="cannot take 0"):
        substitute_dims(_tensor(CTX), {"ctx_len": 0})

    assert substitute_dims(_tensor(CTX), {"ctx_len": 262144}).shape == (262144,)


def test_the_bounds_are_half_open_like_the_specialisations_that_state_them() -> None:
    """A specialisation states its range half-open.

    A specialisation states its range half-open. If substitution disagreed
    about the endpoint, a length would be admitted by one and refused by the
    other.
    """
    assert substitute_dims(_tensor(CTX), {"ctx_len": CTX.lo}).shape == (CTX.lo,)
    with pytest.raises(DimSubstitutionError):
        substitute_dims(_tensor(CTX), {"ctx_len": CTX.hi})


def test_a_non_integer_extent_is_refused() -> None:
    with pytest.raises(DimSubstitutionError, match="takes an integer extent"):
        substitute_dims(_tensor(CTX), {"ctx_len": 4096.0})

    with pytest.raises(DimSubstitutionError, match="takes an integer extent"):
        substitute_dims(_tensor(CTX), {"ctx_len": True})


def test_a_type_with_nothing_to_bind_is_returned_unchanged() -> None:
    """Identity, not a copy: a caller comparing before and after should see that nothing happened.

    Identity, not a copy: a caller comparing before and after should see
    that nothing happened.
    """
    static = _tensor(1, 64, 128)

    assert substitute_dims(static, {"ctx_len": 4096}) is static
    assert dim_vars_in(static) == ()


def test_the_dimensions_present_are_reported_in_first_seen_order() -> None:
    assert dim_vars_in(_tensor(1, SEQ, CTX)) == ("seq_len", "ctx_len")
    assert dim_vars_in(_tensor(CTX, CTX)) == ("ctx_len",)
    assert dim_vars_in(TupleType(fields=(_tensor(SEQ), _tensor(CTX)))) == (
        "seq_len",
        "ctx_len",
    )
