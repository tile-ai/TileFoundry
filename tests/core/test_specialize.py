"""Resolving a decode function to one implementation at one size."""

from __future__ import annotations

import pytest

from tests.models.qwen3_5_30b_a3b.gqa_online import (
    MAX_CTX,
    NUM_SPLITS,
    SMALL_CONTEXT_T,
    GqaOnline,
)
from tilefoundry.ir.hir.specialize import (
    SpecializationError,
    is_concrete,
    residual_dims,
    specialize_function,
    variant_for,
)

ENTRY = GqaOnline.entry_function()
STEADY = {"ctx_len": SMALL_CONTEXT_T, "seq_len": 1}


def test_the_model_is_dynamic_in_its_context_length_alone() -> None:
    """A derived extent is an expression, so it is not a second dimension the
    caller has to know how to compute."""
    assert set(residual_dims(ENTRY)) == {"ctx_len", "seq_len"}


def test_a_size_selects_the_one_implementation_that_covers_it() -> None:
    short = variant_for(ENTRY, {"ctx_len": SMALL_CONTEXT_T - 1, "seq_len": 1})
    long = variant_for(ENTRY, STEADY)

    assert short is not long
    assert [(p.dim_var, p.lo, p.hi) for p in short.specializations] == [
        ("ctx_len", 1, SMALL_CONTEXT_T)
    ]
    assert [(p.dim_var, p.lo, p.hi) for p in long.specializations] == [
        ("ctx_len", SMALL_CONTEXT_T, MAX_CTX + 1)
    ]


def test_a_size_no_implementation_covers_is_refused() -> None:
    with pytest.raises(SpecializationError, match="no variant covering"):
        variant_for(ENTRY, {"ctx_len": MAX_CTX + 1, "seq_len": 1})


def test_choosing_an_implementation_needs_the_dimension_it_turns_on() -> None:
    """Selection reads the dimensions the variants name. Skipping an unstated
    one would pick an implementation the caller never chose."""
    with pytest.raises(SpecializationError, match="was not given a size"):
        variant_for(ENTRY, {"seq_len": 1})


def test_a_dimension_the_function_does_not_have_is_refused() -> None:
    with pytest.raises(SpecializationError, match="no dimension named"):
        specialize_function(ENTRY, {**STEADY, "batch": 4})


def test_specialising_nothing_is_refused() -> None:
    with pytest.raises(SpecializationError, match="at least one dimension"):
        specialize_function(ENTRY, {})


def test_a_specialised_function_states_extents_everywhere() -> None:
    """Nothing may still be a range -- including inside the callees, and in the
    shape-valued attributes their operations carry, which no signature shows."""
    concrete = specialize_function(ENTRY, STEADY)

    assert residual_dims(concrete) == ()
    assert is_concrete(concrete)


def test_specialising_only_substitutes_dimensions() -> None:
    """The signature keeps its rank, dtype and storage; only extents move."""
    chosen = variant_for(ENTRY, STEADY)
    concrete = specialize_function(ENTRY, STEADY)

    assert len(concrete.params) == len(chosen.params)
    for new, old in zip(concrete.params, chosen.params):
        assert len(new.type.shape) == len(old.type.shape)
        assert new.type.dtype == old.type.dtype
        assert new.type.storage == old.type.storage
    assert concrete.return_type.shape == (1, 1, 32, 128)
    assert concrete.return_type.dtype == chosen.return_type.dtype


def test_a_derived_extent_follows_the_dimension_it_derives_from() -> None:
    """The per-split block length is the context length over the splits. It has
    to arrive at that number without being told it."""
    concrete = specialize_function(ENTRY, STEADY)
    shapes = {tuple(param.type.shape) for param in concrete.params}

    assert (1, SMALL_CONTEXT_T, 4, 128) in shapes
    assert all(
        isinstance(extent, int)
        for param in concrete.params
        for extent in param.type.shape
    )
    # Nothing states the block length, so if it were wrong the reshape inside
    # the callee would have rejected the rebuild rather than reach here.
    assert SMALL_CONTEXT_T % NUM_SPLITS == 0


def test_the_same_request_gives_the_same_answer_every_time() -> None:
    """Rebuilding asks a type cache keyed on node identity for the types of
    nodes it then discards. When one of those identities is reused, a node
    reads a type belonging to something else -- which showed up as the same
    call returning a different shape, or failing, from one run to the next."""
    first = specialize_function(ENTRY, STEADY)
    others = [specialize_function(ENTRY, STEADY) for _ in range(4)]

    for other in others:
        assert other.return_type == first.return_type
        assert [p.type for p in other.params] == [p.type for p in first.params]
        assert residual_dims(other) == ()


def test_specialising_one_function_does_not_disturb_the_next() -> None:
    """The callees are shared, so a rebuild that left state on them would show
    up here rather than where it was caused."""
    partials = specialize_function(GqaOnline.lookup("_ctx_partials"), STEADY)
    assert residual_dims(partials) == ()

    entry = specialize_function(ENTRY, STEADY)
    assert entry.return_type.shape == (1, 1, 32, 128)
    assert residual_dims(entry) == ()


def test_a_function_with_nothing_to_bind_is_returned_unchanged() -> None:
    concrete = specialize_function(ENTRY, STEADY)

    assert specialize_function(concrete, {"ctx_len": SMALL_CONTEXT_T}) is concrete
