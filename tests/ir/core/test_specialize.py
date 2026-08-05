"""Resolving a decode function to one implementation at one size."""

from __future__ import annotations

import pytest
import torch

from tests.fixtures.gqa_online import (
    MAX_CTX,
    NUM_SPLITS,
    SMALL_CONTEXT_T,
    GqaOnline,
)
from tilefoundry import func, module
from tilefoundry.dsl import Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- names resolved dynamically
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.hir.specialize import (
    SpecializationError,
    bound_dims_of,
    is_concrete,
    origin_of,
    residual_dims,
    specialize_function,
    variant_for,
)
from tilefoundry.ir.types.dim import DimVar

ENTRY = GqaOnline.entry_function()
STEADY = {"ctx_len": SMALL_CONTEXT_T}
_LOOP_CTX = DimVar("loop_ctx", 1, 4097)


def test_the_model_is_dynamic_in_its_context_length_alone() -> None:
    """A derived extent is an expression, so it is not a second dimension the
    caller has to know how to compute."""
    assert set(residual_dims(ENTRY)) == {"ctx_len"}


def test_a_size_selects_the_one_implementation_that_covers_it() -> None:
    short = variant_for(ENTRY, {"ctx_len": SMALL_CONTEXT_T - 1})
    long = variant_for(ENTRY, STEADY)

    assert short is not long
    assert [(p.dim_var, p.lo, p.hi) for p in short.specializations] == [
        ("ctx_len", 0, SMALL_CONTEXT_T)
    ]
    assert [(p.dim_var, p.lo, p.hi) for p in long.specializations] == [
        ("ctx_len", SMALL_CONTEXT_T, MAX_CTX)
    ]


def test_a_size_no_implementation_covers_is_refused() -> None:
    with pytest.raises(SpecializationError, match="no variant covering"):
        variant_for(ENTRY, {"ctx_len": MAX_CTX})


def test_choosing_an_implementation_needs_the_dimension_it_turns_on() -> None:
    """Selection reads the dimensions the variants name. Skipping an unstated
    one would pick an implementation the caller never chose."""
    with pytest.raises(SpecializationError, match="was not given a size"):
        variant_for(ENTRY, {"batch": 4})


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


def _reshape_targets(fn) -> list[tuple]:
    """Every `new_shape` an operation states anywhere *fn* reaches."""
    found: list[tuple] = []
    seen: set[int] = set()

    def walk(expr, depth: int = 0) -> None:
        if expr is None or depth > 256 or id(expr) in seen:
            return
        seen.add(id(expr))
        target = getattr(expr, "target", None)
        if target is not None and type(target).__name__ == "Reshape":
            found.append(tuple(target.new_shape))
        if isinstance(target, type(ENTRY)):
            walk(target.body, depth + 1)
        for name in ("args", "elements", "init_args", "yield_values"):
            for child in getattr(expr, name, ()) or ():
                walk(child, depth + 1)
        walk(getattr(expr, "body", None), depth + 1)

    walk(fn.body)
    return found


def test_a_derived_extent_reaches_the_operation_that_states_it() -> None:
    """At the split strategy the block length is the context over the splits.
    Nothing supplies it, and it has to arrive inside the callee's own reshape."""
    concrete = specialize_function(ENTRY, STEADY)
    block = SMALL_CONTEXT_T // NUM_SPLITS

    targets = _reshape_targets(concrete)
    assert targets, "no reshape reached; the walk found nothing to check"
    assert any(block in shape for shape in targets), (
        f"no reshape states the derived block length {block}: {targets}"
    )
    assert all(
        isinstance(extent, int) for shape in targets for extent in shape
    ), f"a reshape still states a range: {targets}"


def test_a_specialised_function_computes_what_the_prototype_computes() -> None:
    """The witness that shapes cannot give. Substituting an extent must not
    change what the program means, so the prototype -- which picks its
    implementation from the arguments it was handed -- and the function
    specialised for those same arguments have to agree on the answer.

    A small context keeps this runnable on a CPU; the size is in the first
    strategy's range, and the assertion below is that both routes choose it.
    """
    context = 32
    dims = {"ctx_len": context}

    torch.manual_seed(0)
    q = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
    k = torch.randn(1, context, 4, 128, dtype=torch.bfloat16)
    v = torch.randn(1, context, 4, 128, dtype=torch.bfloat16)
    k_new = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16)
    v_new = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16)

    # What the evaluator would pick, decided the way it decides: the one
    # pattern that admits the context length the arguments actually carry.
    admitted = [
        variant
        for variant in ENTRY.variants
        if variant.specializations[0].match(context)
    ]
    assert len(admitted) == 1
    assert variant_for(ENTRY, dims) is admitted[0]

    expected = evaluate(ENTRY, q, k, v, k_new, v_new, device="cpu")
    got = evaluate(
        specialize_function(ENTRY, dims), q, k, v, k_new, v_new, device="cpu"
    )

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    torch.testing.assert_close(got.float(), expected.float(), atol=0, rtol=0)


@module(entry="main", target="cuda")
class _LoopOnly:
    """A context length that appears nowhere in the signature.

    The loop scans it. Nothing about the parameters or the return says how long
    it is, which is what makes this the case a signature-driven rebuild misses.
    """

    topologies = (Topology("cta", 1),)

    @func
    def main(x: Tensor[(8,), "f32"]):
        total = tf.zeros(shape=(8,), dtype="f32")
        for _ in tile(_LOOP_CTX):
            total = tf.add(total, x)
        return total


def test_a_dimension_only_the_body_uses_is_still_substituted() -> None:
    """Whether to rebuild follows where the dimension occurs, not whether the
    signature moved. Deciding it from the parameters returns this function
    untouched -- shaped correctly, and still scanning a range."""
    authored = _LoopOnly.entry_function()
    assert residual_dims(authored) == ("loop_ctx",)
    assert all(
        isinstance(extent, int)
        for param in authored.params
        for extent in param.type.shape
    ), "the signature is already concrete, which is the point"

    concrete = specialize_function(authored, {"loop_ctx": 512})

    assert concrete is not authored
    assert residual_dims(concrete) == ()
    assert is_concrete(concrete)


def test_two_sizes_of_a_body_only_dimension_are_told_apart() -> None:
    """A rebuild records the extents it was built at, and has to: for this
    function the signature is identical at every size.

    Anything holding two derived functions and asking whether they are the same
    program -- a report gathering several analyses of one selection -- would
    otherwise merge measurements of two different programs, because the only
    place the size occurs is a loop bound the signature never mentions.
    """
    authored = _LoopOnly.entry_function()

    small = specialize_function(authored, {"loop_ctx": 8})
    large = specialize_function(authored, {"loop_ctx": 512})

    assert origin_of(small) is authored
    assert origin_of(large) is authored
    # The premise: nothing in either signature distinguishes them.
    assert [param.type for param in small.params] == [
        param.type for param in large.params
    ]
    assert small.return_type == large.return_type

    assert bound_dims_of(small) == (("loop_ctx", 8),)
    assert bound_dims_of(large) == (("loop_ctx", 512),)
    assert bound_dims_of(small) != bound_dims_of(large)


def test_the_report_refuses_two_sizes_of_a_body_only_dimension() -> None:
    """The refusal itself, through the function that has to make it.

    Asserted against `report` rather than against the recorded extents alone,
    because what matters is that the caller who would mix them is stopped -- and
    a comparison that read the signature would let these two through.
    """
    from tilefoundry.inspection.analysis_report import _same_program  # noqa: PLC0415

    authored = _LoopOnly.entry_function()
    small = specialize_function(authored, {"loop_ctx": 8})
    large = specialize_function(authored, {"loop_ctx": 512})
    again = specialize_function(authored, {"loop_ctx": 8})

    assert not _same_program(small, large)
    assert _same_program(small, again)
    assert _same_program(small, small)
    # An undecorated function has no recorded extents, so it is only ever itself.
    assert not _same_program(authored, small)
