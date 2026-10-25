"""Resolving a decode function to one implementation at one size."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.specialize import (
    SpecializationError,
    bound_dims_of,
    display_name,
    origin_of,
    residual_dims,
    variant_for,
)
from tilefoundry_nvidia import CudaTarget

GQA_DECODE = Path(__file__).parents[4] / "core" / "fixtures" / "placed" / "gqa_decode.py"
IR = Path(__file__).parents[2] / "fixtures" / "ir"


def _gqa():
    """The decode Module, its entry, a context past the short strategy, and the split count."""
    constants = load_namespace(str(GQA_DECODE))[0]
    GqaOnline = load_authored_ir(f"{GQA_DECODE}:GqaOnline")
    steady = {"ctx_len": constants["SMALL_CONTEXT_T"] + constants["NUM_SPLITS"]}
    return GqaOnline, GqaOnline.entry_function(), steady, constants["NUM_SPLITS"]


def test_a_dimension_the_function_does_not_have_is_refused() -> None:
    GqaOnline, ENTRY, STEADY, _ = _gqa()
    with pytest.raises(SpecializationError, match="no dimension named"):
        check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), ENTRY, dims={**STEADY, "batch": 4})[1]


def test_specialising_nothing_is_refused() -> None:
    GqaOnline, ENTRY, _, _ = _gqa()
    with pytest.raises(SpecializationError, match="non-empty mapping"):
        check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), ENTRY, dims={})[1]


class SpecializeCase(Case):
    """A dispatching program resolved at chosen sizes."""


@case
class CallThroughPrototype(SpecializeCase):
    """The rebuilt call records which variant it chose, not the prototype."""

    path: Path = Path(__file__).parents[2] / "fixtures/placed/specialize_through_call.py"
    name: str | None = "ToCallee"

    def guard(self, state, problems):
        run = state.entry_function()

        for extent, expected in ((64, "pick_small"), (512, "pick_big")):
            rebuilt = check_program(replace(state, target=CudaTarget("nvidia.h200_sxm")), run, dims={"n": extent})[1].body.target
            assert display_name(origin_of(rebuilt)) == expected


def test_specialising_through_a_call_says_which_dimension_is_missing() -> None:
    """A callee dispatching on an unbound dimension names that dimension."""
    _MissingCalleeDimension = load_authored_ir(
        f"{IR / 'missing_callee_dimension.py'}:_MissingCalleeDimension"
    )
    caller = _MissingCalleeDimension.entry_function()

    with pytest.raises(SpecializationError, match="'call_n'.*was not given a size"):
        check_program(replace(_MissingCalleeDimension, target=CudaTarget("nvidia.h200_sxm")), caller, dims={"call_m": 4})[1]


@case
class NestedDispatch(SpecializeCase):
    """Selecting a variant keeps rebuilding through dispatches in its body."""

    path: Path = IR / "nested_dispatch.py"
    name: str | None = "_NestedDispatch"

    def guard(self, state, problems):
        run = state.entry_function()

        for extent, suffix in ((64, "small"), (512, "big")):
            concrete = check_program(replace(state, target=CudaTarget("nvidia.h200_sxm")), run, dims={"nested_n": extent})[1]
            middle = concrete.body.target
            inner = middle.body.target

            assert display_name(origin_of(middle)) == f"mid_{suffix}"
            assert display_name(origin_of(inner)) == f"inner_{suffix}"
            assert residual_dims(concrete) == ()


def test_a_specialised_function_states_extents_everywhere() -> None:
    """Nothing may still be a range -- including inside the callees.

    Nothing may still be a range -- including inside the callees, and in the
    shape-valued attributes their operations carry, which no signature shows.
    """
    GqaOnline, ENTRY, STEADY, _ = _gqa()
    concrete = check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), ENTRY, dims=STEADY)[1]

    assert residual_dims(concrete) == ()


def test_specialising_only_substitutes_dimensions() -> None:
    """The signature keeps its rank, dtype and storage; only extents move."""
    GqaOnline, ENTRY, STEADY, _ = _gqa()
    chosen = variant_for(ENTRY, STEADY)
    concrete = check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), ENTRY, dims=STEADY)[1]

    assert len(concrete.params) == len(chosen.params)
    for new, old in zip(concrete.params, chosen.params):
        assert len(new.type.shape) == len(old.type.shape)
        assert new.type.dtype == old.type.dtype
        assert new.type.storage == old.type.storage
    assert concrete.return_type.shape == (1, 1, 32, 128)
    assert concrete.return_type.dtype == chosen.return_type.dtype


def test_a_derived_extent_follows_the_dimension_it_derives_from() -> None:
    """The per-split block length is the context length over the splits.

    The per-split block length is the context length over the splits. It has
    to arrive at that number without being told it.
    """
    GqaOnline, ENTRY, STEADY, NUM_SPLITS = _gqa()
    concrete = check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), ENTRY, dims=STEADY)[1]
    shapes = {tuple(param.type.shape) for param in concrete.params}

    steady_context = STEADY["ctx_len"]
    assert (1, steady_context, 4, 128) in shapes
    assert all(isinstance(extent, int) for param in concrete.params for extent in param.type.shape)

    assert steady_context % NUM_SPLITS == 0


def test_the_same_request_gives_the_same_answer_every_time() -> None:
    """Rebuilding asks a type cache keyed on node identity for the types of nodes it then discards.

    Rebuilding asks a type cache keyed on node identity for the types of
    nodes it then discards. When one of those identities is reused, a node
    reads a type belonging to something else -- which showed up as the same
    call returning a different shape, or failing, from one run to the next.
    """
    GqaOnline, ENTRY, STEADY, _ = _gqa()
    first = check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), ENTRY, dims=STEADY)[1]
    others = [check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), ENTRY, dims=STEADY)[1] for _ in range(4)]

    for other in others:
        assert other.return_type == first.return_type
        assert [p.type for p in other.params] == [p.type for p in first.params]
        assert residual_dims(other) == ()


def test_specialising_one_function_does_not_disturb_the_next() -> None:
    """The callees are shared.

    The callees are shared, so a rebuild that left state on them would show
    up here rather than where it was caused.
    """
    GqaOnline, ENTRY, STEADY, _ = _gqa()
    partials = check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), GqaOnline.lookup("_ctx_partials"), dims=STEADY)[1]
    assert residual_dims(partials) == ()

    entry = check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), ENTRY, dims=STEADY)[1]
    assert entry.return_type.shape == (1, 1, 32, 128)
    assert residual_dims(entry) == ()


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
        if isinstance(target, Function):
            walk(target.body, depth + 1)
        for name in ("args", "elements", "yield_values"):
            for child in getattr(expr, name, ()) or ():
                walk(child, depth + 1)
        walk(getattr(expr, "body", None), depth + 1)

    walk(fn.body)
    return found


def test_a_derived_extent_reaches_the_operation_that_states_it() -> None:
    """At the split strategy the block length is the context over the splits.

    Nothing supplies it, and it has to arrive inside the callee's own reshape.
    """
    GqaOnline, ENTRY, STEADY, NUM_SPLITS = _gqa()
    concrete = check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), ENTRY, dims=STEADY)[1]
    block = STEADY["ctx_len"] // NUM_SPLITS

    targets = _reshape_targets(concrete)
    assert targets, "no reshape reached; the walk found nothing to check"
    assert any(block in shape for shape in targets), (
        f"no reshape states the derived block length {block}: {targets}"
    )
    assert all(isinstance(extent, int) for shape in targets for extent in shape), (
        f"a reshape still states a range: {targets}"
    )


def test_a_specialised_function_computes_what_the_prototype_computes() -> None:
    """The witness that shapes cannot give.

    The witness that shapes cannot give. Substituting an extent must not
    change what the program means, so the prototype -- which picks its
    implementation from the arguments it was handed -- and the function
    specialised for those same arguments have to agree on the answer.

    A small context keeps this runnable on a CPU; the size is in the first
    strategy's range, and the assertion below is that both routes choose it.
    """
    GqaOnline, ENTRY, _, _ = _gqa()
    context = 32
    dims = {"ctx_len": context}

    torch.manual_seed(0)
    q = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
    k = torch.randn(1, context, 4, 128, dtype=torch.bfloat16)
    v = torch.randn(1, context, 4, 128, dtype=torch.bfloat16)
    k_new = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16)
    v_new = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16)

    admitted = [variant for variant in ENTRY.variants if variant.specializations[0].match(context)]
    assert len(admitted) == 1
    assert variant_for(ENTRY, dims) is admitted[0]

    expected = evaluate(ENTRY, q, k, v, k_new, v_new)
    got = evaluate(check_program(replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm")), ENTRY, dims=dims)[1], q, k, v, k_new, v_new)

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    torch.testing.assert_close(got.float(), expected.float(), atol=0, rtol=0)


@case
class BodyOnlyLoopDimension(SpecializeCase):
    """A dimension that only a loop bound in the body uses, never the signature."""

    path: Path = IR / "body_only_loop_dimension.py"
    name: str | None = "_LoopOnly"

    def guard(self, state, problems):
        problems.run("still substituted", self.still_substituted, state)
        problems.run("two sizes told apart", self.two_sizes_told_apart, state)

    def still_substituted(self, state):
        """Whether to rebuild follows where the dimension occurs, not whether the signature moved.

        Whether to rebuild follows where the dimension occurs, not whether the
        signature moved. Deciding it from the parameters returns this function
        untouched -- shaped correctly, and still scanning a range.
        """
        authored = state.entry_function()
        assert residual_dims(authored) == ("loop_ctx",)
        assert all(
            isinstance(extent, int) for param in authored.params for extent in param.type.shape
        ), "the signature is already concrete, which is the point"

        concrete = check_program(replace(state, target=CudaTarget("nvidia.h200_sxm")), authored, dims={"loop_ctx": 512})[1]

        assert concrete is not authored
        assert residual_dims(concrete) == ()

    def two_sizes_told_apart(self, state):
        """A rebuild records the extents it was built at, and has to.

        A rebuild records the extents it was built at, and has to: for this
        function the signature is identical at every size.

        Anything holding two derived functions and asking whether they are the same
        program -- a report gathering several analyses of one selection -- would
        otherwise merge measurements of two different programs, because the only
        place the size occurs is a loop bound the signature never mentions.
        """
        authored = state.entry_function()

        small = check_program(replace(state, target=CudaTarget("nvidia.h200_sxm")), authored, dims={"loop_ctx": 8})[1]
        large = check_program(replace(state, target=CudaTarget("nvidia.h200_sxm")), authored, dims={"loop_ctx": 512})[1]

        assert origin_of(small) is authored
        assert origin_of(large) is authored

        assert [param.type for param in small.params] == [param.type for param in large.params]
        assert small.return_type == large.return_type

        assert bound_dims_of(small) == (("loop_ctx", 8),)
        assert bound_dims_of(large) == (("loop_ctx", 512),)
        assert bound_dims_of(small) != bound_dims_of(large)


CASES = special(SpecializeCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_specialize_case(case: SpecializeCase) -> None:
    check_case(case)
