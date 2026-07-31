"""What `case.py` claims about this model, held to being true today.

The corpus harness asks these questions once the case is named in the registry;
until it is, a case is an unverified assertion, and an unverified case is exactly
what a capability matrix exists to prevent. So each claim is checked here: the
context length the case names is one the description can really be asked at, and
every question the case selects really does get an answer.

These questions were blocked when this package was migrated -- MLA splits each
head into its unrotated and rotated halves and DeepSeek's interleaved rotation
takes every other channel, so both kernels are built out of `tf.slice` and
`tf.concat`, and the cost registry could evaluate neither. It can now, so the
case states no gate and this file is what would notice if that regressed.
"""

from __future__ import annotations

import pytest

from tests.models.deepseek_v4_flash.case import ANALYZED_AT, CASE
from tests.models.deepseek_v4_flash.model import REAL
from tests.models.fixtures import ACCEPTANCE
from tilefoundry.analysis import ANALYSES, analyze
from tilefoundry.ir.hir.specialize import specialize_concretely
from tilefoundry.ir.types.substitute import DimSubstitutionError
from tilefoundry.schedule import ScheduleOptions, schedule

#: One case is one CP-SAT solve, so the budget is stated rather than inherited.
#: The first plan is the one this asks about -- that a plan exists and verifies --
#: and the solver cannot prove a makespan optimal on this workload, so without
#: stopping it improves the answer until the timeout and then reports the same
#: verdict. That was 60 seconds of the suite for a question about existence.
_SOLVER = ScheduleOptions(
    timeout_seconds=60, workers=4, random_seed=0, stop_at_first_solution=True
)


def test_the_case_selects_every_function_the_description_defines():
    """Analyze has no reason to leave a function out, and schedule admits only
    the entry function -- so the other one is untested rather than blocked."""
    module = CASE.build()
    assert CASE.untested("analyze", module) == ()
    assert CASE.selected("schedule") == (module.entry_function().name,)
    assert CASE.untested("schedule", module) == ("mla_kv_update",)


def test_the_context_lengths_the_case_names_are_ones_the_model_has():
    """The window is what bounds the range, so the corpus's usual 1024 is not a
    context this layer type has rather than a long one -- and the length the case
    analyses at sits below the ceiling, which is asked about separately."""
    function = CASE.build().lookup("mla_attend")

    assert ANALYZED_AT["ctx_len"] < REAL.max_ctx < REAL.window
    sized = specialize_concretely(function, dict(ANALYZED_AT))
    cache = next(p for p in sized.params if p.name == "kv_cache")
    assert tuple(int(d) for d in cache.type.shape) == (
        1, ANALYZED_AT["ctx_len"], 1, REAL.head_dim,
    )

    with pytest.raises(
        DimSubstitutionError, match=r"declared over \[0, 128\) and cannot take 1024"
    ):
        specialize_concretely(function, {"ctx_len": 1024})


@pytest.mark.parametrize("family", ANALYSES.selectors_for(type(ACCEPTANCE().target)))
def test_every_analysis_family_answers_for_every_selected_function(family):
    """Each family the acceptance target registers, on each function the case
    selects, at the extents the case states."""
    fixture = ACCEPTANCE()
    for case in CASE.analyze:
        selected, function = CASE.resolve(CASE.build_for(fixture), case.selector)
        analyze(
            selected,
            function,
            analysis=family,
            dims=None if case.dims is None else dict(case.dims),
        )


def test_the_model_can_be_asked_at_a_context_length_of_our_choosing():
    """The `sized` question: a length the caller picks, not the one the kernel
    was authored at."""
    fixture = ACCEPTANCE()
    for case in CASE.sized:
        selected, function = CASE.resolve(CASE.build_for(fixture), case.selector)
        analyze(
            selected,
            function,
            analysis="compute-cost",
            dims=dict(case.dims),
        )


def test_the_partition_plans_the_entry_function_and_the_plan_holds():
    """A plan that exists is not a plan that holds, so it is verified against the
    function it was solved for -- which at a chosen size is the one the result
    carries, not the one that still holds a range."""
    fixture = ACCEPTANCE()
    for case in CASE.schedule:
        selected, function = CASE.resolve(CASE.build_for(fixture), case.selector)
        result = schedule(
            selected,
            function,
            topology=case.topology,
            options=_SOLVER,
            dims=dict(case.dims),
        )
        result.plan.verify(selected, result.function, fixture.level(case.topology))
        assert result.plan.to_json() == result.plan.to_json()
