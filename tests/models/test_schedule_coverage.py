"""Every function the corpus selects for scheduling, on the machine it targets.

Fewer functions appear here than in analysis, and that is a property of the
algorithm rather than an omission: the device-wide partition decides the whole
launch, so it admits only the module entry function. The report still lists the
rest as untested, which is the truth -- nobody selected them.

A plan that exists is not a plan that holds. Each case verifies its own plan
against the module and topology it was solved for, because a solver that
returned a structurally invalid plan would otherwise pass for having returned
something.
"""

from __future__ import annotations

import pytest

from tests.models.corpus import FunctionCase, ModelCase, TargetFixture
from tests.models.coverage_artifact import declare
from tests.models.fixtures import ACCEPTANCE
from tests.models.registry import CORPUS
from tests.models.report import CoverageCollector, build_report
from tilefoundry.schedule import ScheduleError, ScheduleOptions, schedule

#: One case is one CP-SAT solve, so the budget is stated rather than inherited.
#:
#: `stop_at_first_solution`, because what these cases assert is that a plan exists
#: and that it verifies -- not that it is the best plan. Without it the search keeps
#: improving a makespan it cannot prove optimal until the limit, so every solve costs
#: the whole budget: measured, twenty-eight of these each spent its full sixty
#: seconds. The timeout stays as the bound on a search that has found nothing yet.
_SOLVER = ScheduleOptions(
    timeout_seconds=60, workers=4, random_seed=0, stop_at_first_solution=True
)


def _selected() -> list[tuple[ModelCase, TargetFixture, FunctionCase]]:
    fixture = ACCEPTANCE()
    return [(model, fixture, case) for model in CORPUS for case in model.schedule]


def _cases() -> list[object]:
    """Each case carries its gate as its own expected result."""
    return [
        pytest.param(
            model,
            fixture,
            case,
            id=case.function,
            marks=case.gate.expected_failure(expect=ScheduleError),
        )
        for model, fixture, case in _selected()
    ]


@pytest.mark.parametrize(("model", "fixture", "case"), _cases())
def test_every_selected_function_plans_or_says_what_stopped_it(
    model: ModelCase, fixture: TargetFixture, case: FunctionCase, record_property
) -> None:
    declare(
        record_property,
        model=model.model,
        target=fixture.id,
        kind="schedule",
        case=case.id,
        function=case.function,
    )
    module = model.build_for(fixture)
    function = model.function(module, case)
    topology = fixture.level(case.topology)

    def run() -> None:
        result = schedule(
            module,
            function,
            topology=topology.name,
            options=_SOLVER,
            dims=None if case.dims is None else dict(case.dims),
        )
        # The plan is verified against the function it was made for, which at a
        # chosen size is the one the result carries rather than the one asked
        # about: verifying against a function still holding a range would check
        # the plan against a program nothing planned.
        result.plan.verify(module, result.function, topology)
        assert result.plan.to_json() == result.plan.to_json()

    case.gate.hold(run, expect=ScheduleError, label=case.id)


def test_the_functions_no_partition_can_take_are_untested_not_blocked() -> None:
    """The algorithm admits one function per module, so the others were never
    selected. Reporting them as blocked would claim they were tried.

    Counted rather than merely non-empty: every function of the Module except its
    entry has to appear, so a Module whose leaves stopped being reported fails here
    and a Module whose only function is its entry -- which has no leaves to report
    -- is not made to invent one.
    """
    for model in CORPUS:
        module = model.build()
        entry = module.entry_function().name
        untested = model.untested("schedule", module)
        assert model.selected("schedule") == (entry,)
        assert entry not in untested
        assert len(untested) == len(model.inventory(module)) - 1


def test_the_report_separates_what_ran_from_what_nobody_selected() -> None:
    fixture = ACCEPTANCE()
    collector = CoverageCollector()
    for model, _, case in _selected():
        collector.record_gate(
            case.gate,
            model=model.model,
            target=fixture.id,
            kind="schedule",
            case=case.id,
            function=case.function,
        )

    section = build_report(collector, CORPUS)["qwen3_1_7b"]["targets"][fixture.id]
    assert [row["function"] for row in section["schedule"]["tested"]] == [
        "decoder_layer"
    ]
    assert "mlp" in section["schedule"]["untested"]
    assert "decoder_layer" not in section["schedule"]["untested"]
