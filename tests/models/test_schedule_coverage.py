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
from tests.models.fixtures import ACCEPTANCE
from tests.models.registry import CORPUS
from tests.models.report import CoverageCollector, build_report
from tilefoundry.schedule import ScheduleError, ScheduleOptions, schedule

#: One case is one CP-SAT solve, so the budget is stated rather than inherited.
_SOLVER = ScheduleOptions(timeout_seconds=60, workers=4, random_seed=0)


def _cases() -> list[tuple[ModelCase, TargetFixture, FunctionCase]]:
    fixture = ACCEPTANCE()
    return [(model, fixture, case) for model in CORPUS for case in model.schedule]


def _identify(item: object) -> str:
    if isinstance(item, ModelCase | TargetFixture):
        return item.id
    if isinstance(item, FunctionCase):
        return item.function
    return str(item)


@pytest.mark.parametrize(("model", "fixture", "case"), _cases(), ids=_identify)
def test_every_selected_function_plans_or_says_what_stopped_it(
    model: ModelCase, fixture: TargetFixture, case: FunctionCase
) -> None:
    module = model.build_for(fixture)
    function = model.function(module, case)
    topology = fixture.level(case.topology)

    if case.gate.blocked:
        with pytest.raises(ScheduleError) as raised:
            schedule(module, function, topology=topology.name, options=_SOLVER)
        assert case.gate.reason in str(raised.value)
        return

    result = schedule(module, function, topology=topology.name, options=_SOLVER)
    result.plan.verify(module, function, topology)
    assert result.plan.to_json() == result.plan.to_json()


def test_the_functions_no_partition_can_take_are_untested_not_blocked() -> None:
    """The algorithm admits one function per module, so the others were never
    selected. Reporting them as blocked would claim they were tried."""
    for model in CORPUS:
        module = model.build()
        entry = module.entry_function().name
        assert model.selected("schedule") == (entry,)
        assert entry not in model.untested("schedule")
        assert model.untested("schedule")


def test_the_report_separates_what_ran_from_what_nobody_selected() -> None:
    fixture = ACCEPTANCE()
    collector = CoverageCollector()
    for model, _, case in _cases():
        collector.record_gate(
            case.gate,
            model=model.id,
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
