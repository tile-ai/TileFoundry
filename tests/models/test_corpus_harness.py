"""The properties every model-driven test relies on the corpus to hold."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.models.corpus import (
    MODELS_ROOT,
    CapabilityGate,
    CorpusError,
    FunctionCase,
    ModelCase,
    TargetFixture,
)
from tests.models.fixtures import apple_m2_pro, h200_sxm
from tests.models.qwen3_1_7b.case import CASE as QWEN3_1_7B
from tests.models.registry import CORPUS, case
from tests.models.report import CoverageCollector, build_report, render_report
from tilefoundry.analysis import analyze
from tilefoundry.analysis.facts import ParallelCapacityFacts
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.module import Module, select
from tilefoundry.ir.types.shard import Topology


def _source(case: ModelCase) -> Module:
    """The authored Module a case is scoped to, reached the way the case reaches
    it. What a build must share nothing with."""
    return select(case.prototype, case.scope)


def _analysis_records(function) -> int:
    """How many analysis records hang off the calls in *function*'s body."""
    authored = {"SourceSpanMetadata", "BindingMetadata"}
    total = 0

    def walk(expr, depth: int = 0) -> None:
        nonlocal total
        if expr is None or depth > 64:
            return
        if isinstance(expr, Call):
            total += sum(
                1
                for record in expr.metadata
                if type(record).__name__ not in authored
            )
        for attribute in ("args", "elements"):
            for child in getattr(expr, attribute, ()) or ():
                walk(child, depth + 1)
        walk(getattr(expr, "body", None), depth + 1)

    walk(function.body)
    return total


def test_a_build_shares_nothing_with_the_build_before_it() -> None:
    first = QWEN3_1_7B.build()
    second = QWEN3_1_7B.build()

    assert first is not second
    assert first.lookup("mlp") is not second.lookup("mlp")
    assert first.lookup("mlp") is not _source(QWEN3_1_7B).lookup("mlp")


def test_analysing_one_build_leaves_the_next_build_clean() -> None:
    """The property `replace` cannot give: analyses annotate Calls in place. The
    prototype stays clean too, since it outlives every build taken from it."""
    fixture = h200_sxm()
    analysed = QWEN3_1_7B.build_for(fixture)
    assert _analysis_records(analysed.lookup("mlp")) == 0

    analyze(analysed, analysed.lookup("mlp"), analysis="compute-cost")
    assert _analysis_records(analysed.lookup("mlp")) > 0

    fresh = QWEN3_1_7B.build_for(fixture)
    assert _analysis_records(fresh.lookup("mlp")) == 0
    assert _analysis_records(_source(QWEN3_1_7B).lookup("mlp")) == 0


def test_a_stack_analyses_one_layer_without_marking_its_neighbour() -> None:
    """Adjacent layers of the real stack hold different Functions, so a record
    written on one lands nowhere near the other."""
    from tests.models.qwen3_1_7b.model import Qwen3_1_7B  # noqa: PLC0415

    stack = h200_sxm().bind(Qwen3_1_7B.cloned())
    first, second = stack.modules[0], stack.modules[1]
    assert first.lookup("mlp") is not second.lookup("mlp")

    analyze(first, first.lookup("mlp"), analysis="compute-cost")

    assert _analysis_records(first.lookup("mlp")) > 0
    assert _analysis_records(second.lookup("mlp")) == 0


def test_binding_a_target_does_not_reach_the_next_build() -> None:
    fixture = h200_sxm()
    bound = QWEN3_1_7B.build_for(fixture)
    assert bound.resolve_target() is fixture.target
    assert bound.effective_topologies() == fixture.topologies

    unbound = QWEN3_1_7B.build()
    assert unbound.target is None
    assert unbound.topologies is None


def test_one_model_answers_to_more_than_one_machine_in_one_run() -> None:
    """The fixture rebinds, so the same model can be asked twice."""
    cuda = QWEN3_1_7B.build_for(h200_sxm())
    amx = QWEN3_1_7B.build_for(apple_m2_pro())

    assert cuda.resolve_target() is not amx.resolve_target()
    assert cuda.lookup("mlp") is not amx.lookup("mlp")


def test_a_case_is_aimed_by_its_fixture_whatever_its_source_declares() -> None:
    fixture = apple_m2_pro()
    for model in CORPUS:
        bound = model.build_for(fixture)
        assert bound.resolve_target() is fixture.target, model.id
        assert bound.effective_topologies() == fixture.topologies, model.id


def test_fixtures_take_their_extents_from_the_hardware_documents() -> None:
    cuda = h200_sxm()
    amx = apple_m2_pro()

    assert cuda.level("cta").size == cuda.target.as_facts(
        ParallelCapacityFacts
    ).parallel_units
    assert amx.level("core").size == amx.target.as_facts(
        ParallelCapacityFacts
    ).parallel_units
    assert cuda.level("thread").size <= cuda.target.topology_limit("thread")


def test_a_fixture_rejects_a_level_it_does_not_declare() -> None:
    with pytest.raises(CorpusError, match="declares no 'warp' level"):
        h200_sxm().level("warp")


def test_untested_functions_come_from_the_built_model_not_a_written_list() -> None:
    inventory = QWEN3_1_7B.inventory()
    assert set(QWEN3_1_7B.selected("schedule")) < set(inventory)

    untested = QWEN3_1_7B.untested("schedule")
    assert set(untested) == set(inventory) - set(QWEN3_1_7B.selected("schedule"))
    assert "input_rms_norm" in untested

    grown = replace(
        QWEN3_1_7B,
        schedule=(*QWEN3_1_7B.schedule, FunctionCase(id="x", selector="input_rms_norm")),
    )
    assert "input_rms_norm" not in grown.untested("schedule")


def test_a_blocked_capability_must_say_why() -> None:
    with pytest.raises(CorpusError, match="must state why"):
        CapabilityGate(outcome="BLOCKED")
    with pytest.raises(CorpusError, match="states no reason"):
        CapabilityGate(outcome="PASS", reason="unused")

    blocked = CapabilityGate(outcome="BLOCKED", reason="no fp8 atom on this target")
    assert blocked.blocked
    assert not CapabilityGate().blocked


def test_a_case_that_cannot_be_resolved_says_so() -> None:
    """The four ways a case can name something it cannot be measured at: a
    prototype that is not a Module, a name no Module defines, a path that stops at
    a child Module, and a path with an empty segment.

    The last two are refusals rather than conveniences. A path stopping at a child
    would resolve to whatever that child nominates as its default step, answering a
    question nobody asked; and dropping an empty segment would make two different
    paths name one node.
    """
    not_a_module = ModelCase(id="missing", prototype=_source(QWEN3_1_7B).lookup("mlp"))
    with pytest.raises(CorpusError, match="prototype is a Function, not a Module"):
        not_a_module.build()

    built = QWEN3_1_7B.build()
    with pytest.raises(CorpusError, match="must resolve to exactly one"):
        QWEN3_1_7B.resolve(built, "nope")

    nested = case("kimi_linear_48b_a3b")
    tree = nested.build()
    assert nested.resolve(tree, "moe.moe")[1].name == "moe"
    with pytest.raises(CorpusError, match="names the child Module 'moe'"):
        nested.resolve(tree, "moe")
    with pytest.raises(CorpusError, match="empty segment"):
        nested.resolve(tree, "moe..moe")


def test_the_registry_resolves_its_own_ids() -> None:
    assert case("qwen3_1_7b") is QWEN3_1_7B
    with pytest.raises(KeyError, match="no case 'nope'"):
        case("nope")


def test_the_report_groups_model_then_target_then_kind() -> None:
    collector = CoverageCollector()
    fixture = h200_sxm()
    collector.record(
        model=QWEN3_1_7B.id,
        target=fixture.id,
        kind="reference",
        case="qwen3_1_7b/reference/decoder",
        status="PASS",
    )
    collector.record(
        model=QWEN3_1_7B.id,
        target=fixture.id,
        kind="analyze",
        case="qwen3_1_7b/analyze/mlp",
        function="mlp",
        status="PASS",
    )
    collector.record(
        model=QWEN3_1_7B.id,
        target=fixture.id,
        kind="schedule",
        case="qwen3_1_7b/schedule/self_attention",
        function="self_attention",
        status="BLOCKED",
        reason="no atom covers this operation yet",
    )

    report = build_report(collector, CORPUS)
    section = report["qwen3_1_7b"]["targets"]["h200_sxm"]

    assert [row["case"] for row in section["reference"]] == [
        "qwen3_1_7b/reference/decoder"
    ]
    assert [row["function"] for row in section["analyze"]["tested"]] == ["mlp"]
    assert "mlp" not in section["analyze"]["untested"]
    assert "decoder_layer" in section["analyze"]["untested"]
    assert section["schedule"]["tested"][0]["status"] == "BLOCKED"
    assert section["schedule"]["tested"][0]["reason"]

    text = render_report(report)
    assert "qwen3_1_7b" in text
    assert "h200_sxm" in text
    assert "untested" in text


def test_an_unrun_function_is_untested_and_never_blocked() -> None:
    """Nobody selected it, so the report must not claim a limit stopped it."""
    collector = CoverageCollector()
    report = build_report(collector, CORPUS)
    assert report["qwen3_1_7b"]["targets"] == {}

    collector.record(
        model=QWEN3_1_7B.id,
        target="h200_sxm",
        kind="analyze",
        case="qwen3_1_7b/analyze/mlp",
        function="mlp",
        status="PASS",
    )
    section = build_report(collector, CORPUS)["qwen3_1_7b"]["targets"]["h200_sxm"]
    statuses = {row["status"] for row in section["analyze"]["tested"]}
    assert statuses == {"PASS"}
    assert "self_attention" in section["analyze"]["untested"]


def test_a_result_nobody_can_act_on_is_rejected() -> None:
    collector = CoverageCollector()
    with pytest.raises(ValueError, match="without a reason"):
        collector.record(
            model="m", target="t", kind="analyze", case="c", status="FAIL"
        )


def test_a_target_fixture_binds_only_what_it_was_given() -> None:
    built = QWEN3_1_7B.build()
    fixture = TargetFixture(
        id="probe",
        target=h200_sxm().target,
        topologies=(Topology("cta", 4),),
    )
    declared = built.topologies
    bound = fixture.bind(built)

    assert isinstance(bound, Module)
    assert bound.effective_topologies() == (Topology("cta", 4),)
    assert built.topologies == declared


def test_a_blocked_case_that_starts_working_breaks_the_build() -> None:
    """The direction that matters. A skip rots quietly; a block that begins
    passing has to fail here, because that is the only thing that makes anyone
    correct the matrix."""
    gate = CapabilityGate(outcome="BLOCKED", reason="no fp8 atom on this target")

    with pytest.raises(CorpusError, match="succeeded; the capability matrix"):
        gate.hold(lambda: None, expect=ValueError, label="case/x")


def test_a_blocked_case_that_fails_as_stated_re_raises_that_failure() -> None:
    """The runner records the expectation, so the failure has to reach it.
    Swallowing it here would report the case as a plain pass."""
    gate = CapabilityGate(outcome="BLOCKED", reason="no fp8 atom on this target")

    def fail() -> None:
        raise ValueError("no fp8 atom on this target, so nothing covers it")

    with pytest.raises(ValueError, match="no fp8 atom"):
        gate.hold(fail, expect=ValueError, label="case/x")


def test_a_blocked_gate_marks_its_case_as_a_strict_expected_failure() -> None:
    """`strict` so an unexpected success fails; `raises` so only the stated
    kind of failure is absorbed and a wrong-reason CorpusError stays a
    failure rather than passing for the expectation."""
    gate = CapabilityGate(outcome="BLOCKED", reason="no fp8 atom on this target")

    (mark,) = gate.expected_failure(expect=ValueError)

    assert mark.kwargs["strict"] is True
    assert mark.kwargs["raises"] is ValueError
    assert mark.kwargs["reason"] == gate.reason


def test_a_passing_gate_marks_nothing() -> None:
    assert CapabilityGate().expected_failure(expect=ValueError) == ()


def test_a_blocked_case_that_fails_differently_is_not_that_block() -> None:
    """A case that breaks for another reason is not the limit anybody signed
    off on, and recording it as one hides a second defect behind the first."""
    gate = CapabilityGate(outcome="BLOCKED", reason="no fp8 atom on this target")

    def fail() -> None:
        raise ValueError("the module has no entry function")

    with pytest.raises(CorpusError, match="but it failed with"):
        gate.hold(fail, expect=ValueError, label="case/x")


def test_a_passing_case_reports_its_own_failure_unchanged() -> None:
    """Nothing swallows a failure the matrix did not predict."""
    def fail() -> None:
        raise ValueError("cost evaluation blew up")

    with pytest.raises(ValueError, match="cost evaluation blew up"):
        CapabilityGate().hold(fail, expect=ValueError, label="case/x")


def test_the_reference_entry_is_wired_to_the_function_it_names() -> None:
    """M2 runs the oracle; M1 makes sure there is one and that it fits.

    The arity check is the part that rots otherwise: a parameter added to the
    model would leave the reference silently calling the wrong shape until
    somebody ran it.

    The requirement is per *model*, not per Module: a model described by several
    Modules must be held to an oracle somewhere, and a Module whose own boundary is
    measured in its package's tests rather than here says so at its case. Requiring
    one on every Module would push a boundary into the harness for the sake of the
    count, which is how a harness ends up running an oracle nobody wanted."""
    declared = {model.model for model in CORPUS if model.reference is not None}
    missing = {model.model for model in CORPUS} - declared
    assert not missing, f"these models declare no reference at all: {sorted(missing)}"

    for model in CORPUS:
        reference = model.reference
        if reference is None:
            continue
        assert reference.boundary.strip(), f"{reference.id} states no boundary"
        assert callable(reference.oracle)

        if reference.entry is None:
            # A boundary that is not one Function says how it runs instead, and
            # there is no single signature to measure the drawn arguments against.
            assert callable(reference.runner), (
                f"{reference.id} names no entry function and no runner"
            )
            continue

        assert reference.entry in model.inventory()
        built = model.build()
        entry = built.lookup(reference.entry)
        drawn = reference.inputs()
        assert len(drawn.args) == len(entry.params), (
            f"{reference.id} draws {len(drawn.args)} arguments for "
            f"{reference.entry!r}, which takes {len(entry.params)}"
        )


_BLOCKED_CASE = '''
import pytest
from tests.models.corpus import CapabilityGate

GATE = CapabilityGate(outcome="BLOCKED", reason="no fp8 atom on this target")


@pytest.mark.parametrize(
    "run",
    [pytest.param({run}, marks=GATE.expected_failure(expect=ValueError))],
)
def test_case(run):
    GATE.hold(run, expect=ValueError, label="case/x")
'''


def _outcome_of(tmp_path, run: str) -> str:
    """Run one generated case under a real pytest and report how it came out."""
    case = tmp_path / "test_generated_case.py"
    case.write_text(_BLOCKED_CASE.format(run=run), encoding="utf-8")
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", str(case), "-q", "-p", "no:randomly"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    return completed.stdout


def test_a_blocked_case_is_reported_as_an_expected_failure(tmp_path) -> None:
    """Not merely "the test passed". The runner has to say xfail, or nobody
    reading the output can tell a known limit from working code."""
    output = _outcome_of(tmp_path, 'lambda: (_ for _ in ()).throw(ValueError("no fp8 atom on this target"))')

    assert "1 xfailed" in output, output
    assert "passed" not in output, output


def test_a_blocked_case_that_succeeds_fails_the_run(tmp_path) -> None:
    """The strict half: an unexpected success is not an unexpected pass."""
    output = _outcome_of(tmp_path, "lambda: None")

    assert "1 failed" in output, output
    assert "xpassed" not in output, output


def test_being_listed_is_what_puts_a_model_in_the_corpus() -> None:
    """Every named package states a case, and every case names its own package.

    The list is written out rather than read off the filesystem, so this checks
    the two ways that can go wrong: a package named and not ready, and a case
    whose model points at something a reader cannot go and open.

    A package may state several cases, one per Module it selects from, so what has
    to match the list is the set of models the cases name -- in order, because the
    order the report is read in is the order the list is written in.
    """
    from tests.models.registry import MODELS  # noqa: PLC0415

    assert MODELS, "the corpus names no models"
    assert list(dict.fromkeys(model.model for model in CORPUS)) == list(MODELS)
    for model in CORPUS:
        assert (MODELS_ROOT / model.model).is_dir(), (
            f"{model.model} is in the corpus and has no package"
        )


def test_a_package_that_states_no_case_is_refused() -> None:
    """Silence is not emptiness.

    A listed package contributing nothing would read as a model with nothing to
    select, which is a different fact from a model that is not there yet.
    """
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    from tests.models import registry  # noqa: PLC0415

    name = "tests.models._nothing_here.case"
    sys.modules[name] = types.ModuleType(name)
    try:
        with pytest.raises(TypeError, match="must state CASES"):
            registry._cases("_nothing_here")
    finally:
        del sys.modules[name]


def test_an_ungated_case_hands_back_what_it_ran() -> None:
    """A gate that ran a computation and dropped its value would gate only
    whether it raised, and every caller judging the value would judge nothing."""
    assert CapabilityGate().hold(lambda: 7, expect=ValueError, label="case/x") == 7
