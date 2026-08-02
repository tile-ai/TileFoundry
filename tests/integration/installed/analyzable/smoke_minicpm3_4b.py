"""MiniCPM3-4B, as the installation ships it, asked through the commands."""
from __future__ import annotations

import json

import contract
import pytest

from tests.models.minicpm3_4b import reference

MODEL = "minicpm3_4b"
CASES = contract.model_cases(MODEL)

ANALYSED = [
    pytest.param(case, selected, family, id=f"{selected.id}/{family}")
    for case in CASES
    for selected in case.analyze
    for family in contract.FAMILIES
]
PLANNED = [
    pytest.param(case, planned, id=planned.id)
    for case in CASES
    for planned in case.schedule
]
#: One case per Module, as the levels a root declares are a property of the root.
FIRST_PLAN = [pytest.param(case, case.schedule[0], id=case.id) for case in CASES]
SIZED = [pytest.param(case, sized, id=sized.id) for case in CASES for sized in case.sized]

#: The bindings whose cost is the context, so a zero context has to zero them.
ZERO_SIZED = frozenset(("k_cache", "v_cache"))


@pytest.mark.parametrize(("case", "selected", "family"), ANALYSED)
def test_every_selected_function_analyses(tf, shipped_source, case, selected, family) -> None:
    contract.analysed(
        tf, shipped_source(MODEL), case, selected.selector, family, selected.dims
    )


@pytest.mark.parametrize(("case", "planned"), PLANNED)
def test_every_selected_function_plans(tf, shipped_source, case, planned) -> None:
    contract.scheduled(tf, shipped_source(MODEL), case, planned)


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_each_model_is_asked_at_a_size(tf, shipped_source, case, sized) -> None:
    contract.analysed(
        tf, shipped_source(MODEL), case, sized.selector, "compute-cost", sized.dims
    )


@pytest.mark.parametrize(("case", "sized"), SIZED)
@pytest.mark.parametrize("family", contract.FAMILIES)
def test_every_analysis_answers_at_the_largest_context(
    tf, shipped_source, case, sized, family
) -> None:
    """At the ceiling the case states, not at a sample of it."""
    contract.analysed(
        tf, shipped_source(MODEL), case, sized.selector, family, sized.ceiling
    )


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_the_ceiling_is_reasoned_about_at_its_stated_length(
    tf, shipped_source, case, sized
) -> None:
    """What the analysis reports has to grow with the context.

    Growth rather than an absolute number: an analysis quietly working at a length
    it could afford instead of the one it was asked about would report the same
    footprint at both, and a number nobody compares would not show it.
    """
    source = shipped_source(MODEL)
    short = contract.traffic_read(tf, source, case, sized.selector, sized.dims)
    full = contract.traffic_read(tf, source, case, sized.selector, sized.ceiling)

    assert full > short, (
        f"analysing at {dict(sized.ceiling)} reports no more traffic than at "
        f"{dict(sized.dims)}, so the stated length changed nothing"
    )


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_the_open_dimensions_are_analysed_at_zero(tf, shipped_source, case, sized) -> None:
    """A binding whose whole cost is the context has to cost nothing without one."""
    source = shipped_source(MODEL)
    bindings = ZERO_SIZED
    zero = contract.lifetimes(
        tf, source, case, sized.selector, {name: 0 for name in sized.dims}
    )
    nonzero = contract.lifetimes(tf, source, case, sized.selector, sized.dims)

    assert bindings <= zero.keys()
    assert all(zero[binding] == 0 for binding in bindings)
    assert all(nonzero[binding] > 0 for binding in bindings)


@pytest.mark.parametrize(("case", "planned"), FIRST_PLAN)
def test_the_command_reports_a_real_model_as_json(tf, shipped_source, case, planned) -> None:
    done = contract.analysed(
        tf,
        shipped_source(MODEL),
        case,
        planned.selector,
        "compute-cost",
        planned.dims,
        json_output=True,
    )

    assert json.loads(done.stdout)


@pytest.mark.parametrize(("case", "planned"), FIRST_PLAN)
def test_the_command_reads_the_machine_off_the_shipped_source(
    tf, shipped_source, case, planned
) -> None:
    """Nothing tells the command which target to use; the source has to say."""
    done = contract.capabilities(tf, shipped_source(MODEL), case, planned.selector)

    assert done.stdout.strip()

# ── against Hugging Face ─────────────────────────────────────────────────────
def test_the_decode_step_and_the_cache_entry_it_hands_back(
    tf, shipped_source, tmp_path
) -> None:
    """One decode step of one layer, and the state the step hands back.

    The boundary: the whole layer, whose MLA cache design this is the claim for --
    the entry is the assembled per-head key and the up-projected value, which is what
    Hugging Face's own cache holds.

    The returned key and value are this token's cache entry: they are compared
    against a cache rebuilt over a context one token longer, not against the step's
    own inputs, so a step that returned its inputs unchanged fails. The cache handed
    in is the oracle's own, so the appended entry is the only computed part and the
    one whose precision the bound follows.

    The command requires a predicate for every output a function returns, so the
    step's own output is judged here too rather than discarded.
    """
    drawn = reference.decode_step_inputs(device="cpu")
    source, case = shipped_source(MODEL), CASES[0]
    want_out = reference.decode_step_oracle(drawn)
    want_k, want_v = reference.appended_cache_oracle(drawn)
    entry_k, entry_v = want_k[:, drawn.ctx_len:], want_v[:, drawn.ctx_len:]

    contract.compared(
        tf, tmp_path, source, case, "decoder_layer",
        activations=drawn.args,
        weights=drawn.loaded.constants,
        expected=(want_out, entry_k, entry_v),
        held=(
            contract.one_rounding(want_out),
            contract.one_rounding(entry_k),
            contract.one_rounding(entry_v),
        ),
        dims={"ctx_len": drawn.ctx_len},
    )

    assert want_k.shape[1] == drawn.ctx_len + 1
    assert entry_k.shape[1] == 1 and entry_v.shape[1] == 1
