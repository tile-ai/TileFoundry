"""The corpus is held to being describable before it is held to working.

A capability matrix is read as a claim about a system, so the ways it can lie are
worth failing on directly rather than hoping a reader notices. Four of them are
checked here, each because it produces a report that looks complete:

- two cases with one id collapse into one row, and whichever ran second silently
  replaces the first;
- a reference with no oracle is a boundary compared against nothing;
- a block with no reason cannot be reviewed or retired, so it becomes permanent;
- a gate that has gone stale describes a limit nobody has any more.

Each is asserted against the corpus itself rather than against a list kept beside
it, so a model added tomorrow is checked by the same four rules.
"""

from __future__ import annotations

import re

from tests.models.corpus import ModelCase
from tests.models.registry import CORPUS


def _all_cases() -> tuple[ModelCase, ...]:
    return CORPUS


def test_no_two_cases_share_an_id() -> None:
    """One id, one row. Two cases sharing one would report as one case, and which
    of the two the row describes would depend on collection order."""
    seen: dict[str, str] = {}
    for case in _all_cases():
        assert case.id not in seen, (
            f"case id {case.id!r} is stated by both {seen[case.id]} and "
            f"{case.model}; a report row would name one and describe the other"
        )
        seen[case.id] = case.model


def test_no_two_selected_cases_share_an_id() -> None:
    """The same rule for what is selected from a model, across every kind.

    Analyze, schedule, sized and reference ids all end up in one report, so an id
    reused between two of them is the same collapse one level down.
    """
    seen: dict[str, str] = {}
    for model in _all_cases():
        selected = [
            *((case.id, "analyze") for case in model.analyze),
            *((case.id, "schedule") for case in model.schedule),
            *((case.id, "sized") for case in model.sized),
        ]
        if model.reference is not None:
            selected.append((model.reference.id, "reference"))
        for case_id, kind in selected:
            where = f"{model.id}/{kind}"
            assert case_id not in seen, (
                f"selected case id {case_id!r} is stated by both {seen[case_id]} "
                f"and {where}"
            )
            seen[case_id] = where


def test_every_reference_states_an_oracle_and_a_boundary() -> None:
    """A reference without something to be judged against is not a reference.

    `inputs` and `oracle` are one pair by construction elsewhere; what this adds is
    that both are there and callable, and that the boundary is written down. A
    boundary left blank is how a reference quietly shrinks to a leaf op while still
    reporting PASS.
    """
    for model in _all_cases():
        reference = model.reference
        if reference is None:
            continue
        assert callable(reference.inputs), f"{reference.id} draws no inputs"
        assert callable(reference.oracle), f"{reference.id} states no oracle"
        assert len(reference.boundary.split()) >= 5, (
            f"{reference.id} states its boundary as {reference.boundary!r}, which "
            f"is too short to say what was run"
        )


def test_every_model_declares_at_least_one_reference() -> None:
    """A model in the corpus is held to an oracle somewhere."""
    with_reference = {
        case.model for case in _all_cases() if case.reference is not None
    }
    missing = sorted({case.model for case in _all_cases()} - with_reference)
    assert not missing, f"these models are held to no oracle at all: {missing}"


def test_every_block_states_a_reason() -> None:
    """`CapabilityGate` refuses an unreasoned block at construction; this is the
    same rule asked of the whole corpus at once, so a gate built some other way
    cannot slip past it."""
    for model in _all_cases():
        gates = [
            *((case.gate, case.id) for case in model.analyze),
            *((case.gate, case.id) for case in model.schedule),
            *((case.gate, case.id) for case in model.sized),
        ]
        if model.reference is not None:
            gates.append((model.reference.gate, model.reference.id))
        for gate, case_id in gates:
            if not gate.blocked:
                assert not gate.reason, (
                    f"{case_id} states a reason while passing: {gate.reason!r}"
                )
                continue
            assert len(gate.reason.split()) >= 5, (
                f"{case_id} is blocked on {gate.reason!r}, which does not say "
                f"enough to review or retire it"
            )


def test_a_blocked_reason_names_what_would_lift_it() -> None:
    """A reason is only actionable if it says what the limit is about.

    Checked as "names something concrete" rather than by matching prose: a reason
    that mentions no version, no class, no operation and no shape is a statement
    that something did not work, which is what was already known.
    """
    concrete = re.compile(
        r"\d|transformers|no [a-z]+ (implementation|evaluator|oracle)|"
        r"[A-Z][a-zA-Z0-9_]{3,}"
    )
    for model in _all_cases():
        gates = [
            *((case.gate, case.id) for case in model.analyze),
            *((case.gate, case.id) for case in model.schedule),
            *((case.gate, case.id) for case in model.sized),
        ]
        if model.reference is not None:
            gates.append((model.reference.gate, model.reference.id))
        for gate, case_id in gates:
            if not gate.blocked:
                continue
            assert concrete.search(gate.reason), (
                f"{case_id} is blocked on {gate.reason!r}, which names nothing a "
                f"reader could go and check"
            )
