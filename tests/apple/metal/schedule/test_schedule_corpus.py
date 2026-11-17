"""Authored Metal HIR checked through the public candidate report, without execution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import Case, Problems, case, cases
from tilefoundry import analysis
from tilefoundry.schedule import candidates

ROOT = Path(__file__).parents[1] / "fixtures"


@dataclass(frozen=True)
class CandidateState:
    module: object
    entry: object
    candidates: dict


@dataclass(frozen=True)
class MetalCase(Case):
    """One authored Metal HIR fixture and the candidate report it should give."""


def _row(state, label):
    rows = [row for row in state.candidates["lines"] if row["line"].split(":")[0] == label]
    assert len(rows) == 1, (label, state.candidates)
    return rows[0]


def _bindings(fit):
    return dict((p["name"], p["value"]) for p in fit["path"])


def _requires(state, label, instruction, **bindings):
    row = _row(state, label)
    fits = [fit for fit in row["candidates"] if fit["id"] == instruction]
    assert fits, (label, instruction, row)
    assert any(
        all(str(value) == _bindings(fit).get(key) for key, value in bindings.items()) for fit in fits
    ), (label, instruction, bindings, fits)


def _rejects(state, label, instruction, reason):
    row = _row(state, label)
    assert all(fit["id"] != instruction for fit in row["candidates"]), row
    assert any(fit["id"] == instruction and reason in fit["reason"] for fit in row["refused"]), row


@case
class MatMul(MetalCase):
    path: Path = ROOT / "hir_matmul.py"

    def guard(self, state, problems):
        _requires(state, "single", "T.metal.matmul", dtype="f32", groups=2)
        _rejects(state, "integer", "T.metal.matmul", "dtype")
        _rejects(state, "memory", "T.metal.matmul", "no applicable HIR declaration")


@case
class Memory(MetalCase):
    path: Path = ROOT / "hir_memory.py"

    def guard(self, state, problems):
        _requires(state, "load", "T.metal.ldmatrix", dtype="f32", groups=2, row_stride=8)
        _requires(state, "store", "T.metal.stmatrix", dtype="f32", groups=2, row_stride=8)
        _rejects(state, "load", "T.metal.stmatrix", "no applicable HIR declaration")
        _rejects(state, "store", "T.metal.ldmatrix", "no applicable HIR declaration")


@case
class Reduce(MetalCase):
    path: Path = ROOT / "hir_reduce.py"

    def guard(self, state, problems):
        _requires(state, "total", "T.metal.reduce", kind="SUM", dtype="f32", groups=2, width=32)
        _rejects(state, "brain_total", "T.metal.reduce", "dtype")


CASES = cases(ROOT, case_type=MetalCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_schedule_case(case: MetalCase) -> None:
    problems = Problems(case)
    module = problems.run("load", case.load)
    checked = problems.run("check", lambda m: analysis.check_program(m, m.entry_function()), module)
    report = problems.run("candidates", lambda pair: candidates(*pair), checked)
    if report is not None:
        problems.run("guard", case.guard, CandidateState(*checked, report), problems)
    problems.dump()
