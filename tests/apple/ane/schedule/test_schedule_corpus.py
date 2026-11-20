"""ANE HIR offered to the candidate report.

A fixture under ``hir/`` is checked through the candidate report: each labelled call
lists the ANE instructions that accept it, or the reason each one refuses it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import Case, Problems, case, cases
from tilefoundry import analysis
from tilefoundry.interfaces import schedule
from tilefoundry.ir import core, expr
from tilefoundry.schedule import candidates
from tilefoundry_apple.ane.ir import tir

ROOT = Path(__file__).parents[1] / "fixtures"


@dataclass(frozen=True)
class AneCase(Case):
    """One ANE fixture."""


@dataclass(frozen=True)
class CandidateState:
    module: object
    entry: object
    candidates: dict


def _rows(state, label):
    return [row for row in state.candidates["lines"] if row["line"].split(":")[0] == label]


def _row(state, label):
    rows = _rows(state, label)
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


def _attributes(state, label, instruction):
    """The attributes the HIR match gives *instruction* for the call labelled *label*."""
    (call,) = (
        node for node in core.collect_exprs(state.entry.body)
        if isinstance(node, expr.Call) and (expr.value_label(node) or "").split(":")[0] == label
    )
    attributes, _ = schedule.candidate_operands(call, instruction)
    return attributes


def _no_ane(state, label):
    """No ``T.ane.*`` instruction lists *label*, whether its row refuses them all or is absent.

    A call whose HIR op has no instruction at all has no row.
    """
    for row in _rows(state, label):
        assert all(not fit["id"].startswith("T.ane.") for fit in row["candidates"]), row


def _hir(case, problems):
    module = problems.run("load", case.load)
    checked = problems.run("check", lambda m: analysis.check_program(m, m.entry_function()), module)
    report = problems.run("candidates", lambda pair: candidates(*pair), checked)
    if report is not None:
        problems.run("guard", case.guard, CandidateState(*checked, report), problems)


@case
class ConvCandidates(AneCase):
    path: Path = ROOT / "hir" / "conv.py"

    def guard(self, state, problems):
        _requires(state, "same", "T.ane.conv", stride=(1, 1))
        _requires(state, "strided", "T.ane.conv", stride=(2, 2))
        _requires(state, "grouped", "T.ane.conv", stride=(1, 1), groups=2)
        _rejects(state, "single", "T.ane.conv", "no applicable HIR declaration")
        _rejects(state, "dilated", "T.ane.conv", "dilation")
        _rejects(state, "wide_step", "T.ane.conv", "stride")
        _rejects(state, "wide_pad", "T.ane.conv", "padding")


@case
class MatMulCandidates(AneCase):
    path: Path = ROOT / "hir" / "matmul.py"

    def guard(self, state, problems):
        _requires(state, "product", "T.ane.conv")
        _rejects(state, "rows", "T.ane.conv", "layout")
        _rejects(state, "single", "T.ane.conv", "no applicable HIR declaration")


@case
class ActivationCandidates(AneCase):
    path: Path = ROOT / "hir" / "activation.py"

    def guard(self, state, problems):
        _requires(state, "rectified", "T.ane.bypass", nonlinear="RELU")
        _no_ane(state, "squashed")
        _rejects(state, "single", "T.ane.bypass", "no applicable HIR declaration")


@case
class PoolCandidates(AneCase):
    path: Path = ROOT / "hir" / "pool.py"

    def guard(self, state, problems):
        for label in ("maximum", "average", "padded", "ceiled", "ceiled_valid"):
            _requires(state, label, "T.ane.pool")
        _rejects(state, "wide_step", "T.ane.pool", "stride")
        _rejects(state, "ceiled_average", "T.ane.pool", "no applicable HIR declaration")


@case
class UnaryCandidates(AneCase):
    """The report path omits the attributes a match fixes, so they are read from the pairing."""

    path: Path = ROOT / "hir" / "unary.py"

    def guard(self, state, problems):
        for label, fixed in (("absolute", {"abs": True}), ("squared", {"square": True}), ("negated", {"scale": -1.0})):
            _requires(state, label, "T.ane.unary")
            assert _attributes(state, label, tir.Unary) == fixed, label
        _no_ane(state, "rounded")


@case
class BinaryCandidates(AneCase):
    path: Path = ROOT / "hir" / "binary.py"

    def guard(self, state, problems):
        for label, kind in (
            ("added", "ADD"), ("subtracted", "SUB"), ("multiplied", "MUL"), ("smaller", "MIN"), ("larger", "MAX"),
        ):
            _requires(state, label, "T.ane.binary", kind=kind)
        _rejects(state, "divided", "T.ane.binary", "kind")
        _rejects(state, "broadcast", "T.ane.binary", "shape")
        _rejects(state, "single", "T.ane.binary", "no applicable HIR declaration")


@case
class ReduceCandidates(AneCase):
    path: Path = ROOT / "hir" / "reduce.py"

    def guard(self, state, problems):
        for label in ("total", "largest", "smallest", "width", "flat"):
            _requires(state, label, "T.ane.reduce")
        _rejects(state, "mean", "T.ane.reduce", "kind")
        _rejects(state, "channels", "T.ane.reduce", "axes")
        _rejects(state, "height", "T.ane.reduce", "axes")


@case
class TransposeCandidates(AneCase):
    path: Path = ROOT / "hir" / "transpose.py"

    def guard(self, state, problems):
        _requires(state, "swapped", "T.ane.bypass", perm=(2, 1, 0))
        _requires(state, "rotated", "T.ane.bypass", perm=(1, 2, 0))
        _rejects(state, "batch", "T.ane.bypass", "perm")


CASES = cases(ROOT / "hir", case_type=AneCase)
STAGES = {"hir": _hir}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_schedule_case(case: AneCase) -> None:
    problems = Problems(case)
    STAGES[case.path.parent.name](case, problems)
    problems.dump()
