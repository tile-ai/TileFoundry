"""What actually ran, grouped the way a reader asks the question.

A reader asks "does this model work on this machine", so the report is grouped
`model -> target -> reference / analyze / schedule` and never as a flat list of
subsystem results. Assembling per-subsystem passes into a claim that a model
works is exactly the inversion this corpus exists to undo.

Two distinctions the report keeps apart, because collapsing them is how coverage
gets overstated:

- **untested** is a function nobody selected. It is derived from the built
  Module's own inventory minus what the registry selected, so a function added to
  a model shows up here instead of vanishing.
- **BLOCKED** is a case somebody did select and that a known limit stopped. It
  carries a reason and it is a strict expected failure -- if it starts passing,
  that is a result, and the matrix is what has to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from tests.models.corpus import CapabilityGate, ModelCase

Kind = Literal["reference", "analyze", "schedule", "sized"]
Status = Literal["PASS", "BLOCKED", "FAIL"]

_KINDS: tuple[Kind, ...] = ("reference", "analyze", "schedule", "sized")


@dataclass(frozen=True)
class CaseResult:
    """One selected case, on one machine, and how it came out."""

    model: str
    target: str
    kind: Kind
    case: str
    function: str | None
    status: Status
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status in ("BLOCKED", "FAIL") and not self.reason.strip():
            raise ValueError(
                f"{self.case!r} is {self.status} without a reason; a result "
                "nobody can act on is not a result"
            )


@dataclass
class CoverageCollector:
    """Everything one run actually executed."""

    results: list[CaseResult] = field(default_factory=list)

    def record(
        self,
        *,
        model: str,
        target: str,
        kind: Kind,
        case: str,
        status: Status,
        function: str | None = None,
        reason: str = "",
    ) -> CaseResult:
        if kind not in _KINDS:
            raise ValueError(f"unknown case kind {kind!r}; expected one of {_KINDS}")
        result = CaseResult(
            model=model,
            target=target,
            kind=kind,
            case=case,
            function=function,
            status=status,
            reason=reason,
        )
        self.results.append(result)
        return result

    def record_gate(
        self,
        gate: CapabilityGate,
        *,
        model: str,
        target: str,
        kind: Kind,
        case: str,
        function: str | None = None,
    ) -> CaseResult:
        """Record a case whose outcome its own capability gate already states."""
        return self.record(
            model=model,
            target=target,
            kind=kind,
            case=case,
            function=function,
            status="BLOCKED" if gate.blocked else "PASS",
            reason=gate.reason,
        )


def build_report(
    collector: CoverageCollector, corpus: tuple[ModelCase, ...]
) -> dict[str, object]:
    """Group what ran under `model -> target -> reference / analyze / schedule`.

    One row per model, not per Module. A model described by several Modules -- a
    hybrid stack's two token mixers and its expert block are three execution
    domains -- still answers one question, "does this model work on this machine",
    and splitting it into three rows would answer that three times and count one
    model as three.

    `untested` is computed per model from its built inventories, so it answers
    "what did nobody look at" rather than "what did somebody remember to list".
    """
    report: dict[str, object] = {}
    by_model: dict[str, list[CaseResult]] = {}
    for result in collector.results:
        by_model.setdefault(result.model, []).append(result)

    models: dict[str, list[ModelCase]] = {}
    for case in corpus:
        models.setdefault(case.model, []).append(case)

    for model_id, cases in models.items():
        inventory = tuple(
            dict.fromkeys(name for case in cases for name in case.inventory())
        )
        targets: dict[str, object] = {}
        for result in by_model.get(model_id, ()):
            targets.setdefault(
                result.target,
                {
                    "reference": [],
                    "analyze": {"tested": [], "untested": []},
                    "schedule": {"tested": [], "untested": []},
                    "sized": {"tested": [], "untested": []},
                },
            )
        for name, section in targets.items():
            executed = [
                result for result in by_model.get(model_id, ()) if result.target == name
            ]
            section["reference"] = [
                _row(result) for result in executed if result.kind == "reference"
            ]
            for kind in ("analyze", "schedule", "sized"):
                rows = [_row(result) for result in executed if result.kind == kind]
                section[kind]["tested"] = rows
                covered = {
                    result.function
                    for result in executed
                    if result.kind == kind and result.function
                }
                section[kind]["untested"] = [
                    function for function in inventory if function not in covered
                ]
        report[model_id] = {"inventory": list(inventory), "targets": targets}
    return report


def _row(result: CaseResult) -> dict[str, object]:
    row: dict[str, object] = {"case": result.case, "status": result.status}
    if result.function:
        row["function"] = result.function
    if result.reason:
        row["reason"] = result.reason
    return row


def render_report(report: dict[str, object]) -> str:
    """The same grouping as indented text, for reading in a terminal."""
    lines: list[str] = []
    for model_id in sorted(report):
        model = report[model_id]
        lines.append(model_id)
        targets = model["targets"]  # type: ignore[index]
        for target_id in sorted(targets):
            section = targets[target_id]
            lines.append(f"    {target_id}")
            for kind in _KINDS:
                lines.append(f"        {kind.capitalize()}")
                if kind == "reference":
                    for row in section[kind]:
                        lines.append(f"            {_line(row)}")
                    continue
                lines.append("            tested")
                for row in section[kind]["tested"]:
                    lines.append(f"                {_line(row)}")
                lines.append("            untested")
                for function in section[kind]["untested"]:
                    lines.append(f"                {function}")
    return "\n".join(lines)


def _line(row: dict[str, object]) -> str:
    text = f"{row.get('function') or row['case']}   {row['status']}"
    if row.get("reason"):
        text += f" ({row['reason']})"
    return text


__all__ = [
    "CaseResult",
    "CoverageCollector",
    "Kind",
    "Status",
    "build_report",
    "render_report",
]
