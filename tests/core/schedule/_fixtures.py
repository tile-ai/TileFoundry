"""Shared facts CLI runner; expectations live in each Case guard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.core._corpus import Case, Problems
from tilefoundry.cli import main as cli_main


@dataclass(frozen=True)
class FactsCase(Case):
    target: str = "cpu"

    def guard(self, state, problems: Problems) -> None:
        out, captured = state
        assert captured == ("", "")
        assert out.read_text() == self.path.read_text()


def check_facts_case(case: FactsCase, tmp_path: Path, capsys) -> None:
    problems = Problems(case)
    out = tmp_path / "facts.txt"
    code = problems.run("facts", cli_main, [
        "schedule", "facts", "--target", case.target, str(out),
    ])
    captured = capsys.readouterr()
    problems.expect("exit code", lambda value: value == 0, code)
    if code == 0:
        problems.run("guard", case.guard, (out, captured), problems)
    problems.dump()
