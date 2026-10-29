"""CPU instruction details and constraints against a text golden."""

from pathlib import Path

import pytest

from tests.core._corpus import case
from tests.core.schedule._fixtures import FactsCase, check_facts_case


@case
class CpuFacts(FactsCase):
    path: Path = Path(__file__).with_name("fixtures") / "cpu.facts.txt"
    target: str = "cpu"


@pytest.mark.parametrize("case", (CpuFacts(),), ids=lambda case: case.id)
def test_facts_case(case, tmp_path, capsys):
    check_facts_case(case, tmp_path, capsys)
