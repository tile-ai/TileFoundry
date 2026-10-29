"""AMX instruction details and constraints against a text golden."""

from pathlib import Path

import pytest

from tests.core._corpus import case
from tests.core.schedule._fixtures import FactsCase, check_facts_case


@case
class AppleFacts(FactsCase):
    path: Path = Path(__file__).parents[1] / "fixtures" / "amx.facts.txt"
    target: str = "apple.m2_pro"


@pytest.mark.parametrize("case", (AppleFacts(),), ids=lambda case: case.id)
def test_facts_case(case, tmp_path, capsys):
    check_facts_case(case, tmp_path, capsys)
