"""ANE instruction details and constraints against a text golden."""

from pathlib import Path

import pytest

from tests.core._corpus import case
from tests.core.schedule._fixtures import FactsCase, check_facts_case


@case
class AneFacts(FactsCase):
    path: Path = Path(__file__).parents[1] / "fixtures" / "ane.facts.txt"
    target: str = "apple.m2_pro.ane"


@pytest.mark.parametrize("case", (AneFacts(),), ids=lambda case: case.id)
def test_facts_case(case, tmp_path, capsys):
    check_facts_case(case, tmp_path, capsys)
