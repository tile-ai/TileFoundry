"""Metal instruction details and constraints against a text golden."""

from pathlib import Path

import pytest

from tests.core._corpus import case
from tests.core.schedule._fixtures import FactsCase, check_facts_case


@case
class MetalFacts(FactsCase):
    path: Path = Path(__file__).parents[1] / "fixtures" / "metal.facts.txt"
    target: str = "apple.m2_pro.gpu"


@pytest.mark.parametrize("case", (MetalFacts(),), ids=lambda case: case.id)
def test_facts_case(case, tmp_path, capsys):
    check_facts_case(case, tmp_path, capsys)
