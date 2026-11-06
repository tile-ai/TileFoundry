"""One installed Metal facts text golden, using the shared CLI Case runner."""
from pathlib import Path
from tests.core.schedule._fixtures import FactsCase, check_facts_case


def test_metal_facts(tmp_path, capsys):
    check_facts_case(FactsCase(path=Path(__file__).parents[1] / "fixtures/metal.facts.txt", target="apple.m2_pro.gpu"), tmp_path, capsys)
