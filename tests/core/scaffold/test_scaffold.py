"""Target-neutral tests and helpers."""

from pathlib import Path

import pytest

import tilefoundry
from tilefoundry.cli.source import load_authored_ir

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scaffold"


@pytest.mark.parametrize("fixture", ("hir_moe", "hir_layers"))
def test_full_refuses_hir(fixture):
    module = load_authored_ir(f"{FIXTURES / (fixture + '.py')}:Root")
    with pytest.raises(ValueError, match='emit="skeleton" or lowering to TIR'):
        tilefoundry.compile(module)
