"""Byte baseline for the complete checked-in HIR fixture corpus."""

import hashlib
import importlib.util
import sys
from pathlib import Path

from tilefoundry.inspection import as_script
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function

ROOT = Path(__file__).parents[1] / "fixtures"
EXPECTED = Path(__file__).with_name("hir_fixture_baseline.sha256")


def _programs():
    for path in sorted((*((ROOT / "logical").glob("*.py")), *((ROOT / "placed").glob("*.py")))):
        if path.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(f"_hir_fixture_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
        sys.path.pop(0)
        for value in vars(module).values():
            if isinstance(value, (Module, Function)):
                yield path.name, value


def test_hir_fixture_output_matches_checked_in_baseline() -> None:
    rendered = "".join(as_script(program) for _, program in _programs())
    assert hashlib.sha256(rendered.encode()).hexdigest()[:16] == EXPECTED.read_text().strip()
