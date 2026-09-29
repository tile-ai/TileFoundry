"""Dependency boundaries between the two IR levels."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_IR = Path(__file__).parents[2] / "src" / "tilefoundry" / "ir"


def _imports(path: Path) -> tuple[tuple[int, str], ...]:
    relative = path.relative_to(_IR.parent.parent)
    module = ".".join(relative.with_suffix("").parts)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            stated = "." * node.level + (node.module or "")
            base = importlib.util.resolve_name(stated, package) if node.level else stated
            found.append((node.lineno, base))
            found.extend((node.lineno, f"{base}.{alias.name}") for alias in node.names)
    return tuple(found)


@pytest.mark.parametrize(
    ("source", "forbidden"),
    (("hir", "tilefoundry.ir.tir"), ("tir", "tilefoundry.ir.hir")),
)
def test_hir_and_tir_do_not_import_each_other(source: str, forbidden: str) -> None:
    violations = [
        f"{path.relative_to(_IR.parent.parent)}:{line}: {module}"
        for path in (_IR / source).rglob("*.py")
        for line, module in _imports(path)
        if module == forbidden or module.startswith(f"{forbidden}.")
    ]
    assert not violations, "cross-level IR imports:\n" + "\n".join(violations)
