"""Category re-export coverage (code-organization.md Rule 5).

Every ``@register_op``-decorated Op class defined under an HIR category
package (``math`` / ``nn`` / ``tensor`` / ``shape`` / ``sharding``) must be
importable from that category's package (``from tilefoundry.ir.hir.<cat>
import <Cls>``), not just reachable via ``_auto_import`` registration
side-effects.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

_CATEGORIES = ("math", "nn", "tensor", "shape", "sharding")


def _registered_op_classes(category: str) -> list[tuple[str, type]]:
    pkg = importlib.import_module(f"tilefoundry.ir.hir.{category}")
    found: list[tuple[str, type]] = []
    for _, modname, _ in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        module = importlib.import_module(modname)
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and obj.__module__ == modname
                and hasattr(obj, "_op_schema")
            ):
                found.append((name, obj))
    return found


@pytest.mark.parametrize("category", _CATEGORIES)
def test_every_registered_op_is_category_reexported(category: str) -> None:
    pkg = importlib.import_module(f"tilefoundry.ir.hir.{category}")
    for name, cls in _registered_op_classes(category):
        assert getattr(pkg, name, None) is cls, (
            f"{cls.__module__}.{name} is registered but not re-exported from "
            f"tilefoundry.ir.hir.{category}"
        )
