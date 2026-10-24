"""TIR fixture loading and canonical source roundtrip checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.core._source import import_dsl
from tilefoundry.inspection import as_script
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.visitor_registry.verify import verify_module


def _module_functions(module):
    return tuple(frame.function for frame in list_call_paths(module).values())


def _module_in(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    modules = [value for value in vars(loaded).values() if isinstance(value, Module)]
    return loaded.Root if path.parent.name == "scaffold" else modules[0]


def check_tir_roundtrip(path: Path, canonical) -> None:
    module = _module_in(path)
    printed = as_script(module)
    if path in canonical:
        assert printed == path.read_text()
    rebuilt = import_dsl(printed, name=module.name)
    assert as_script(rebuilt) == printed
    verify_module(rebuilt)
    original = [param.is_const for fn in _module_functions(module) for param in fn.params]
    restored = [param.is_const for fn in _module_functions(rebuilt) for param in fn.params]
    assert restored == original
    if any(original):
        assert "ConstTensor[" in printed
    if path.stem == "tir_launches":
        assert "launch(child.run," in printed
