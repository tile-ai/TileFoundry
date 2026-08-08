from __future__ import annotations

import importlib.util
import tempfile
import uuid
from pathlib import Path

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function


def import_dsl(source: str, name: str | None = None) -> Module | Function:
    """Write DSL source to a real file and return the object its decorators build."""
    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "source.py"
        source_path.write_text(source)
        module_name = f"_tilefoundry_test_source_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)

    if name is not None:
        value = getattr(loaded, name)
        assert isinstance(value, (Module, Function))
        return value

    values = [
        value
        for value in vars(loaded).values()
        if isinstance(value, (Module, Function))
    ]
    assert len(values) == 1, "name the DSL binding when source builds more than one"
    return values[0]
