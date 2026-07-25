"""Load a declarative model source with a config injected."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

__all__ = ["load_model"]


def load_model(path: "str | Path", **namespace) -> ModuleType:
    """Execute the model source at *path* with *namespace* pre-bound in its
    globals. Never enters ``sys.modules``, so each call re-executes the source
    and yields its own ``Module`` objects."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"load_model: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(namespace)
    spec.loader.exec_module(module)
    return module
