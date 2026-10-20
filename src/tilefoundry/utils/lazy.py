"""Modules that import on first attribute access."""

import importlib


class LazyModule:
    """Stand in for module *name*; import it when an attribute is first read."""

    def __init__(self, name):
        self._name = name
        self._module = None

    def __getattr__(self, attribute):
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, attribute)


torch = LazyModule("torch")
F = LazyModule("torch.nn.functional")
