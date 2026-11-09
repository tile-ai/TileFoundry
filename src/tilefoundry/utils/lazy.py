"""Modules that import on first attribute access."""

import importlib

__all__ = ["F", "LazyModule", "torch"]


class LazyModule:
    """Stand in for module *name*; import it when an attribute is first read."""

    def __init__(self, name):
        self.module_name = name
        self.loaded = None

    def __getattr__(self, attribute):
        if self.loaded is None:
            self.loaded = importlib.import_module(self.module_name)
        return getattr(self.loaded, attribute)


torch = LazyModule("torch")
F = LazyModule("torch.nn.functional")
