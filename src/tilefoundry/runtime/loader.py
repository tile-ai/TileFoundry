"""Load a linked product through the standalone compiled-library loader."""

from __future__ import annotations

from .module import CompiledModule


def load_linked_module(linked: "LinkedModule") -> CompiledModule:
    """Restore the root from its library without using in-memory signatures."""
    return CompiledModule.from_library(linked.library_path)


__all__ = ["load_linked_module"]
