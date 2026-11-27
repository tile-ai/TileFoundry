"""Bind a linked library's entry into a ``CompiledModule``."""

from __future__ import annotations

from dataclasses import replace

from .module import CompiledModule


def load_linked_module(linked: "LinkedModule") -> CompiledModule:
    """Load *linked*'s shared library and bind its entry into a ``CompiledModule``."""
    entry = linked.functions[0]
    return CompiledModule.from_library(
        linked.library_path, replace(entry.signature, name=entry.name)
    )


__all__ = ["load_linked_module"]
