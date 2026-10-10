"""Declare which operand's bytes a value-form operation re-addresses."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Call

from .registries import DispatchRegistry

buffer_alias_registry: DispatchRegistry = DispatchRegistry("buffer_alias")


@dataclass(frozen=True)
class Alias:
    """An input's backing buffer, optionally selecting one tuple element."""

    operand: int
    element: int | None = None


register_buffer_alias = buffer_alias_registry.decorator()


def aliased_operand(call: Call) -> Alias | None:
    """Return the input buffer reused by this call, or None for a new value."""
    handler = buffer_alias_registry.lookup(type(call.target))
    return None if handler is None else handler(call)
