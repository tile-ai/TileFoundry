"""Declare which operand's bytes a value-form operation re-addresses."""

from __future__ import annotations

from tilefoundry.ir.core import Call

from .registries import DispatchRegistry

buffer_alias_registry: DispatchRegistry = DispatchRegistry("buffer_alias")


register_buffer_alias = buffer_alias_registry.decorator()


def aliased_operand(call: Call) -> int | None:
    """The input whose bytes this view re-addresses, or None for a new value."""
    handler = buffer_alias_registry.lookup(type(call.target))
    return None if handler is None else handler(call)
