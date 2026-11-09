"""Declare which operand's bytes a value-form operation re-addresses."""

from __future__ import annotations

from tilefoundry import dispatch
from tilefoundry.ir import expr

alias_registry: dispatch.Registry = dispatch.Registry("buffer_alias")


register_alias = alias_registry.decorator()


def aliased_operand(call: expr.Call) -> int | None:
    """The input whose bytes this view re-addresses, or None for a new value."""
    handler = alias_registry.get(type(call.target), None)
    return None if handler is None else handler(call)


__all__ = ["alias_registry", "aliased_operand", "register_alias"]
