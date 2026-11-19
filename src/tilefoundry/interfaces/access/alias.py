"""Declare which operand's bytes a value-form operation re-addresses."""

from __future__ import annotations

from tilefoundry import dispatch
from tilefoundry import target as tilefoundry_target
from tilefoundry.ir import expr

alias_registry: dispatch.Registry = dispatch.Registry("buffer_alias")


register_alias = alias_registry.decorator()


def aliased_operand(call: expr.Call, owners: tilefoundry_target.StorageOwners) -> int | None:
    """The input whose bytes this view re-addresses, or None for a new value.

    *owners* says which topologies hold their own copy of each storage.
    """
    handler = alias_registry.get(type(call.target), None)
    return None if handler is None else handler(call, owners)


__all__ = ["alias_registry", "aliased_operand", "register_alias"]
