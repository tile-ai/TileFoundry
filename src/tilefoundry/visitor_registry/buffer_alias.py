"""Declare which operand's bytes a value-form operation re-addresses."""

from __future__ import annotations

from tilefoundry.ir.core import Call
from tilefoundry.ir.core.param_def import ParamDef

from .registries import DispatchRegistry

buffer_alias_registry: DispatchRegistry = DispatchRegistry("buffer_alias")


def register_buffer_alias(op_cls: type, param: ParamDef) -> None:
    """Declare that op_cls's result re-addresses its ``param`` operand's bytes."""
    buffer_alias_registry.register(op_cls, param)


def aliased_operand(call: Call) -> int | None:
    """Return the aliased input's argument position, or None for a new value."""
    param = buffer_alias_registry.lookup(type(call.target))
    if param is None:
        return None
    inputs = tuple(p for p in type(call.target)._op_schema.signature if p.kind == "input")
    return inputs.index(param)
