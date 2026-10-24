"""Instruction capability declarations shared by target tests."""

from __future__ import annotations

from tilefoundry.ir.core import OpCapability
from tilefoundry.ir.core.op_registry import iter_schemas


def instruction_declarations():
    instructions = []
    for schema in iter_schemas():
        if schema.op_class is None:
            continue
        stated = vars(schema.op_class).get("capability")
        capabilities = (stated,) if isinstance(stated, OpCapability) else stated or ()
        instructions.extend(
            (schema.op_class, capability)
            for capability in capabilities
            if capability.name is not None
        )
    instructions = tuple(instructions)
    return instructions
