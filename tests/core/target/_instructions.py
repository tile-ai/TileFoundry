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


def check_architecture_capabilities(target_type, module_prefix: str) -> None:
    instructions = tuple(
        (op_type, capability)
        for op_type, capability in instruction_declarations()
        if op_type.__module__.startswith(module_prefix)
    )
    capabilities = {capability.name for _op_type, capability in instructions}
    architecture_capabilities = {
        capability
        for target in target_type.available()
        for capability in target.architecture.capabilities
    }
    assert capabilities <= architecture_capabilities
