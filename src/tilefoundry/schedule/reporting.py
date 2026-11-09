"""Shared ordering for reports over target-supported Ops."""

from tilefoundry import target
from tilefoundry.ir import expr


def schema_key(family: tuple[type, tuple[expr.OpCapability, ...]]) -> tuple[str, str]:
    schema = expr.schema_of(family[0])
    return schema.category, schema.name


def capability_families(tgt: target.Target) -> tuple[tuple[type, tuple[expr.OpCapability, ...]], ...]:
    """Group admitted variants and order their Ops by schema category and name."""
    grouped: dict[type, list[expr.OpCapability]] = {}
    for op_type, capability in expr.supported_op_capabilities(tgt):
        grouped.setdefault(op_type, []).append(capability)
    families = [(op_type, tuple(capabilities)) for op_type, capabilities in grouped.items()]
    families.sort(key=schema_key)
    return tuple(families)


__all__ = ["capability_families"]
