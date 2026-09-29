"""Shared ordering for reports over target-supported Ops."""

from tilefoundry.ir.core import OpCapability, supported_op_capabilities
from tilefoundry.target import Target


def _schema_key(family: tuple[type, tuple[OpCapability, ...]]) -> tuple[str, str]:
    schema = family[0]._op_schema
    return schema.category, schema.name


def capability_families(target: Target) -> tuple[tuple[type, tuple[OpCapability, ...]], ...]:
    """Group admitted variants and order their Ops by schema category and name."""
    grouped: dict[type, list[OpCapability]] = {}
    for op_type, capability in supported_op_capabilities(target):
        grouped.setdefault(op_type, []).append(capability)
    families = [(op_type, tuple(capabilities)) for op_type, capabilities in grouped.items()]
    families.sort(key=_schema_key)
    return tuple(families)


__all__ = ["capability_families"]
