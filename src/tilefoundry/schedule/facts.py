"""Instruction declarations admitted by one compilation target."""

from __future__ import annotations

from typing import Any

from tilefoundry.inspection import PatternPrinter
from tilefoundry.ir.core import OpCapability, op_identifier
from tilefoundry.ir.pattern import between_rules, declared_execution_mesh
from tilefoundry.schedule._reporting import capability_families
from tilefoundry.target import Target


def _family_capability(
    capabilities: tuple[OpCapability, ...],
) -> str | tuple[str, ...] | None:
    names = tuple(capability.name for capability in capabilities if capability.name is not None)
    if not names:
        return None
    return names[0] if len(names) == 1 else names


def instructions(target: Target) -> tuple[type, ...]:
    """Return the registered instruction Ops supported by ``target``."""
    return tuple(op_type for op_type, _capabilities in capability_families(target))


def listing(target: Target) -> dict[str, Any]:
    """Describe every instruction declaration supported by ``target``."""
    return {
        "target": target.identity,
        "instructions": [
            {
                "id": op_identifier(op_type),
                "capability": _family_capability(capabilities),
            }
            for op_type, capabilities in capability_families(target)
        ],
    }


def _selection(target: Target, wanted_id: str) -> tuple[type, str | tuple[str, ...] | None]:
    matches: list[tuple[type, str | tuple[str, ...] | None]] = []
    available = []
    for op_type, capabilities in capability_families(target):
        op_id = op_identifier(op_type)
        family_capability = _family_capability(capabilities)
        available.append(op_id)
        if op_id == wanted_id:
            matches.append((op_type, family_capability))
        for capability in capabilities:
            if capability.attribute is None:
                continue
            declaration = capability.declaration
            declaration_id = op_identifier(declaration)
            available.append(declaration_id)
            if declaration_id == wanted_id:
                matches.append((declaration, capability.name))
    if len(matches) != 1:
        raise ValueError(
            f"unknown instruction {wanted_id!r} for target {target.identity!r}; "
            f"available: {available}"
        )
    return matches[0]


def one(target: Target, wanted: str | type) -> dict[str, Any]:
    """Describe one supported instruction declaration by its exact name."""
    wanted_id = op_identifier(wanted) if isinstance(wanted, type) else wanted
    op_type, capability = _selection(target, wanted_id)
    execution_mesh = declared_execution_mesh(op_type)
    sections = PatternPrinter().declaration(op_type)
    return {
        "target": target.identity,
        "id": op_identifier(op_type),
        "capability": capability,
        "execution_mesh": PatternPrinter().described(execution_mesh),
        "parameters": list(sections.get("parameters", ())),
        "operands": list(sections.get("operands", ())),
        "between": [rule.written() for rule in between_rules(op_type)],
    }


def _column(label: str, value: str) -> list[str]:
    first, *rest = value.splitlines()
    prefix = f"  {label:<19}"
    return [prefix + first, *(" " * len(prefix) + line for line in rest)]


def _written_capability(capability: str | tuple[str, ...] | None) -> str:
    if capability is None:
        return "all targets"
    if isinstance(capability, tuple):
        return ", ".join(capability)
    return capability


def render(data: dict[str, Any]) -> str:
    """Render one listing or one declaration report as stable text."""
    if "instructions" in data:
        rows = data["instructions"]
        width = max((len(row["id"]) for row in rows), default=0)
        lines = [f"target {data['target']}", "instructions"]
        lines.extend(
            f"  {row['id']:<{width}}  {_written_capability(row['capability'])}" for row in rows
        )
        return "\n".join(lines)

    lines = [data["id"], *_column("target", data["target"])]
    if data["capability"] is not None:
        lines.extend(_column("target capability", _written_capability(data["capability"])))
    lines.extend(_column("execution mesh", data["execution_mesh"]))
    for heading in ("parameters", "operands"):
        if data[heading]:
            lines.append(f"  {heading}")
            lines.extend(data[heading])
    if data["between"]:
        lines.append("  between operands")
        lines.extend(f"    {rule}" for rule in data["between"])
    return "\n".join(lines)


__all__ = ["instructions", "listing", "one", "render"]
