"""Instruction declarations admitted by one compilation target."""

from __future__ import annotations

from typing import Any

from tilefoundry.inspection import PatternPrinter
from tilefoundry.ir.core.param_def import collect_param_defs
from tilefoundry.ir.pattern import between_rules
from tilefoundry.target import Target

from .instructions import families


def instructions(target: Target) -> tuple[type, ...]:
    """Return the registered instruction Ops supported by ``target``."""
    return tuple(family.op_type for family in families(target))


def listing(target: Target) -> dict[str, Any]:
    """Describe every instruction declaration supported by ``target``."""
    return {
        "target": target.identity,
        "instructions": [
            {
                "id": family.id,
                "capability": family.capability,
            }
            for family in families(target)
        ],
    }


def _instruction_id(op_type: type) -> str:
    reference = getattr(op_type, "reference_name", "")
    if reference:
        return reference
    schema = op_type._op_schema
    return f"{schema.dialect}.{schema.name}"


def _selection(target: Target, wanted_id: str) -> tuple[type, str | tuple[str, ...] | None]:
    matches: list[tuple[type, str | tuple[str, ...] | None]] = []
    available = []
    for family in families(target):
        available.append(family.id)
        if family.id == wanted_id:
            matches.append((family.op_type, family.capability))
        for declaration in family.declarations:
            if not declaration.is_variant:
                continue
            available.append(declaration.id)
            if declaration.id == wanted_id:
                matches.append((declaration.declaration, declaration.capability))
    if len(matches) != 1:
        raise ValueError(
            f"unknown instruction {wanted_id!r} for target {target.identity!r}; "
            f"available: {available}"
        )
    return matches[0]


def _issuer_pattern(op_type: type):
    scope_pattern = getattr(op_type, "scope_pattern", None)
    if callable(scope_pattern):
        return scope_pattern()
    scope = next((param for param in collect_param_defs(op_type) if param.name == "scope"), None)
    return None if scope is None else scope.pattern


def one(target: Target, wanted: str | type) -> dict[str, Any]:
    """Describe one supported instruction declaration by its exact name."""
    wanted_id = _instruction_id(wanted) if isinstance(wanted, type) else wanted
    op_type, capability = _selection(target, wanted_id)
    issuer = _issuer_pattern(op_type)
    sections = PatternPrinter().declaration_sections(op_type)
    return {
        "target": target.identity,
        "id": _instruction_id(op_type),
        "capability": capability,
        "issued_by": None if issuer is None else PatternPrinter().described(issuer),
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
    if data["issued_by"] is not None:
        lines.extend(_column("issuer mesh", data["issued_by"]))
    for heading in ("parameters", "operands"):
        if data[heading]:
            lines.append(f"  {heading}")
            lines.extend(data[heading])
    if data["between"]:
        lines.append("  between operands")
        lines.extend(f"    {rule}" for rule in data["between"])
    return "\n".join(lines)


__all__ = ["instructions", "listing", "one", "render"]
