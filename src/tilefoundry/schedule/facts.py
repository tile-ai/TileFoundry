"""Select registered Op declarations and print them for one target."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal

from tilefoundry.inspection import PatternPrinter
from tilefoundry.ir.core import Op, OpCapability, get_registered_name
from tilefoundry.ir.pattern import Pattern, SwitchPattern, between_rules, declared_execution_mesh
from tilefoundry.schedule._reporting import capability_families
from tilefoundry.target import Target


def listing(target: Target) -> tuple[type[Op], ...]:
    """Return original Op classes admitted by the target capabilities."""
    return tuple(op_type for op_type, _ in capability_families(target))


def _selection(target: Target, wanted: str) -> tuple[type, tuple[OpCapability, ...]]:
    matches = []
    available = []
    for op_type, capabilities in capability_families(target):
        name = get_registered_name(op_type)
        available.append(name)
        if name == wanted:
            matches.append((op_type, capabilities))
        for capability in capabilities:
            if capability.declaration is None:
                continue
            name = get_registered_name(capability.declaration)
            available.append(name)
            if name == wanted:
                matches.append((capability.declaration, (capability,)))
    if len(matches) != 1:
        raise ValueError(f"unknown op {wanted!r} for target {target.identity!r}; available: {available}")
    return matches[0]


def one(target: Target, wanted: str | type) -> type:
    """Resolve an exact registered Op or associated Atom name to its class."""
    name = get_registered_name(wanted) if isinstance(wanted, type) else wanted
    return _selection(target, name)[0]


def _variants(declaration: type, capabilities: tuple[OpCapability, ...]) -> tuple[OpCapability, ...]:
    return tuple(item for item in capabilities
                 if item.declaration is not None and item.declaration is not declaration)


def _execution_mesh(declaration: type, target: Target, capabilities: tuple[OpCapability, ...]) -> Pattern:
    variants = _variants(declaration, capabilities)
    if not variants:
        return declared_execution_mesh(declaration, target)
    return SwitchPattern(variants[0].attribute, {
        get_registered_name(item.declaration): declared_execution_mesh(
            declaration(**{item.attribute: item.declaration}), target,
        )
        for item in variants
    })


def _json(declaration: type, target: Target, capabilities: tuple[OpCapability, ...]) -> dict:
    """Serialize a declaration at the JSON output boundary."""
    printer = PatternPrinter()
    sections = printer.declaration(declaration)
    names = tuple(dict.fromkeys(item.name for item in capabilities if item.name is not None))
    data = {
        "target": target.identity,
        "id": get_registered_name(declaration),
        "capability": None if not names else names[0] if len(names) == 1 else names,
        "execution_mesh": printer.described(_execution_mesh(declaration, target, capabilities)),
        "parameters": list(sections.get("parameters", sections.get("attributes", ()))),
        "operands": list(sections.get("operands", ())),
        "between": [rule.written() for rule in between_rules(declaration)],
    }
    variants = _variants(declaration, capabilities)
    if variants:
        data["variants"] = [_json(item.declaration, target, (item,)) for item in variants]
    return data


def _column(label: str, value: str) -> list[str]:
    first, *rest = value.splitlines()
    prefix = f"  {label:<19}"
    return [prefix + first, *(" " * len(prefix) + line for line in rest)]


def render(declaration: type, target: Target, *, format: Literal["text", "json"] = "text") -> str:
    """Read a registered declaration directly and print text or JSON."""
    if format not in ("text", "json"):
        raise ValueError(f"unknown facts format {format!r}")
    declaration, capabilities = _selection(target, get_registered_name(declaration))
    if format == "json":
        return json.dumps(_json(declaration, target, capabilities), indent=2)

    printer = PatternPrinter()
    lines = [get_registered_name(declaration), *_column("target", target.identity)]
    names = tuple(dict.fromkeys(item.name for item in capabilities if item.name is not None))
    if names:
        lines.extend(_column("target capability", ", ".join(names)))
    lines.extend(_column("execution mesh", printer.described(_execution_mesh(declaration, target, capabilities))))
    sections = printer.declaration(declaration)
    for heading, items in (("parameters", sections.get("parameters", sections.get("attributes", ()))),
                           ("operands", sections.get("operands", ()))):
        if items:
            lines.extend((f"  {heading}", *items))
    between = between_rules(declaration)
    if between:
        lines.extend(("  between operands", *(f"    {rule.written()}" for rule in between)))
    for item in _variants(declaration, capabilities):
        lines.extend(("", render(item.declaration, target)))
    return "\n".join(lines)


def render_all(declarations: Iterable[type], target: Target, *, format: Literal["text", "json"] = "text") -> str:
    """Print the supplied original declarations in their given order."""
    if format == "json":
        return json.dumps({
            "target": target.identity,
            "ops": [_json(declaration, target, _selection(target, get_registered_name(declaration))[1])
                    for declaration in declarations],
        }, indent=2)
    if format != "text":
        raise ValueError(f"unknown facts format {format!r}")
    return "\n\n".join((f"target {target.identity}\nops", *(render(op, target) for op in declarations)))


__all__ = ["listing", "one", "render", "render_all"]
