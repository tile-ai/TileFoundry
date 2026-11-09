"""Select registered Op declarations and print them for one target."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal

from tilefoundry import inspection
from tilefoundry import target as tilefoundry_target
from tilefoundry.ir import expr, pattern

from .reporting import capability_families


def listing(target: tilefoundry_target.Target) -> tuple[type[expr.Op], ...]:
    """Return original Op classes admitted by the target capabilities."""
    return tuple(op_type for op_type, _ in capability_families(target))


def selection_of(target: tilefoundry_target.Target, wanted: str) -> tuple[type, tuple[expr.OpCapability, ...]]:
    matches = []
    available = []
    for op_type, capabilities in capability_families(target):
        name = expr.get_registered_name(op_type)
        available.append(name)
        if name == wanted:
            matches.append((op_type, capabilities))
        for capability in capabilities:
            if capability.declaration is None:
                continue
            name = expr.get_registered_name(capability.declaration)
            available.append(name)
            if name == wanted:
                matches.append((capability.declaration, (capability,)))
    if len(matches) != 1:
        raise ValueError(f"unknown op {wanted!r} for target {target.identity!r}; available: {available}")
    return matches[0]


def one(target: tilefoundry_target.Target, wanted: str | type) -> type:
    """Resolve an exact registered Op or associated Atom name to its class."""
    name = expr.get_registered_name(wanted) if isinstance(wanted, type) else wanted
    return selection_of(target, name)[0]


def variants_of(declaration: type, capabilities: tuple[expr.OpCapability, ...]) -> tuple[expr.OpCapability, ...]:
    return tuple(item for item in capabilities
                 if item.declaration is not None and item.declaration is not declaration)


def execution_mesh_of(declaration: type, target: tilefoundry_target.Target, capabilities: tuple[expr.OpCapability, ...]) -> pattern.Pattern:
    variants = variants_of(declaration, capabilities)
    if not variants:
        return pattern.declared_execution_mesh(declaration, target)
    return pattern.Switch(variants[0].attribute, {
        expr.get_registered_name(item.declaration): pattern.declared_execution_mesh(
            declaration(**{item.attribute: item.declaration}), target,
        )
        for item in variants
    })


def json_of(declaration: type, target: tilefoundry_target.Target, capabilities: tuple[expr.OpCapability, ...]) -> dict:
    """Serialize a declaration at the JSON output boundary."""
    printer = inspection.PatternPrinter()
    sections = printer.declaration(declaration)
    names = tuple(dict.fromkeys(item.name for item in capabilities if item.name is not None))
    data = {
        "target": target.identity,
        "id": expr.get_registered_name(declaration),
        "capability": None if not names else names[0] if len(names) == 1 else names,
        "execution_mesh": printer.described(execution_mesh_of(declaration, target, capabilities)),
        "parameters": list(sections.get("parameters", sections.get("attributes", ()))),
        "operands": list(sections.get("operands", ())),
        "between": [rule.written() for rule in pattern.between_rules(declaration)],
    }
    variants = variants_of(declaration, capabilities)
    if variants:
        data["variants"] = [json_of(item.declaration, target, (item,)) for item in variants]
    return data


def column_of(label: str, value: str) -> list[str]:
    first, *rest = value.splitlines()
    prefix = f"  {label:<19}"
    return [prefix + first, *(" " * len(prefix) + line for line in rest)]


def render(declaration: type, target: tilefoundry_target.Target, *, format: Literal["text", "json"] = "text") -> str:
    """Read a registered declaration directly and print text or JSON."""
    if format not in ("text", "json"):
        raise ValueError(f"unknown facts format {format!r}")
    declaration, capabilities = selection_of(target, expr.get_registered_name(declaration))
    if format == "json":
        return json.dumps(json_of(declaration, target, capabilities), indent=2)

    printer = inspection.PatternPrinter()
    lines = [expr.get_registered_name(declaration), *column_of("target", target.identity)]
    names = tuple(dict.fromkeys(item.name for item in capabilities if item.name is not None))
    if names:
        lines.extend(column_of("target capability", ", ".join(names)))
    lines.extend(column_of("execution mesh", printer.described(execution_mesh_of(declaration, target, capabilities))))
    sections = printer.declaration(declaration)
    for heading, items in (("parameters", sections.get("parameters", sections.get("attributes", ()))),
                           ("operands", sections.get("operands", ()))):
        if items:
            lines.extend((f"  {heading}", *items))
    between = pattern.between_rules(declaration)
    if between:
        lines.extend(("  between operands", *(f"    {rule.written()}" for rule in between)))
    for item in variants_of(declaration, capabilities):
        lines.extend(("", render(item.declaration, target)))
    return "\n".join(lines)


def render_all(declarations: Iterable[type], target: tilefoundry_target.Target, *, format: Literal["text", "json"] = "text") -> str:
    """Print the supplied original declarations in their given order."""
    if format == "json":
        return json.dumps({
            "target": target.identity,
            "ops": [json_of(declaration, target, selection_of(target, expr.get_registered_name(declaration))[1])
                    for declaration in declarations],
        }, indent=2)
    if format != "text":
        raise ValueError(f"unknown facts format {format!r}")
    return "\n\n".join((f"target {target.identity}\nops", *(render(op, target) for op in declarations)))


__all__ = ["listing", "one", "render", "render_all"]
