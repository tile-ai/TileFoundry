"""Instruction declarations admitted by one compilation target.

These are declarations an author can name directly, deliberately not the set
of TIR ops with registered schedule access relations. In particular,
``TiledMma`` is an op whose selected atom owns the declaration.
"""

from __future__ import annotations

from typing import Any

from tilefoundry.inspection import PatternPrinter
from tilefoundry.ir.core.param_def import collect_param_defs
from tilefoundry.ir.pattern import between_rules
from tilefoundry.ir.tir.cuda.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry.ir.tir.cuda.memory.ldmatrix import LdMatrix
from tilefoundry.ir.tir.cuda.nn.sm80_mma import Mma
from tilefoundry.ir.tir.cuda.nn.wgmma import Wgmma
from tilefoundry.target import Target

SCHEDULED = (Mma, Wgmma, LdMatrix, CopyAsyncTensor)


def _instruction_id(op_type: type) -> str:
    reference = getattr(op_type, "reference_name", "")
    if reference:
        return reference
    schema = op_type._op_schema
    return f"{schema.dialect}.{schema.name}"


def _capabilities(target: Target) -> frozenset[str]:
    architecture = getattr(target, "architecture", None)
    return frozenset(getattr(architecture, "capabilities", ()))


def instructions(target: Target) -> tuple[type, ...]:
    """Return the named instruction declarations supported by ``target``."""
    capabilities = _capabilities(target)
    return tuple(
        op_type
        for op_type in SCHEDULED
        if isinstance(getattr(op_type, "capability", None), str)
        and op_type.capability in capabilities
    )


def listing(target: Target) -> dict[str, Any]:
    """Describe every instruction declaration supported by ``target``."""
    return {
        "target": target.identity,
        "instructions": [
            {
                "id": _instruction_id(op_type),
                "capability": op_type.capability,
            }
            for op_type in instructions(target)
        ],
    }


def _issuer_pattern(op_type: type):
    scope_pattern = getattr(op_type, "scope_pattern", None)
    if callable(scope_pattern):
        return scope_pattern()
    scope = next((param for param in collect_param_defs(op_type) if param.name == "scope"), None)
    return None if scope is None else scope.pattern


def one(target: Target, wanted: str | type) -> dict[str, Any]:
    """Describe one supported instruction declaration by its exact name."""
    supported = instructions(target)
    wanted_id = _instruction_id(wanted) if isinstance(wanted, type) else wanted
    matches = tuple(op_type for op_type in supported if _instruction_id(op_type) == wanted_id)
    if len(matches) != 1:
        available = [_instruction_id(op_type) for op_type in supported]
        raise ValueError(
            f"unknown instruction {wanted_id!r} for target {target.identity!r}; "
            f"available: {available}"
        )
    (op_type,) = matches
    issuer = _issuer_pattern(op_type)
    sections = PatternPrinter().declaration_sections(op_type)
    return {
        "target": target.identity,
        "id": _instruction_id(op_type),
        "capability": op_type.capability,
        "issued_by": None if issuer is None else PatternPrinter().described(issuer),
        "parameters": list(sections.get("parameters", ())),
        "operands": list(sections.get("operands", ())),
        "between": [rule.written() for rule in between_rules(op_type)],
    }


def _column(label: str, value: str) -> list[str]:
    first, *rest = value.splitlines()
    prefix = f"  {label:<12}"
    return [prefix + first, *(" " * len(prefix) + line for line in rest)]


def render(data: dict[str, Any]) -> str:
    """Render one listing or one declaration report as stable text."""
    if "instructions" in data:
        rows = data["instructions"]
        width = max((len(row["id"]) for row in rows), default=0)
        lines = [f"target {data['target']}", "instructions"]
        lines.extend(f"  {row['id']:<{width}}  {row['capability']}" for row in rows)
        return "\n".join(lines)

    lines = [data["id"], *_column("target", data["target"])]
    if data["capability"] is not None:
        lines.extend(_column("needs", data["capability"]))
    if data["issued_by"] is not None:
        lines.extend(_column("issued by", data["issued_by"]))
    for heading in ("parameters", "operands"):
        if data[heading]:
            lines.append(f"  {heading}")
            lines.extend(data[heading])
    if data["between"]:
        lines.append("  between operands")
        lines.extend(f"    {rule}" for rule in data["between"])
    return "\n".join(lines)


__all__ = ["SCHEDULED", "instructions", "listing", "one", "render"]
