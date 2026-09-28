"""Operand-pattern matches selected by an already lowered TIR function."""

from __future__ import annotations

from typing import Any

from tilefoundry.inspection import PatternPrinter
from tilefoundry.ir.tir import PrimFunction


def matched(function: PrimFunction, *, source: str | None = None) -> dict[str, Any]:
    """Return the uniquely selected arrangement and captures of every instruction."""
    report = PatternPrinter().matches(function)
    return {"source": source or function.name, **report}


def render(data: dict[str, Any]) -> str:
    """Render one stable human-readable match report."""
    lines = [
        f"source {data['source']}",
        f"function {data['function']}",
        f"target {data['target']}",
    ]
    for call in data["calls"]:
        instruction = "" if call["instruction"] is None else f" [{call['instruction']}]"
        lines.append(f"  call {call['index']}  {call['op']}{instruction}")
        for operand in call["operands"]:
            lines.append(
                f"    {operand['name']}  {operand['type']}  matches {operand['arrangement']}"
            )
            captures = ", ".join(
                f"{name}={value}" for name, value in operand["captures"].items()
            )
            if captures:
                lines.append(f"      captures {captures}")
    return "\n".join(lines)


__all__ = ["matched", "render"]
