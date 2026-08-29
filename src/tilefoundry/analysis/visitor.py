"""Shared structural facts collected from one normalized HIR graph."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.module import Module
from tilefoundry.target import Target


@dataclass
class AnalyzeContext:
    """Per-call inputs and the current shared lexical scope."""

    module: Module
    target: Target
    level: str | None
    options: object | None
    root: "Scope"
    current: "Scope"


__all__ = ["AnalyzeContext"]
