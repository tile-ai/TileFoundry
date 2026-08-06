"""Immutable compilation target values and their fact projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeVar

from tilefoundry.target.registration import register_target

FactsT = TypeVar("FactsT")


class Architecture:
    """Base value for compilation architecture facts."""

    name: str
    max_threads_per_cta: int


class Device:
    """Base value for one concrete device's resource facts."""

    name: str
    sm_count: int


@dataclass(frozen=True)
class Target:
    """Identify a compilation backend.

    A target is a value: what it knows is its hardware, and the only way an
    algorithm reads that is by naming the facts it wants. There is nothing
    mutable on it and nothing registered against it, so two equal targets are
    interchangeable everywhere.
    """

    name: ClassVar[str]

    def get_analyzer(self, selector: str) -> "Analyzer":
        """Return the analysis service selected by this concrete Target."""
        raise ValueError(
            f"{type(self).__name__} ({getattr(type(self), 'name', '<unregistered>')}): "
            f"no analyzer for {selector!r}"
        )

    def get_scheduler(self, topology: str) -> "Scheduler":
        """Return the scheduler selected by this concrete Target."""
        raise ValueError(
            f"{type(self).__name__} ({getattr(type(self), 'name', '<unregistered>')}): "
            f"no scheduler for {topology!r}"
        )

    def get_code_generator(self) -> "CodeGenerator":
        """Return the code-generation service selected by this Target."""
        raise ValueError(
            f"{type(self).__name__} ({getattr(type(self), 'name', '<unregistered>')}): "
            "no code generator"
        )

    def get_facts(
        self, facts_type: type[FactsT], query: object | None = None
    ) -> FactsT:
        """Return one immutable hardware-facts aggregate for this Target."""
        raise ValueError(
            f"{type(self).__name__} ({getattr(type(self), 'name', '<unregistered>')}): "
            "no Facts projection for "
            f"{getattr(facts_type, '__name__', facts_type)!r}"
        )


@register_target
@dataclass(frozen=True)
class CpuTarget(Target):
    """Identify the CPU host backend."""

    name: ClassVar[str] = "cpu"

    def get_code_generator(self) -> "CodeGenerator":
        from tilefoundry.codegen.cpu.module import (  # noqa: PLC0415
            CPU_CODE_GENERATOR,
        )

        return CPU_CODE_GENERATOR


__all__ = ["Architecture", "CpuTarget", "Device", "Target"]
