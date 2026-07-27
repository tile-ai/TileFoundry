"""Immutable compilation target values and their fact projections."""

from __future__ import annotations

from dataclasses import dataclass, field


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

    name: str

    def as_facts(self, facts_type: type, query: object = None) -> object:
        """Project this target's specification into *facts_type*.

        This converts what the target already knows; it does not analyze IR,
        build a constraint model, solve, or export a plan. *query* is optional
        and owned by the requesting algorithm: a hardware-only projection omits
        it, while a program-dependent one passes its own private value.
        """
        from tilefoundry.target.facts import TARGET_FACTS  # noqa: PLC0415

        return TARGET_FACTS.project(self, facts_type, query)


@dataclass(frozen=True)
class CpuTarget(Target):
    """Identify the CPU host backend."""

    name: str = field(default="cpu", init=False)


__all__ = ["Architecture", "CpuTarget", "Device", "Target"]
