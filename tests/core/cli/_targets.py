"""Tests and helpers for the owning target."""

from __future__ import annotations

from tilefoundry.target import CpuTarget


class ListedCpuTarget(CpuTarget):
    name = "tests.cli.listed_cpu"
