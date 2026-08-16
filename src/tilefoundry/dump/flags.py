"""``DumpFlags`` IntFlag — which categories actually emit when a scope is active.

``DumpFlags`` IntFlag — which categories actually emit when a scope is
active.

Initial set per `#tilefoundry:3beef367` 2026-04-27:
``PASS_IR | CODEGEN_SOURCE | BUILD_LOG | ANALYSIS``, plus ``ALL`` as their union.
Finer per-pass flags are out of scope for the MVP.
"""
from __future__ import annotations

from enum import IntFlag


class DumpFlags(IntFlag):
    NONE = 0
    PASS_IR = 1 << 0
    CODEGEN_SOURCE = 1 << 1
    BUILD_LOG = 1 << 2
    ANALYSIS = 1 << 3
    ALL = PASS_IR | CODEGEN_SOURCE | BUILD_LOG | ANALYSIS


__all__ = ["DumpFlags"]
