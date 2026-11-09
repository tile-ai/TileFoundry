"""Compilation Targets: the base value, the registry, the Facts, and the hardware documents."""

from __future__ import annotations

from . import cpu, hardware
from .analyzer import AnalysisCallable, AnalysisChecker, Analyzer
from .facts import (
    ExplicitMemoryLevelFacts,
    FactsError,
    ImplicitMemoryLevelFacts,
    MemoryExecutionFacts,
    MemoryHierarchyFacts,
    MemoryLevelRelation,
    MemoryRelationKind,
    PerformanceServiceFacts,
    ThroughputFacts,
    TopologyFacts,
    TopologyLevelFacts,
    facts_result,
)
from .registry import Catalog, LoadError, catalog, load, register, registered, unregister
from .spec import HardwareSpec, architecture_of, available_device_ids, check_compatible, select
from .target import (
    Architecture,
    Device,
    Target,
    UnsupportedCapabilityError,
    bind,
    bindings,
    instance,
    summary,
)
