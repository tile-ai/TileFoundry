"""Immutable compilation target values, class registration, and fact projections."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Mapping, TypeVar

from tilefoundry.ir.types.shard import Topology
from tilefoundry.target.facts import TopologyLimitFacts
from tilefoundry.target.services import Analyzer, CodeGenerator, Scheduler
from tilefoundry.utils.python_source import PythonExpr, dataclass_to_python

FactsT = TypeVar("FactsT")


def _target_summary(target: object) -> str:
    """The stable Target identity suitable for user-facing diagnostics."""
    target_type = type(target)
    return f"{target_type.__name__} ({getattr(target_type, 'name', '<unregistered>')})"


class UnsupportedCapabilityError(Exception):
    """A Target cannot provide the capability a compiler operation requested."""


class Architecture:
    """Base value for compilation architecture facts."""

    name: str
    max_threads_per_cta: int

    def _python_import_module(self) -> str:
        return type(self).__module__

    def to_python(self) -> PythonExpr:
        return dataclass_to_python(self, self._python_import_module())


class Device:
    """Base value for one concrete device's resource facts."""

    name: str
    sm_count: int

    def _python_import_module(self) -> str:
        return type(self).__module__

    def to_python(self) -> PythonExpr:
        return dataclass_to_python(self, self._python_import_module())


@dataclass(frozen=True)
class Target:
    """Identify a compilation backend.

    A target is a value: what it knows is its hardware, and the only way an
    algorithm reads that is by naming the facts it wants. There is nothing
    mutable on it and nothing registered against it, so two equal targets are
    interchangeable everywhere.
    """

    name: ClassVar[str]
    topology_levels: ClassVar[tuple[str, ...]] = ()

    def _python_import_module(self) -> str:
        return type(self).__module__

    def to_python(self) -> PythonExpr:
        return dataclass_to_python(self, self._python_import_module())

    def get_analyzer(self, selector: str) -> Analyzer:
        """Return the analysis service selected by this concrete Target."""
        raise UnsupportedCapabilityError(
            f"{_target_summary(self)}: no analyzer for {selector!r}"
        )

    def get_scheduler(self, topology: str) -> Scheduler:
        """Return the scheduler selected by this concrete Target."""
        raise UnsupportedCapabilityError(
            f"{_target_summary(self)}: no scheduler for {topology!r}"
        )

    def get_code_generator(self) -> CodeGenerator:
        """Return the code-generation service selected by this Target."""
        raise UnsupportedCapabilityError(
            f"{_target_summary(self)}: no code generator"
        )

    def get_facts(
        self, facts_type: type[FactsT], query: object | None = None
    ) -> FactsT:
        """Return one immutable hardware-facts aggregate for this Target."""
        raise UnsupportedCapabilityError(
            f"{_target_summary(self)}: no Facts projection for "
            f"{getattr(facts_type, '__name__', facts_type)!r}"
        )

    def validate_program_topology(self, topology: Topology) -> None:
        """Validate one declared topology against this Target's Facts."""
        target_summary = _target_summary(self)
        if topology.name not in self.topology_levels:
            raise ValueError(
                f"{target_summary}: unsupported topology level {topology.name!r}; "
                f"supported levels are {self.topology_levels}"
            )
        limit = self.get_facts(
            TopologyLimitFacts, topology.name
        ).max_static_extent
        if topology.size is None:
            if limit is None:
                return
            raise ValueError(
                f"{target_summary}: topology {topology.name!r} requires a positive "
                f"static integer extent, got {topology.size!r}"
            )
        if not isinstance(topology.size, int) or isinstance(topology.size, bool):
            raise ValueError(
                f"{target_summary}: topology {topology.name!r} requires a positive "
                f"static integer extent, got {topology.size!r}"
            )
        if topology.size < 1:
            raise ValueError(
                f"{target_summary}: topology {topology.name!r} extent {topology.size} "
                "must be positive"
            )
        if limit is not None and topology.size > limit:
            raise ValueError(
                f"{target_summary}: topology {topology.name!r} extent {topology.size} "
                f"must satisfy 1 <= extent <= {limit}"
            )


class _BuiltinAnalysisTarget(Target):
    """Target base for backends that serve the standard analysis families."""

    def get_analyzer(self, selector: str) -> Analyzer:
        from tilefoundry.analysis.registry import builtin_analyzer  # noqa: PLC0415

        analyzer = builtin_analyzer(selector)
        if analyzer is not None:
            return analyzer
        return super().get_analyzer(selector)


def target_instance(value: object, *, subject: str = "target") -> Target:
    """Return a constructed Target value or raise the authored-boundary error."""
    if isinstance(value, Target):
        return value
    if isinstance(value, str):
        raise TypeError(
            f"{subject} must be a Target instance, not a string; import and "
            "construct the Target class explicitly"
        )
    raise TypeError(
        f"{subject} must be a Target instance, got {type(value).__name__}"
    )


TargetT = TypeVar("TargetT", bound=Target)
_TARGET_CLASSES: dict[str, type[Target]] = {}
_TARGET_PROVIDERS: dict[str, tuple[str, str]] = {}


def _provider_identity(target_type: type[Target]) -> tuple[str, str]:
    return (target_type.__module__, target_type.__qualname__)


def register_target(target_type: type[TargetT]) -> type[TargetT]:
    """Register a concrete Target class under its explicitly declared name."""
    if not isinstance(target_type, type) or not issubclass(target_type, Target):
        raise TypeError("@register_target expects a Target subclass")
    if target_type is Target or inspect.isabstract(target_type):
        raise TypeError("@register_target expects a concrete Target subclass")
    if "name" not in vars(target_type):
        raise ValueError(
            f"@register_target {target_type.__qualname__}: declare a class-level "
            "non-empty name instead of inheriting one"
        )
    name = target_type.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"@register_target {target_type.__qualname__}: name must be a "
            "non-empty string"
        )
    provider = _provider_identity(target_type)
    existing_provider = _TARGET_PROVIDERS.get(name)
    if existing_provider is not None and existing_provider != provider:
        existing = _TARGET_CLASSES[name]
        raise ValueError(
            f"@register_target: name {name!r} is already owned by "
            f"{existing.__module__}.{existing.__qualname__}; cannot register "
            f"{target_type.__module__}.{target_type.__qualname__}"
        )
    if existing_provider != provider:
        _TARGET_CLASSES[name] = target_type
        _TARGET_PROVIDERS[name] = provider
    return target_type


def registered_targets() -> Mapping[str, type[Target]]:
    """Return a read-only view of registered Target classes by name."""
    return MappingProxyType(_TARGET_CLASSES)


__all__ = [
    "Architecture",
    "Device",
    "Target",
    "UnsupportedCapabilityError",
    "register_target",
    "registered_targets",
    "target_instance",
]
