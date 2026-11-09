"""Immutable compilation target values, their bindings, and fact projections."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from functools import cached_property
from types import MappingProxyType
from typing import ClassVar, TypeVar

from tilefoundry import dispatch, services, utils
from tilefoundry.ir import expr, types

from .analyzer import Analyzer
from .facts import (
    FactsT,
    MemoryExecutionFacts,
    MemoryHierarchyFacts,
    TopologyFacts,
    facts_result,
)

T = TypeVar("T")
bindings: dispatch.Registry[tuple[type, type], type] = dispatch.Registry("target bindings")


def bind(target: type[Target], base: type) -> Callable[[type[T]], type[T]]:
    """Bind a Target's implementation of a base class, refusing a second owner."""

    def register(implementation: type[T]) -> type[T]:
        if not issubclass(implementation, base):
            raise TypeError(f"{implementation.__name__} must inherit {base.__name__}")
        key = (target, base)
        if bindings.has(key):
            raise ValueError(
                f"{dispatch.spelled(target)}: {base.__name__} already bound to "
                f"{bindings.require(key).__name__}"
            )
        bindings.register(key, implementation)
        return implementation

    return register


def exporting_module(cls: type) -> str:
    """The shortest package on the path of *cls* that exports it by name."""
    parts = cls.__module__.split(".")
    for size in range(1, len(parts) + 1):
        name = ".".join(parts[:size])
        module = sys.modules.get(name)
        if module is not None and vars(module).get(cls.__name__) is cls:
            return name
    return cls.__module__


def value_to_python(value: object) -> utils.PythonExpr:
    """Render a Target value as a Python expression with its imports."""
    if isinstance(value, expr.DType):
        return utils.PythonExpr(("from tilefoundry.ir.expr import DType",), f"DType.{value.name}")
    if isinstance(value, tuple):
        values = tuple(value_to_python(item) for item in value)
        return utils.PythonExpr(
            utils.merge_imports(*(item.imports for item in values)),
            "(" + ", ".join(item.text for item in values) + ("," if len(values) == 1 else "") + ")",
        )
    if isinstance(value, (str, int, float, bool, type(None))):
        return utils.PythonExpr((), repr(value))
    if is_dataclass(value):
        module = exporting_module(type(value))
        rendered = tuple(
            (field.name, value_to_python(getattr(value, field.name)))
            for field in fields(value)
            if field.init or field.compare
        )
        return utils.PythonExpr(
            utils.merge_imports((f"from {module} import {type(value).__name__}",), *(item.imports for _, item in rendered)),
            f"{type(value).__name__}(" + ", ".join(f"{name}={item.text}" for name, item in rendered) + ")",
        )
    raise TypeError(f"cannot render {type(value).__name__} as target Python")


def summary(target: object) -> str:
    """The stable Target identity suitable for user-facing diagnostics."""
    target_type = type(target)
    label = dispatch.spelled(target_type)
    if label != target_type.__name__:
        return label
    return f"{label} ({getattr(target_type, 'name', '<unregistered>')})"


class UnsupportedCapabilityError(Exception):
    """A Target cannot provide the capability a compiler operation requested."""


class Architecture:
    """Base value for compilation architecture facts."""

    name: str
    max_threads_per_cta: int

    def to_python(self) -> utils.PythonExpr:
        """Render this value as a Python expression."""
        return value_to_python(self)


class Device:
    """Base value for one concrete device's resource facts."""

    name: str
    sm_count: int

    def to_python(self) -> utils.PythonExpr:
        """Render this value as a Python expression."""
        return value_to_python(self)


@dataclass(frozen=True)
class Target:
    """Identify a compilation backend.

    A target is a value: what it knows is its hardware, and the only way an
    algorithm reads that is by naming the facts it wants. There is nothing
    mutable on it and nothing registered against it, so two equal targets are
    interchangeable everywhere.
    """

    name: ClassVar[str]

    def torch_device_type(self) -> str:
        """The torch device type holding this program's tensor arguments."""
        return "cpu"

    @cached_property
    def storage_owners(self) -> expr.StorageOwners:
        """Project storage owners once from this target's immutable facts."""
        try:
            hierarchy = self.get_facts(MemoryHierarchyFacts)
        except UnsupportedCapabilityError:
            return expr.StorageOwners((), {})
        order = tuple(item.name for item in self.get_facts(TopologyFacts).topologies)
        owners = {}
        for item in hierarchy.explicit_levels:
            if item.owner != expr.TARGET_MEMORY_OWNER and (not item.owner or item.owner not in order):
                raise ValueError(
                    f"storage {item.name!r}: memory owner {item.owner!r} is not a target topology in {order}"
                )
            owners[expr.resolve_storage(item.name)] = item.owner
        return expr.StorageOwners(order, MappingProxyType(owners))

    @property
    def identity(self) -> str:
        """The stable identity of this concrete Target value."""
        return self.name

    @classmethod
    def from_installed(cls) -> tuple[Target, ...]:
        """Return the values this Target class can currently construct."""
        return (cls(),)

    def to_python(self) -> utils.PythonExpr:
        """Render this Target as a Python expression."""
        return value_to_python(self)

    def get_analyzer(self, selector: str) -> Analyzer:
        """Return the analysis service selected by this concrete Target."""
        analyzer = services.builtin_analyzer(selector)
        if analyzer is not None:
            return analyzer
        raise UnsupportedCapabilityError(f"{summary(self)}: no analyzer for {selector!r}")

    def resolve(self, base: type[T], *args) -> T:
        """Create the bound implementation, or the base class's default."""
        return bindings.get((type(self), base), base).create(self, *args)

    def get_facts(self, facts_type: type[FactsT], query: object | None = None) -> FactsT:
        """Return one immutable hardware-facts aggregate for this Target.

        A backend that states no topology levels runs one program per call, so
        the empty level list is the default rather than a missing projection.
        """
        if facts_type is TopologyFacts and query is None:
            return facts_result(self, facts_type, TopologyFacts(()))
        if facts_type is MemoryExecutionFacts and query is None:
            return facts_result(self, facts_type, MemoryExecutionFacts(()))
        raise UnsupportedCapabilityError(
            f"{summary(self)}: no Facts projection for "
            f"{getattr(facts_type, '__name__', facts_type)!r}"
        )

    def validate_program_topology(self, topology: types.Topology) -> None:
        """Validate one declared topology against this Target's Facts."""
        described = summary(self)
        levels = self.get_facts(TopologyFacts).topologies
        limits = {level.name: level.max_logical_units for level in levels}
        if topology.name not in limits:
            raise ValueError(
                f"{described}: unsupported topology level {topology.name!r}; "
                f"supported levels are {tuple(limits)}"
            )
        limit = limits[topology.name]
        if isinstance(topology.size, bool):
            raise ValueError(
                f"{described}: topology {topology.name!r} extent {topology.size!r} "
                "must be positive"
            )
        if isinstance(topology.size, int) and topology.size < 1:
            raise ValueError(
                f"{described}: topology {topology.name!r} extent {topology.size} "
                "must be positive"
            )
        if isinstance(topology.size, int) and limit is not None and topology.size > limit:
            raise ValueError(
                f"{described}: topology {topology.name!r} extent {topology.size} "
                f"must satisfy 1 <= extent <= {limit}"
            )


def instance(value: object, *, subject: str = "target") -> Target:
    """Return a constructed Target value or raise the authored-boundary error."""
    if isinstance(value, Target):
        return value
    if isinstance(value, str):
        raise TypeError(
            f"{subject} must be a Target instance, not a string; import and "
            "construct the Target class explicitly"
        )
    raise TypeError(f"{subject} must be a Target instance, got {type(value).__name__}")


__all__ = [
    "Architecture",
    "Device",
    "Target",
    "UnsupportedCapabilityError",
    "bind",
    "bindings",
    "instance",
    "summary",
]
