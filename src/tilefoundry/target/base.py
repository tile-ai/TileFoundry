"""Immutable compilation target values, class registration, and fact projections."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, TypeVar

from tilefoundry.ir.types.shard import Topology
from tilefoundry.target.facts import TopologyLimitFacts
from tilefoundry.target.hardware.envelope import (
    DuplicateRegistrationError,
    HardwareDocument,
    IncompatiblePairError,
    ResolvedResource,
    UnknownDocumentError,
    UnknownSchemaError,
    parse_document,
)
from tilefoundry.target.services import Analyzer, CodeGenerator, Scheduler
from tilefoundry.utils.python_source import PythonExpr, dataclass_to_python

FactsT = TypeVar("FactsT")
SchemaBuilder = Callable[[HardwareDocument], Any]


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


class HardwareSpec:
    """One Target class's document formats and available products."""

    def __init__(self, package: str, schemas: Mapping[str, SchemaBuilder]) -> None:
        self._package = package
        self._schemas = dict(schemas)
        self._documents: dict[str, HardwareDocument] = {}
        self._cache: dict[str, ResolvedResource] = {}
        self._adopted: set[str] = set()
        self._scanned = False

    def _build(self, document: HardwareDocument) -> ResolvedResource:
        try:
            builder = self._schemas[document.schema]
        except KeyError:
            raise UnknownSchemaError(
                f"{document.id}: unsupported schema {document.schema!r}; "
                f"supported: {sorted(self._schemas)}"
            ) from None
        return ResolvedResource(
            value=builder(document),
            id=document.id,
            digest=document.digest,
            document=document,
        )

    def _scan(self) -> None:
        if self._scanned:
            return
        for resource in files(self._package).iterdir():
            if not resource.name.endswith(".toml"):
                continue
            document = parse_document(
                resource.read_text(encoding="utf-8"),
                origin_label=f"{self._package}/{resource.name}",
            )
            self._add(document)
        self._scanned = True

    def read(self, path: Path) -> ResolvedResource:
        """Build a complete document without adding it to this Target's products."""
        location = Path(path)
        document = parse_document(
            location.read_text(encoding="utf-8"), origin_label=str(location)
        )
        return self._build(document)

    def _add(self, document: HardwareDocument) -> None:
        if document.schema not in self._schemas:
            raise UnknownSchemaError(
                f"{document.id}: unsupported schema {document.schema!r}; "
                f"supported: {sorted(self._schemas)}"
            )
        if document.id in self._documents:
            raise DuplicateRegistrationError(
                f"hardware document {document.id!r} is already registered"
            )
        self._documents[document.id] = document

    def adopt(self, document: HardwareDocument) -> None:
        """Add one document whose schema is owned by this Target."""
        self._scan()
        self._add(document)
        self._adopted.add(document.id)

    def discard(self, document_id: str) -> HardwareDocument | None:
        """Remove one adopted document without affecting built-in package data."""
        self._scan()
        if document_id not in self._adopted:
            return None
        self._adopted.remove(document_id)
        self._cache.pop(document_id, None)
        return self._documents.pop(document_id, None)

    def supports_schema(self, schema: str) -> bool:
        """Whether this Target owns the typed builder for *schema*."""
        return schema in self._schemas

    def resolve(self, spec_id: str) -> ResolvedResource:
        """Build one available product by exact document ID."""
        self._scan()
        try:
            document = self._documents[spec_id]
        except KeyError:
            raise UnknownDocumentError(
                f"no hardware document {spec_id!r}; available: {sorted(self._documents)}"
            ) from None
        resolved = self._cache.get(spec_id)
        if resolved is None:
            resolved = self._build(document)
            self._cache[spec_id] = resolved
        return resolved

    def documents(self) -> Mapping[str, HardwareDocument]:
        """Return this Target's available documents by ID."""
        self._scan()
        return MappingProxyType(self._documents)


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

    @property
    def identity(self) -> str:
        """The stable identity of this concrete Target value."""
        return self.name

    @classmethod
    def available(cls) -> tuple[Target, ...]:
        """Return the values this Target class can currently construct."""
        return (cls(),)

    def _python_import_module(self) -> str:
        return type(self).__module__

    def to_python(self) -> PythonExpr:
        return dataclass_to_python(self, self._python_import_module())

    def get_analyzer(self, selector: str) -> Analyzer:
        """Return the analysis service selected by this concrete Target."""
        from tilefoundry.analysis.registry import builtin_analyzer  # noqa: PLC0415

        analyzer = builtin_analyzer(selector)
        if analyzer is not None:
            return analyzer
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
        if isinstance(topology.size, bool):
            raise ValueError(
                f"{target_summary}: topology {topology.name!r} extent {topology.size!r} "
                "must be positive"
            )
        if isinstance(topology.size, int) and topology.size < 1:
            raise ValueError(
                f"{target_summary}: topology {topology.name!r} extent {topology.size} "
                "must be positive"
            )
        if isinstance(topology.size, int) and limit is not None and topology.size > limit:
            raise ValueError(
                f"{target_summary}: topology {topology.name!r} extent {topology.size} "
                f"must satisfy 1 <= extent <= {limit}"
            )


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


def _unregister_target(target_type: type[Target]) -> None:
    """Remove one exact provider class while undoing a persisted module load."""
    name = target_type.name
    if _TARGET_CLASSES.get(name) is target_type:
        _TARGET_CLASSES.pop(name)
        _TARGET_PROVIDERS.pop(name)


def select(
    value: Any,
    base_type: type,
    *,
    role: str,
    hardware: HardwareSpec,
) -> ResolvedResource:
    """Resolve an ID or path through one Target's spec, or retain a direct value."""
    if isinstance(value, str):
        try:
            resolved = hardware.resolve(value)
        except UnknownDocumentError:
            for target_type in registered_targets().values():
                other = getattr(target_type, "hardware", None)
                if other is hardware or not isinstance(other, HardwareSpec):
                    continue
                if value in other.documents():
                    raise UnknownDocumentError(
                        f"{role} {value!r} is a hardware document owned by "
                        f"{target_type.__name__}, not {role.partition('.')[0]}"
                    ) from None
            raise
    elif isinstance(value, Path):
        resolved = hardware.read(value)
    elif isinstance(value, base_type):
        return ResolvedResource(value=value)
    else:
        article = "an" if base_type.__name__[0].lower() in "aeiou" else "a"
        raise TypeError(
            f"{role} must be an installed ID string, a document path, or "
            f"{article} {base_type.__name__}; got {type(value).__name__}"
        )
    if not isinstance(resolved.value, base_type):
        raise UnknownDocumentError(
            f"{role} got hardware document {resolved.id!r}, which builds "
            f"{type(resolved.value).__name__}; expected {base_type.__name__}"
        )
    return resolved


def _available_device_ids(hardware: HardwareSpec) -> tuple[str, ...]:
    """Return devices whose sole compatible architecture is available."""
    documents = hardware.documents()
    return tuple(
        document.id
        for document in sorted(documents.values(), key=lambda item: item.id)
        if document.kind == "device"
        and len(document.compatibility) == 1
        and document.compatibility[0] in documents
        and documents[document.compatibility[0]].kind == "architecture"
    )


def _architecture_of(
    device: Any,
    *,
    device_type: type[Device],
    role: str,
    hardware: HardwareSpec,
) -> str:
    """Return the sole architecture declared by one device document."""
    target_name = role.partition(".")[0]
    if isinstance(device, device_type):
        raise ValueError(
            f"{target_name}: a {device_type.__name__} supplied directly carries "
            "no document to read a compatible architecture from; name the "
            "architecture as well"
        )
    architectures = select(
        device, device_type, role=role, hardware=hardware
    ).document.compatibility
    if len(architectures) != 1:
        raise IncompatiblePairError(
            f"device {device!r} declares {list(architectures)} as compatible "
            "architectures; name the one to build against"
        )
    return architectures[0]


def check_compatible(
    architecture: ResolvedResource, device: ResolvedResource
) -> None:
    """Require that an architecture and device declare each other usable."""
    if architecture.document is None or device.document is None:
        raise IncompatiblePairError("compatibility requires two hardware documents")
    if architecture.document.kind != "architecture":
        raise IncompatiblePairError(
            f"{architecture.id}: expected an architecture document, "
            f"got kind {architecture.document.kind!r}"
        )
    if device.document.kind != "device":
        raise IncompatiblePairError(
            f"{device.id}: expected a device document, got kind {device.document.kind!r}"
        )
    device_allows = device.document.compatibility
    architecture_allows = architecture.document.compatibility
    if device_allows and architecture.id not in device_allows:
        raise IncompatiblePairError(
            f"device {device.id!r} declares compatibility with "
            f"{list(device_allows)}, not {architecture.id!r}"
        )
    if architecture_allows and device.id not in architecture_allows:
        raise IncompatiblePairError(
            f"architecture {architecture.id!r} declares compatibility with "
            f"{list(architecture_allows)}, not {device.id!r}"
        )
    if not device_allows and not architecture_allows:
        raise IncompatiblePairError(
            f"neither {architecture.id!r} nor {device.id!r} declares the pair "
            "compatible; one side must name the other"
        )


__all__ = [
    "Architecture",
    "Device",
    "HardwareSpec",
    "Target",
    "UnsupportedCapabilityError",
    "check_compatible",
    "register_target",
    "registered_targets",
    "select",
    "target_instance",
]
