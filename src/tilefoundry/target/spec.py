"""One Target class's hardware documents, and the selection of a resource from them."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tilefoundry import dispatch

from .hardware import (
    Document,
    DuplicateRegistrationError,
    IncompatiblePairError,
    ResolvedResource,
    UnknownDocumentError,
    UnknownSchemaError,
    parse_document,
)
from .registry import catalog
from .target import Device

SchemaBuilder = Callable[[Document], Any]


class HardwareSpec:
    """One Target class's document formats and available products."""

    def __init__(self, package: str, schemas: Mapping[str, SchemaBuilder]) -> None:
        self.package = package
        self.schemas = dict(schemas)
        self.installed: dict[str, Document] = {}
        self.cache: dict[str, ResolvedResource] = {}
        self.adopted: set[str] = set()
        self.scanned = False

    def build(self, document: Document) -> ResolvedResource:
        """Build the typed value of one document through its schema builder."""
        try:
            builder = self.schemas[document.schema]
        except KeyError:
            raise UnknownSchemaError(
                f"{document.id}: unsupported schema {document.schema!r}; "
                f"supported: {sorted(self.schemas)}"
            ) from None
        return ResolvedResource(
            value=builder(document),
            id=document.id,
            digest=document.digest,
            document=document,
        )

    def scan(self) -> None:
        """Read the documents shipped in this Target's package, once."""
        if self.scanned:
            return
        for resource in files(self.package).iterdir():
            if not resource.name.endswith(".toml"):
                continue
            document = parse_document(
                resource.read_text(encoding="utf-8"),
                origin_label=f"{self.package}/{resource.name}",
            )
            self.insert(document)
        self.scanned = True

    def read(self, path: Path) -> ResolvedResource:
        """Build a complete document without adding it to this Target's products."""
        location = Path(path)
        document = parse_document(
            location.read_text(encoding="utf-8"), origin_label=str(location)
        )
        return self.build(document)

    def insert(self, document: Document) -> None:
        """Record one document under its ID, refusing an unknown schema or a repeat."""
        if document.schema not in self.schemas:
            raise UnknownSchemaError(
                f"{document.id}: unsupported schema {document.schema!r}; "
                f"supported: {sorted(self.schemas)}"
            )
        if document.id in self.installed:
            raise DuplicateRegistrationError(
                f"hardware document {document.id!r} is already registered"
            )
        self.installed[document.id] = document

    def adopt(self, document: Document) -> None:
        """Add one document whose schema is owned by this Target."""
        self.scan()
        self.insert(document)
        self.adopted.add(document.id)

    def discard(self, document_id: str) -> Document | None:
        """Remove one adopted document without affecting built-in package data."""
        self.scan()
        if document_id not in self.adopted:
            return None
        self.adopted.remove(document_id)
        self.cache.pop(document_id, None)
        return self.installed.pop(document_id, None)

    def supports_schema(self, schema: str) -> bool:
        """Whether this Target owns the typed builder for *schema*."""
        return schema in self.schemas

    def resolve(self, spec_id: str) -> ResolvedResource:
        """Build one available product by exact document ID."""
        self.scan()
        try:
            document = self.installed[spec_id]
        except KeyError:
            raise UnknownDocumentError(
                f"no hardware document {spec_id!r}; available: {sorted(self.installed)}"
            ) from None
        resolved = self.cache.get(spec_id)
        if resolved is None:
            resolved = self.build(document)
            self.cache[spec_id] = resolved
        return resolved

    def documents(self) -> Mapping[str, Document]:
        """Return this Target's available documents by ID."""
        self.scan()
        return MappingProxyType(self.installed)


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
            for target_type in catalog.registered().values():
                other = getattr(target_type, "hardware", None)
                if other is hardware or not isinstance(other, HardwareSpec):
                    continue
                if value in other.documents():
                    raise UnknownDocumentError(
                        f"{role} {value!r} is a hardware document owned by "
                        f"{dispatch.spelled(target_type)}, not {role.partition('.')[0]}"
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


def available_device_ids(hardware: HardwareSpec) -> tuple[str, ...]:
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


def architecture_of(
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
    "HardwareSpec",
    "SchemaBuilder",
    "architecture_of",
    "available_device_ids",
    "check_compatible",
    "select",
]
