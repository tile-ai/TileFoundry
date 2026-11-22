"""One Target class's hardware documents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .hardware import (
    Document,
    DuplicateRegistrationError,
    ResolvedResource,
    UnknownDocumentError,
    UnknownSchemaError,
    parse_document,
)

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


__all__ = ["HardwareSpec", "SchemaBuilder"]
