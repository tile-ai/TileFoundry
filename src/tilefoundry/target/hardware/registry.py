"""The installed hardware-document registry and its typed schema bindings.

A target package registers two things as an import side effect: the documents
it installs, under stable IDs, and the typed schema that turns one of those
documents into an immutable runtime value. Resolution is by exact ID; there is
no search path and no overlay.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from tilefoundry.target.hardware.envelope import (
    DuplicateRegistrationError,
    HardwareDocument,
    IncompatiblePairError,
    UnknownDocumentError,
    UnknownSchemaError,
    parse_document,
)

# A schema turns one validated document into the immutable value a target uses.
SchemaBuilder = Callable[[HardwareDocument], Any]


@dataclass(frozen=True)
class ResolvedResource:
    """One immutable hardware value plus the provenance of the document it
    came from. The ID and digest travel with the value so a compiled artifact
    can name the exact resources it was built against.

    A value supplied directly as a Python object rather than selected from the
    installed namespace carries no ID, digest, or document: it is a distinct
    hardware value, not a revision of an installed one.
    """

    value: Any
    id: str | None = None
    digest: str | None = None
    document: HardwareDocument | None = None


class HardwareSpecRegistry:
    """Installed hardware documents and the typed schemas that build them."""

    def __init__(self) -> None:
        self._documents: dict[str, tuple[str, str]] = {}
        self._schemas: dict[str, SchemaBuilder] = {}
        self._cache: dict[str, ResolvedResource] = {}

    def install(self, spec_id: str, package: str, resource: str) -> None:
        """Install the document *resource* of *package* under *spec_id*."""
        if spec_id in self._documents:
            raise DuplicateRegistrationError(
                f"hardware document {spec_id!r} is already installed"
            )
        self._documents[spec_id] = (package, resource)

    def register_schema(self, name: str, builder: SchemaBuilder) -> None:
        """Bind the typed schema *builder* to a document ``schema`` *name*."""
        if name in self._schemas:
            raise DuplicateRegistrationError(
                f"hardware schema {name!r} is already registered"
            )
        self._schemas[name] = builder

    def installed_ids(self) -> tuple[str, ...]:
        """Every installed document ID, in sorted order."""
        return tuple(sorted(self._documents))

    def document(self, spec_id: str) -> HardwareDocument:
        """The parsed document installed under *spec_id*."""
        try:
            package, resource = self._documents[spec_id]
        except KeyError:
            raise UnknownDocumentError(
                f"no installed hardware document {spec_id!r}; "
                f"installed: {list(self.installed_ids())}"
            ) from None
        text = files(package).joinpath(resource).read_text(encoding="utf-8")
        document = parse_document(text, origin_label=spec_id)
        if document.id != spec_id:
            raise UnknownDocumentError(
                f"hardware document installed as {spec_id!r} declares "
                f"id={document.id!r}"
            )
        return document

    def _build(self, document: HardwareDocument) -> ResolvedResource:
        """Run *document* through its typed schema."""
        try:
            builder = self._schemas[document.schema]
        except KeyError:
            raise UnknownSchemaError(
                f"{document.id}: no registered schema {document.schema!r}; "
                f"registered: {sorted(self._schemas)}"
            ) from None
        return ResolvedResource(
            value=builder(document),
            id=document.id,
            digest=document.digest,
            document=document,
        )

    def resolve(self, spec_id: str) -> ResolvedResource:
        """The immutable value installed under *spec_id*, built once."""
        cached = self._cache.get(spec_id)
        if cached is None:
            cached = self._build(self.document(spec_id))
            self._cache[spec_id] = cached
        return cached

    def load_path(self, path: str | Path) -> ResolvedResource:
        """Build a complete document read from an explicit filesystem *path*.

        A document loaded this way is never entered into the installed-ID
        namespace, so it cannot shadow or replace an installed resource.
        """
        location = Path(path)
        text = location.read_text(encoding="utf-8")
        return self._build(parse_document(text, origin_label=str(location)))


HARDWARE_SPECS = HardwareSpecRegistry()


def select(
    value: Any, base_type: type, *, role: str, registry: HardwareSpecRegistry | None = None
) -> ResolvedResource:
    """Resolve *value*, which is either an installed ID or a concrete value.

    Selecting a different installed document is not the same as overriding a
    recorded number: the first names another complete, attributed resource,
    while the second would leave the value with no document behind it.
    """
    into = HARDWARE_SPECS if registry is None else registry
    if isinstance(value, str):
        resolved = into.resolve(value)
        if not isinstance(resolved.value, base_type):
            raise UnknownDocumentError(
                f"{role} {value!r} builds a {type(resolved.value).__name__}, "
                f"not a {base_type.__name__}"
            )
        return resolved
    if isinstance(value, base_type):
        return ResolvedResource(value=value)
    raise TypeError(
        f"{role} must be an installed ID string or a {base_type.__name__}, "
        f"got {type(value).__name__}"
    )


def check_compatible(architecture: ResolvedResource, device: ResolvedResource) -> None:
    """Require that an architecture and a device declare each other usable.

    Compatibility is declared, never inferred: a pair composes only when at
    least one side names the other and neither side excludes it.
    """
    if architecture.document.kind != "architecture":
        raise IncompatiblePairError(
            f"{architecture.id}: expected an architecture document, "
            f"got kind {architecture.document.kind!r}"
        )
    if device.document.kind != "device":
        raise IncompatiblePairError(
            f"{device.id}: expected a device document, "
            f"got kind {device.document.kind!r}"
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
            f"compatible; one side must name the other"
        )


__all__ = [
    "HARDWARE_SPECS",
    "HardwareSpecRegistry",
    "ResolvedResource",
    "SchemaBuilder",
    "check_compatible",
    "select",
]
