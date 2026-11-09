"""Source-attributed hardware specifications."""

from __future__ import annotations

from .envelope import (
    Document,
    DocumentFormatError,
    DuplicateRegistrationError,
    EvidenceFormatError,
    Fact,
    IncompatiblePairError,
    ResolvedResource,
    SchemaValidationError,
    SpecError,
    UnknownDocumentError,
    UnknownSchemaError,
    parse_document,
)
from .schema import SchemaReader
from .spec import documents, format_capabilities
