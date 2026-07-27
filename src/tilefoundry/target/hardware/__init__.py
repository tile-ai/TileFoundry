"""Installed, source-attributed target hardware specifications."""

from __future__ import annotations

from tilefoundry.target.hardware.envelope import (
    DocumentFormatError,
    DuplicateRegistrationError,
    EvidenceFormatError,
    Fact,
    HardwareDocument,
    HardwareSpecError,
    IncompatiblePairError,
    SchemaValidationError,
    UnknownDocumentError,
    UnknownSchemaError,
    parse_document,
)
from tilefoundry.target.hardware.registry import (
    HARDWARE_SPECS,
    HardwareSpecRegistry,
    ResolvedResource,
    check_compatible,
    select,
)
from tilefoundry.target.hardware.schema import SchemaReader
from tilefoundry.target.hardware.spec import format_capabilities, hardware_documents

__all__ = [
    "HARDWARE_SPECS",
    "DocumentFormatError",
    "DuplicateRegistrationError",
    "EvidenceFormatError",
    "Fact",
    "HardwareDocument",
    "HardwareSpecError",
    "HardwareSpecRegistry",
    "IncompatiblePairError",
    "ResolvedResource",
    "SchemaReader",
    "SchemaValidationError",
    "UnknownDocumentError",
    "UnknownSchemaError",
    "check_compatible",
    "format_capabilities",
    "hardware_documents",
    "parse_document",
    "select",
]
