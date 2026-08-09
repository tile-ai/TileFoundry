"""Source-attributed target hardware specifications."""

from __future__ import annotations

from importlib import import_module

from tilefoundry.target.hardware.envelope import (
    DocumentFormatError,
    DuplicateRegistrationError,
    EvidenceFormatError,
    Fact,
    HardwareDocument,
    HardwareSpecError,
    IncompatiblePairError,
    ResolvedResource,
    SchemaValidationError,
    UnknownDocumentError,
    UnknownSchemaError,
    parse_document,
)
from tilefoundry.target.hardware.schema import SchemaReader
from tilefoundry.target.hardware.spec import format_capabilities, hardware_documents


def __getattr__(name: str):
    if name == "HardwareSpec":
        return import_module("tilefoundry.target.base").HardwareSpec
    raise AttributeError(name)


__all__ = [
    "DocumentFormatError",
    "DuplicateRegistrationError",
    "EvidenceFormatError",
    "Fact",
    "HardwareDocument",
    "HardwareSpecError",
    "HardwareSpec",
    "IncompatiblePairError",
    "ResolvedResource",
    "SchemaReader",
    "SchemaValidationError",
    "UnknownDocumentError",
    "UnknownSchemaError",
    "format_capabilities",
    "hardware_documents",
    "parse_document",
]
