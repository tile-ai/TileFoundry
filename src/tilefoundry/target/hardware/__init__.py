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

_BASE_EXPORTS = {"HardwareSpec", "check_compatible", "select"}
_SPEC_EXPORTS = {"format_capabilities", "hardware_documents"}


def __getattr__(name: str):
    if name in _BASE_EXPORTS:
        return getattr(import_module("tilefoundry.target.base"), name)
    if name in _SPEC_EXPORTS:
        return getattr(import_module("tilefoundry.target.hardware.spec"), name)
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
    "check_compatible",
    "format_capabilities",
    "hardware_documents",
    "parse_document",
    "select",
]
