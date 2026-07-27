"""Generic hardware-document envelope: parsing, evidence leaves, digests.

This layer fixes only the document envelope and the shape of an evidence leaf.
The namespace below ``facts`` belongs to the target package named by the
document's ``schema``, which validates exact paths, types, units, and
cross-field invariants of its own tree.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DOCUMENT_KINDS = ("architecture", "device")

FACT_EVIDENCE_KEYS = ("unit", "origin", "source", "conditions")

# How a value was obtained. ``vendor`` is a figure the vendor publishes,
# ``measured`` one taken on the described host, ``reference`` one cited from a
# third party, ``derived`` one computed from other facts, and ``estimated`` a
# reading that no source states outright. The last two state how in
# ``conditions``, so an inference is never mistaken for a published number.
FACT_ORIGINS = ("vendor", "measured", "reference", "derived", "estimated")

_ENVELOPE_KEYS = ("schema", "kind", "id")


class HardwareSpecError(Exception):
    """Base for every hardware-document diagnostic."""


class DocumentFormatError(HardwareSpecError):
    """The document is not readable TOML or its envelope is malformed."""


class EvidenceFormatError(HardwareSpecError):
    """A leaf under ``facts`` is not a well-formed evidence record."""


class SchemaValidationError(HardwareSpecError):
    """A document's fact tree does not satisfy its target-owned schema."""


class UnknownDocumentError(HardwareSpecError):
    """No document is installed under the requested ID."""


class UnknownSchemaError(HardwareSpecError):
    """No typed schema is registered under the name a document claims."""


class DuplicateRegistrationError(HardwareSpecError):
    """A document ID or schema name was registered twice."""


class IncompatiblePairError(HardwareSpecError):
    """An architecture and a device do not declare each other compatible."""


@dataclass(frozen=True)
class Fact:
    """One leaf of a hardware document together with its evidence.

    An available leaf carries ``value``; an unavailable one carries
    ``status`` instead and states why in ``conditions``. The two are
    exclusive, so no caller ever reads a placeholder string as a number.
    """

    path: str
    value: Any = None
    unit: str | None = None
    origin: str | None = None
    source: str | None = None
    conditions: str | None = None
    status: str | None = None

    @property
    def available(self) -> bool:
        """Whether this leaf records a usable value."""
        return self.status is None


@dataclass(frozen=True)
class HardwareDocument:
    """One parsed architecture or device document.

    ``facts`` is keyed by dotted leaf path (``memory.hbm.bandwidth``), which is
    what a typed schema validates against and what the evidence sidecar is
    keyed by. ``digest`` covers the document's exact content, so any edit to a
    recorded fact or its evidence changes it.
    """

    id: str
    kind: str
    schema: str
    compatibility: tuple[str, ...]
    facts: Mapping[str, Fact]
    digest: str

    def fact(self, path: str) -> Fact:
        """The leaf at *path*, which must exist in this document."""
        try:
            return self.facts[path]
        except KeyError:
            raise SchemaValidationError(
                f"{self.id}: no fact at path {path!r}"
            ) from None


def _leaf_from(path: str, table: dict[str, Any]) -> Fact:
    """One evidence leaf, validated against the generic leaf format."""
    has_value = "value" in table
    status = table.get("status")
    if has_value and status is not None:
        raise EvidenceFormatError(
            f"fact {path!r}: a leaf records either 'value' or 'status', not both"
        )
    if not has_value and status is None:
        raise EvidenceFormatError(
            f"fact {path!r}: a leaf must record 'value', or 'status' when the "
            f"value is unavailable"
        )
    if status is not None and status != "unavailable":
        raise EvidenceFormatError(
            f"fact {path!r}: unknown status {status!r}, expected 'unavailable'"
        )
    if status is not None and not table.get("conditions"):
        raise EvidenceFormatError(
            f"fact {path!r}: an unavailable leaf must state why in 'conditions'"
        )
    origin = table.get("origin")
    if has_value and origin not in FACT_ORIGINS:
        raise EvidenceFormatError(
            f"fact {path!r}: origin must be one of {list(FACT_ORIGINS)}, "
            f"got {origin!r}"
        )
    unknown = set(table) - {"value", "status", *FACT_EVIDENCE_KEYS}
    if unknown:
        raise EvidenceFormatError(
            f"fact {path!r}: unknown evidence keys {sorted(unknown)}"
        )
    return Fact(
        path=path,
        value=table.get("value"),
        unit=table.get("unit"),
        origin=origin,
        source=table.get("source"),
        conditions=table.get("conditions"),
        status=status,
    )


def _is_leaf(table: dict[str, Any]) -> bool:
    """Whether a table under ``facts`` terminates a path rather than nesting."""
    return "value" in table or "status" in table


def _walk_facts(tree: dict[str, Any], prefix: str = "") -> dict[str, Fact]:
    """Every evidence leaf under a freely nested ``facts`` namespace."""
    leaves: dict[str, Fact] = {}
    for key, node in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if not isinstance(node, dict):
            raise EvidenceFormatError(
                f"fact {path!r}: expected a table, got {type(node).__name__}"
            )
        if _is_leaf(node):
            nested = sorted(
                key for key, value in node.items() if isinstance(value, dict)
            )
            if nested:
                raise EvidenceFormatError(
                    f"fact {path!r}: a leaf cannot also be a namespace; it records "
                    f"a value but nests {nested}"
                )
            leaves[path] = _leaf_from(path, node)
        else:
            leaves.update(_walk_facts(node, path))
    return leaves


def parse_document(text: str, *, origin_label: str) -> HardwareDocument:
    """Parse one hardware document from TOML *text*.

    *origin_label* names the document in diagnostics: an installed ID for a
    registered document, or a filesystem path for one loaded explicitly.
    """
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise DocumentFormatError(f"{origin_label}: not readable TOML: {exc}") from exc

    envelope = raw.get("spec")
    if not isinstance(envelope, dict):
        raise DocumentFormatError(
            f"{origin_label}: missing the [spec] envelope table"
        )
    missing = [key for key in _ENVELOPE_KEYS if not envelope.get(key)]
    if missing:
        raise DocumentFormatError(
            f"{origin_label}: [spec] is missing {missing}"
        )
    unknown_envelope = set(envelope) - set(_ENVELOPE_KEYS)
    if unknown_envelope:
        raise DocumentFormatError(
            f"{origin_label}: unknown [spec] keys {sorted(unknown_envelope)}"
        )
    kind = envelope["kind"]
    if kind not in DOCUMENT_KINDS:
        raise DocumentFormatError(
            f"{origin_label}: kind must be one of {list(DOCUMENT_KINDS)}, "
            f"got {kind!r}"
        )

    compatibility_table = raw.get("compatibility", {})
    if not isinstance(compatibility_table, dict):
        raise DocumentFormatError(
            f"{origin_label}: [compatibility] must be a table"
        )
    peer_key = "devices" if kind == "architecture" else "architectures"
    unknown_compat = set(compatibility_table) - {peer_key}
    if unknown_compat:
        raise DocumentFormatError(
            f"{origin_label}: a {kind} document declares compatibility under "
            f"{peer_key!r}, got unknown {sorted(unknown_compat)}"
        )
    declared_peers = compatibility_table.get(peer_key, [])
    # A bare string is iterable, so an unchecked tuple() would silently turn one
    # ID into a tuple of its characters.
    if not isinstance(declared_peers, list) or not all(
        isinstance(peer, str) and peer for peer in declared_peers
    ):
        raise DocumentFormatError(
            f"{origin_label}: [compatibility] {peer_key} must be a list of "
            f"non-empty ID strings, got {declared_peers!r}"
        )
    compatibility = tuple(declared_peers)

    facts_tree = raw.get("facts", {})
    if not isinstance(facts_tree, dict):
        raise DocumentFormatError(f"{origin_label}: [facts] must be a table")
    unknown_top = set(raw) - {"spec", "compatibility", "facts"}
    if unknown_top:
        raise DocumentFormatError(
            f"{origin_label}: unknown top-level tables {sorted(unknown_top)}"
        )

    return HardwareDocument(
        id=envelope["id"],
        kind=kind,
        schema=envelope["schema"],
        compatibility=compatibility,
        facts=_walk_facts(facts_tree),
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "DOCUMENT_KINDS",
    "FACT_ORIGINS",
    "DocumentFormatError",
    "DuplicateRegistrationError",
    "EvidenceFormatError",
    "Fact",
    "HardwareDocument",
    "HardwareSpecError",
    "IncompatiblePairError",
    "SchemaValidationError",
    "UnknownDocumentError",
    "UnknownSchemaError",
    "parse_document",
]
