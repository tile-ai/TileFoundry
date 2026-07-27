"""Exact-key reading of a hardware document against a typed schema.

A target package drives a reader over the fact paths it knows, then closes it.
Closing is what makes a schema exact: any leaf the document carries that the
schema never read is a spelling mistake or an unused fact, and is reported
rather than silently ignored.
"""

from __future__ import annotations

from tilefoundry.target.hardware.envelope import (
    Fact,
    HardwareDocument,
    SchemaValidationError,
)


class SchemaReader:
    """Read a document's facts, tracking which paths a schema consumed."""

    def __init__(self, document: HardwareDocument) -> None:
        self._document = document
        self._seen: set[str] = set()

    @property
    def document(self) -> HardwareDocument:
        """The document being read."""
        return self._document

    def _leaf(self, path: str) -> Fact:
        self._seen.add(path)
        try:
            return self._document.facts[path]
        except KeyError:
            raise SchemaValidationError(
                f"{self._document.id}: required fact {path!r} is missing"
            ) from None

    def _available(self, path: str, unit: str | None) -> Fact:
        fact = self._leaf(path)
        if not fact.available:
            raise SchemaValidationError(
                f"{self._document.id}: fact {path!r} is unavailable, but the "
                f"schema requires a value"
            )
        if unit is not None and fact.unit != unit:
            raise SchemaValidationError(
                f"{self._document.id}: fact {path!r} must be recorded in "
                f"{unit!r}, got {fact.unit!r}"
            )
        return fact

    def integer(self, path: str, *, unit: str) -> int:
        """A positive integer fact recorded in *unit*."""
        fact = self._available(path, unit)
        value = fact.value
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaValidationError(
                f"{self._document.id}: fact {path!r} must be an integer, got "
                f"{type(value).__name__}"
            )
        if value <= 0:
            raise SchemaValidationError(
                f"{self._document.id}: fact {path!r} must be positive, got {value}"
            )
        return value

    def optional_integer(self, path: str, *, unit: str) -> int | None:
        """A positive integer fact recorded in *unit*, or ``None``.

        The leaf must still be declared, so the document states either the
        number or that no number is available. ``None`` is a recorded absence,
        never a missing key.
        """
        fact = self._leaf(path)
        if not fact.available:
            return None
        return self.integer(path, unit=unit)

    def text(self, path: str) -> str:
        """A string fact, such as an identity name or a geometry."""
        fact = self._available(path, None)
        if not isinstance(fact.value, str) or not fact.value:
            raise SchemaValidationError(
                f"{self._document.id}: fact {path!r} must be a non-empty string, "
                f"got {fact.value!r}"
            )
        return fact.value

    def names(self, path: str) -> tuple[str, ...]:
        """A list-of-names fact, such as an instruction capability set."""
        fact = self._available(path, "name")
        value = fact.value
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise SchemaValidationError(
                f"{self._document.id}: fact {path!r} must be a list of non-empty "
                f"strings, got {value!r}"
            )
        return tuple(value)

    def declared_unavailable(self, path: str) -> None:
        """Consume a leaf the schema knows about but requires no value from."""
        fact = self._leaf(path)
        if fact.available:
            raise SchemaValidationError(
                f"{self._document.id}: fact {path!r} is recorded as available, "
                f"but the schema models it as having no usable value"
            )

    def close(self) -> None:
        """Reject any fact the document carries that the schema never read."""
        unknown = sorted(set(self._document.facts) - self._seen)
        if unknown:
            raise SchemaValidationError(
                f"{self._document.id}: unknown facts for schema "
                f"{self._document.schema!r}: {unknown}"
            )


__all__ = ["SchemaReader"]
