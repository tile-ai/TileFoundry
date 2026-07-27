"""The shared exact-key algorithm registry.

Analyze and Schedule select algorithms the same way, so they share one
implementation with separate instances rather than growing two differently
shaped dispatch systems.

Lookup is by the exact `(Target concrete type, selector)` pair. A
target-independent algorithm is still registered explicitly for each supported
target, which makes the support matrix something you can read off the
registrations instead of inferring from an inheritance chain.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

AlgorithmT = TypeVar("AlgorithmT")


class AlgorithmRegistryError(Exception):
    """Base for every algorithm registration or resolution diagnostic."""


class UnknownAlgorithmError(AlgorithmRegistryError):
    """No algorithm is registered for this exact target and selector."""


class DuplicateAlgorithmError(AlgorithmRegistryError):
    """An algorithm for this exact target and selector already exists."""


class AlgorithmRegistry(Generic[AlgorithmT]):
    """Algorithms keyed by the exact `(Target concrete type, selector)` pair.

    *kind* names what this instance dispatches, so a diagnostic says which
    registry the caller missed rather than just reporting an unknown selector.
    """

    def __init__(self, kind: str) -> None:
        if not kind:
            raise AlgorithmRegistryError("an algorithm registry needs a kind name")
        self.kind = kind
        self._table: dict[tuple[type, str], AlgorithmT] = {}

    def register(
        self, target_type: type, selector: str, algorithm: AlgorithmT
    ) -> None:
        """Bind *algorithm* to the exact *target_type* and *selector* pair."""
        if not isinstance(target_type, type):
            raise AlgorithmRegistryError(
                f"{self.kind}: target type must be a class, got "
                f"{type(target_type).__name__}"
            )
        if not isinstance(selector, str) or not selector:
            raise AlgorithmRegistryError(
                f"{self.kind}: selector must be a non-empty string, got {selector!r}"
            )
        key = (target_type, selector)
        if key in self._table:
            raise DuplicateAlgorithmError(
                f"{self.kind}: {selector!r} is already registered for "
                f"{target_type.__name__}"
            )
        self._table[key] = algorithm

    def resolve(self, target: Any, selector: str) -> AlgorithmT:
        """The algorithm bound to *target*'s exact type and *selector*.

        A base-class registration does not serve a subclass: two targets that
        share a base can need different implementations, so inheriting one
        would silently run the wrong algorithm.
        """
        try:
            return self._table[(type(target), selector)]
        except KeyError:
            raise UnknownAlgorithmError(
                f"{self.kind}: no {selector!r} registered for "
                f"{type(target).__name__}; available: "
                f"{list(self.selectors_for(type(target)))}"
            ) from None

    def selectors_for(self, target_type: type) -> tuple[str, ...]:
        """Every selector registered for *target_type*, in sorted order."""
        return tuple(
            sorted(
                selector
                for registered_type, selector in self._table
                if registered_type is target_type
            )
        )

    def registered_pairs(self) -> tuple[tuple[str, str], ...]:
        """Every registered pair by name, in sorted order."""
        return tuple(
            sorted(
                (target_type.__name__, selector)
                for target_type, selector in self._table
            )
        )


__all__ = [
    "AlgorithmRegistry",
    "AlgorithmRegistryError",
    "DuplicateAlgorithmError",
    "UnknownAlgorithmError",
]
