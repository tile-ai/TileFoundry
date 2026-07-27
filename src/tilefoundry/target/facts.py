"""Projection of a Target's hardware specification into algorithm Facts.

An algorithm declares the immutable aggregate it needs and the Target package
registers the conversion that builds it. Lookup is by the exact
``(Target concrete type, Facts type)`` pair: a target-aware algorithm with no
registered conversion for the selected target fails immediately rather than
running against facts that describe different hardware.

This module is generic. It names no concrete target, so adding a backend adds a
registration rather than a branch here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

# A conversion reads the target and an optional algorithm-private query. The
# query type belongs to the requesting algorithm; there is no common base for it.
FactsConversion = Callable[..., Any]


class TargetFactsError(Exception):
    """Base for every Facts registration or projection diagnostic."""


class UnknownFactsConversionError(TargetFactsError):
    """No conversion is registered for this exact target and Facts pair."""


class DuplicateFactsConversionError(TargetFactsError):
    """A conversion for this exact target and Facts pair already exists."""


class InvalidFactsTypeError(TargetFactsError):
    """The requested Facts type is not an immutable aggregate."""


def _require_facts_type(facts_type: type) -> None:
    """Require *facts_type* to be a frozen dataclass.

    An aggregate Facts value is immutable by contract, so the constraint is
    checked once at registration instead of trusted at every projection.
    """
    if not isinstance(facts_type, type):
        raise InvalidFactsTypeError(
            f"Facts type must be a class, got {type(facts_type).__name__}"
        )
    if not dataclasses.is_dataclass(facts_type):
        raise InvalidFactsTypeError(
            f"{facts_type.__name__} must be a dataclass to be a Facts aggregate"
        )
    if not facts_type.__dataclass_params__.frozen:
        raise InvalidFactsTypeError(
            f"{facts_type.__name__} must be a frozen dataclass: an aggregate "
            f"Facts value is immutable"
        )


class TargetFactsRegistry:
    """Conversions from a concrete Target to an algorithm's Facts aggregate."""

    def __init__(self) -> None:
        self._conversions: dict[tuple[type, type], FactsConversion] = {}

    def register(
        self, target_type: type, facts_type: type, conversion: FactsConversion
    ) -> None:
        """Bind *conversion* to the exact *target_type* and *facts_type* pair."""
        if not isinstance(target_type, type):
            raise TargetFactsError(
                f"target type must be a class, got {type(target_type).__name__}"
            )
        _require_facts_type(facts_type)
        if not callable(conversion):
            raise TargetFactsError(
                f"conversion for ({target_type.__name__}, {facts_type.__name__}) "
                f"must be callable, got {type(conversion).__name__}"
            )
        key = (target_type, facts_type)
        if key in self._conversions:
            raise DuplicateFactsConversionError(
                f"a conversion for ({target_type.__name__}, "
                f"{facts_type.__name__}) is already registered"
            )
        self._conversions[key] = conversion

    def registered_pairs(self) -> tuple[tuple[str, str], ...]:
        """Every registered pair by name, in sorted order."""
        return tuple(
            sorted(
                (target_type.__name__, facts_type.__name__)
                for target_type, facts_type in self._conversions
            )
        )

    def project(
        self, target: "Target", facts_type: type, query: Any = None
    ) -> Any:
        """Build *facts_type* from *target*.

        Resolution is by the target's exact concrete type. A base-class
        registration does not serve a subclass: two targets that share a base
        can describe different hardware, so inheriting a conversion would
        silently project the wrong facts.
        """
        _require_facts_type(facts_type)
        key = (type(target), facts_type)
        try:
            conversion = self._conversions[key]
        except KeyError:
            raise UnknownFactsConversionError(
                f"no Facts conversion registered for "
                f"({type(target).__name__}, {facts_type.__name__}); "
                f"registered: {list(self.registered_pairs())}"
            ) from None
        facts = conversion(target, query)
        if not isinstance(facts, facts_type):
            raise TargetFactsError(
                f"the conversion for ({type(target).__name__}, "
                f"{facts_type.__name__}) returned "
                f"{type(facts).__name__}"
            )
        return facts


TARGET_FACTS = TargetFactsRegistry()


def register_target_facts(
    target_type: type, facts_type: type, conversion: FactsConversion
) -> None:
    """Register *conversion* into the shared Facts registry."""
    TARGET_FACTS.register(target_type, facts_type, conversion)


__all__ = [
    "TARGET_FACTS",
    "DuplicateFactsConversionError",
    "FactsConversion",
    "InvalidFactsTypeError",
    "TargetFactsError",
    "TargetFactsRegistry",
    "UnknownFactsConversionError",
    "register_target_facts",
]
