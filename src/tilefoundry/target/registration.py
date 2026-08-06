"""Registration of concrete Target classes.

Registration identifies a provider class; it never constructs a Target value.
"""

from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import Mapping, TypeVar

TargetT = TypeVar("TargetT", bound="Target")


_TARGET_CLASSES: dict[str, type[Target]] = {}
_TARGET_PROVIDERS: dict[str, tuple[str, str]] = {}


def _provider_identity(target_type: type[Target]) -> tuple[str, str]:
    return (target_type.__module__, target_type.__qualname__)


def register_target(target_type: type[TargetT]) -> type[TargetT]:
    """Register a concrete Target class under its explicitly declared name.

    Re-importing a provider after reload leaves its one logical registration in
    place. A different provider cannot claim it.
    """
    from tilefoundry.target.base import Target  # noqa: PLC0415

    if not isinstance(target_type, type) or not issubclass(target_type, Target):
        raise TypeError("@register_target expects a Target subclass")
    if target_type is Target or inspect.isabstract(target_type):
        raise TypeError("@register_target expects a concrete Target subclass")
    if "name" not in vars(target_type):
        raise ValueError(
            f"@register_target {target_type.__qualname__}: declare a class-level "
            "non-empty name instead of inheriting one"
        )
    name = target_type.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"@register_target {target_type.__qualname__}: name must be a "
            "non-empty string"
        )
    provider = _provider_identity(target_type)
    existing_provider = _TARGET_PROVIDERS.get(name)
    if existing_provider is not None and existing_provider != provider:
        existing = _TARGET_CLASSES[name]
        raise ValueError(
            f"@register_target: name {name!r} is already owned by "
            f"{existing.__module__}.{existing.__qualname__}; cannot register "
            f"{target_type.__module__}.{target_type.__qualname__}"
        )
    if existing_provider == provider:
        return target_type
    _TARGET_CLASSES[name] = target_type
    _TARGET_PROVIDERS[name] = provider
    return target_type


def registered_targets() -> Mapping[str, type[Target]]:
    """Return a read-only view of registered Target classes by name."""
    return MappingProxyType(_TARGET_CLASSES)


def is_registered_target_type(target_type: type[Target]) -> bool:
    """Whether *target_type* belongs to a registered logical provider."""
    name = getattr(target_type, "name", None)
    registered = _TARGET_CLASSES.get(name)
    return registered is not None and _provider_identity(target_type) == _provider_identity(
        registered
    )


__all__ = [
    "is_registered_target_type",
    "register_target",
    "registered_targets",
]
