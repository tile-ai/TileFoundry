from __future__ import annotations

from collections.abc import Callable

from .services import ServiceCollection

ProviderFactory = Callable[[object, str], ServiceCollection]

_PROVIDERS: dict[tuple[type, str], ProviderFactory] = {}


def register_provider(target_type: type, *, level: str, factory: ProviderFactory) -> None:
    if not isinstance(target_type, type):
        raise TypeError("provider target_type must be a concrete type")
    if not level:
        raise ValueError("provider level must not be empty")
    _PROVIDERS[(target_type, level)] = factory


def resolve_provider_services(target: object, level: str) -> ServiceCollection:
    factory = _PROVIDERS.get((type(target), level))
    if factory is None:
        for (target_type, registered_level), candidate in _PROVIDERS.items():
            if registered_level == level and isinstance(target, target_type):
                factory = candidate
                break
    if factory is None:
        raise ValueError(
            f"no target provider registered for {type(target).__name__!r} "
            f"at level {level!r}"
        )
    return factory(target, level)


__all__ = ["register_provider", "resolve_provider_services"]
