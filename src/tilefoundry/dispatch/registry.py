"""One key-to-handler map.

A registry answers one question; what the key is depends on that question.
Registration happens at import; a key with no entry raises
``MissingHandlerError`` where it is looked up.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["MissingHandlerError", "Registry", "spelled"]


def spelled(key: object) -> str:
    """A key as a message names it: a class by its name, a tuple by its parts.

    A class named ``Target`` that carries a string ``name`` is written ``<name>.Target``,
    because every backend names its Target class the same way inside its own package.
    """
    if isinstance(key, tuple):
        return "(" + ", ".join(spelled(part) for part in key) + ")"
    if isinstance(key, type):
        owner = getattr(key, "name", None) if key.__name__ == "Target" else None
        return f"{owner}.Target" if isinstance(owner, str) else key.__name__
    return getattr(key, "__qualname__", str(key)) if callable(key) else str(key)


class MissingHandlerError(LookupError):
    """A registry has no entry for a key that a caller requires."""


class Registry[Key, Value]:
    """Key-to-handler map. A second registration of one key raises."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.entries: dict[Key, Value] = {}

    def register(self, key: Key, value: Value) -> None:
        """Add the entry for *key*."""
        if key in self.entries:
            raise RuntimeError(f"{self.name}: {spelled(key)} already registered")
        self.entries[key] = value

    def decorator(self) -> Callable[[Key], Callable[[Value], Value]]:
        """The ``register(key)`` decorator factory of this registry."""
        return self.register_for

    def register_for(self, key: Key) -> Callable[[Value], Value]:
        """Decorator that registers the decorated value under *key*."""

        def decorate(value: Value) -> Value:
            self.register(key, value)
            return value

        return decorate

    def require(self, key: Key) -> Value:
        """The entry for *key*; raises ``MissingHandlerError`` when absent."""
        try:
            return self.entries[key]
        except KeyError:
            raise MissingHandlerError(f"{self.name}: no handler for {spelled(key)}") from None

    def get(self, key: Key, default: Value | None) -> Value | None:
        """The entry for *key*, or the explicitly given *default*."""
        return self.entries.get(key, default)

    def has(self, key: Key) -> bool:
        """Whether *key* has an entry."""
        return key in self.entries
