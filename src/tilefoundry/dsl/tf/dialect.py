"""The tf dialect surface, resolved by registered namespace and name."""

from __future__ import annotations

from tilefoundry.dsl import namespace

root = namespace.Dialect.from_name("tf")


def __getattr__(name):
    return root.__getattr__(name)


def __dir__():
    return root.__dir__()


__all__ = ["__dir__", "__getattr__"]
