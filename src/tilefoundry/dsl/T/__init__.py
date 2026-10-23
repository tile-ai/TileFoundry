"""The T dialect surface, resolved by registered namespace and name."""

from __future__ import annotations

from .._namespace import make_dialect

_DIALECT = make_dialect("T")
__getattr__, __dir__ = _DIALECT.__getattr__, _DIALECT.__dir__
