"""``Function`` — the implementation base class for a node's body.

Its ``type`` is the signature codegen produced for that function. See
[runtime §1.2](docs/spec/runtime.md#12-runtimefunctionpy).
"""
from __future__ import annotations

from tilefoundry import codegen


class Function:
    """Implementation base class.

    Implementation base class: a ``CallableSignature`` ``type`` plus a
    subclass-overridden ``__call__`` that takes whatever it needs (weights,
    caches) at construction and returns its value(s) directly.
    """

    def __init__(self, type: codegen.CallableSignature) -> None:
        self.type = type

    def __call__(self, *args):
        raise NotImplementedError(
            f"runtime.Function {self.type.name!r}: subclass must implement __call__()"
        )


__all__ = ["Function"]
