"""``Function`` — the implementation base class for a node's body.

Its ``type`` is the signature codegen produced for that function. See
[runtime §1.2](docs/spec/runtime.md#12-runtimefunctionpy).
"""
from __future__ import annotations

import inspect

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


class CompiledFunction(Function):
    """A named entry that follows its owner's current resource and native binding."""

    def __init__(self, owner, name, signature) -> None:
        super().__init__(signature)
        self.owner = owner
        self.__name__ = name
        self.__qualname__ = f"{owner.name}.{name}"
        self.__doc__ = None
        self.__signature__ = inspect.Signature(tuple(
            inspect.Parameter(
                param.name, inspect.Parameter.POSITIONAL_ONLY, annotation=param.type,
                default=None if index >= signature.input_count else inspect.Parameter.empty,
            )
            for index, param in enumerate(signature.params) if param.name not in owner.weights
        ))

    def __call__(self, *args):
        return self.owner.forward(*args)


__all__ = ["CompiledFunction", "Function"]
