"""What an IR ``Type`` is once its function has been compiled to C++.

An IR function returns a value; in C++ the outputs are trailing parameters and
the call carries ids the IR never declared, so the two layers need separate
vocabulary. One class per IR ``Type``, plus ``ScalarSignature`` for a parameter
that exists in C++ only -- having no IR type is what makes it hidden.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from tilefoundry.ir.types import TensorType


@dataclass(frozen=True)
class Signature:
    """One parameter, or one whole call, as C++ spells it."""


@dataclass(frozen=True)
class TensorSignature(Signature):
    """A declared tensor parameter: its name and the IR type it carries.

    Dtype, shape, storage and layout are reached through ``type`` rather than
    restated here.
    """

    name: str
    type: TensorType


@dataclass(frozen=True)
class ScalarSignature(Signature):
    """A parameter with no IR type: named and typed in C++ only."""

    name: str
    ctype: str


@dataclass(frozen=True)
class TupleSignature(Signature):
    """The C++ side of a ``TupleType``: its fields, in order."""

    fields: tuple[Signature, ...] = ()


@dataclass(frozen=True)
class UnitSignature(Signature):
    """The C++ side of ``UnitType``: nothing is passed."""


@dataclass(frozen=True)
class CallableSignature(Signature):
    """One calling convention of one IR function.

    ``params`` are the parameters the IR declares, outputs last; ``leading``
    and ``trailing`` are what the convention adds around them. That data is
    the whole difference between a host entry, a launch shim and a kernel, so
    a fourth convention is a fourth instance, not a fourth subclass.
    """

    name: str
    params: tuple[Signature, ...] = ()
    output_count: int = 0
    leading: tuple[Signature, ...] = ()
    trailing: tuple[Signature, ...] = ()

    @property
    def input_count(self) -> int:
        return len(self.params) - self.output_count

    @property
    def input_params(self) -> tuple[Signature, ...]:
        return self.params[: self.input_count]

    @property
    def output_params(self) -> tuple[Signature, ...]:
        return self.params[self.input_count :]

    @property
    def all_params(self) -> tuple[Signature, ...]:
        """Everything the C++ declaration lists, in call order."""
        return (*self.leading, *self.params, *self.trailing)


LAUNCH_ABI = (
    ScalarSignature("grid_x", "int"),
    ScalarSignature("grid_y", "int"),
    ScalarSignature("grid_z", "int"),
    ScalarSignature("block_x", "int"),
    ScalarSignature("block_y", "int"),
    ScalarSignature("block_z", "int"),
    ScalarSignature("dynamic_smem", "int"),
    ScalarSignature("stream", "void*"),
)
"""What a launch shim is told beyond the kernel's own arguments."""


def _ctype(param: Signature, tensor_ctype: Callable[[TensorSignature], str]) -> str:
    """The C++ type of one parameter; a tensor's depends on the convention."""
    if isinstance(param, ScalarSignature):
        return param.ctype
    if isinstance(param, TensorSignature):
        return tensor_ctype(param)
    raise ValueError(f"{type(param).__name__} has no C++ parameter spelling")


def declare(params: Iterable[Signature], tensor_ctype: Callable[[TensorSignature], str]) -> str:
    """The parameter list a definition writes: a C++ type and a name each."""
    return ", ".join(f"{_ctype(p, tensor_ctype)} {p.name}" for p in params)


def declare_types(
    params: Iterable[Signature], tensor_ctype: Callable[[TensorSignature], str]
) -> str:
    """The same list as a forward declaration writes it: types only."""
    return ", ".join(_ctype(p, tensor_ctype) for p in params)


def tensor_signature_of(var) -> TensorSignature:
    """The C++ side of one declared parameter ``Var``."""
    ty = var.type
    assert isinstance(ty, TensorType), (
        f"param {var.name!r} must be TensorType, got {type(ty).__name__}"
    )
    return TensorSignature(name=var.name, type=ty)


def callable_signature_of(fn) -> CallableSignature:
    """The C++ side of a HIR ``Function``: one signature per declared parameter.

    ``output_count=0`` -- a value-returning implementation, not an entry whose
    outputs are trailing parameters.
    """
    params = tuple(tensor_signature_of(var) for var in fn.params)
    return CallableSignature(name=fn.name, params=params, output_count=0)


__all__ = [
    "LAUNCH_ABI",
    "CallableSignature",
    "ScalarSignature",
    "Signature",
    "TensorSignature",
    "TupleSignature",
    "UnitSignature",
    "callable_signature_of",
    "declare",
    "declare_types",
    "tensor_signature_of",
]
