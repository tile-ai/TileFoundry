"""ABI layer — ``EntryABI`` / ``ParamABI`` and ``RuntimeFunction`` (the
implementation base class). See [runtime §1.1.1](docs/spec/runtime.md#111-runtimefunction)."""
from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import TensorType


@dataclass(frozen=True)
class ParamABI:
    """One parameter of a host-visible entry: its name and IR ``TensorType``."""
    name: str
    type: TensorType


@dataclass(frozen=True)
class EntryABI:
    """Host-visible ABI for a function entry. ``params`` lists ALL parameters
    (inputs + outputs) in declaration order; ``output_count`` is the trailing
    count of output parameters."""
    name: str
    params: tuple[ParamABI, ...]
    output_count: int = 0

    @property
    def input_count(self) -> int:
        return len(self.params) - self.output_count

    @property
    def input_params(self) -> tuple[ParamABI, ...]:
        return self.params[:self.input_count]

    @property
    def output_params(self) -> tuple[ParamABI, ...]:
        return self.params[self.input_count:]


class RuntimeFunction:
    """Implementation base class: an ABI ``type`` plus a subclass-overridden
    ``__call__`` that takes whatever it needs (weights, caches) at
    construction and returns its value(s) directly."""

    def __init__(self, type: EntryABI) -> None:
        self.type = type

    def __call__(self, *args):
        raise NotImplementedError(
            f"RuntimeFunction {self.type.name!r}: subclass must implement __call__()"
        )


def param_abi_of(var) -> ParamABI:
    """One ``ParamABI`` for a declared parameter ``Var``: its name and IR
    ``TensorType``."""
    ty = var.type
    assert isinstance(ty, TensorType), (
        f"param {var.name!r} must be TensorType, got {type(ty).__name__}"
    )
    return ParamABI(name=var.name, type=ty)


def entry_abi_of(fn) -> EntryABI:
    """Derive an ``EntryABI`` for a HIR ``Function``: one ``ParamABI`` per
    parameter, ``output_count=0`` (value-returning, not an out-param entry)."""
    params = tuple(param_abi_of(var) for var in fn.params)
    return EntryABI(name=fn.name, params=params, output_count=0)


__all__ = [
    "EntryABI",
    "ParamABI",
    "RuntimeFunction",
    "entry_abi_of",
    "param_abi_of",
]
