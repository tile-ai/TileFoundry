"""ABI layer — ``EntryABI`` / ``ParamABI`` and ``RuntimeFunction`` (the implementation base class).

ABI layer — ``EntryABI`` / ``ParamABI`` and ``RuntimeFunction`` (the
implementation base class). See [runtime §1.1.1](docs/spec/runtime.md#111-runtimefunction).
"""
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
    """Host-visible ABI for a function entry.

    Host-visible ABI for a function entry. ``params`` lists ALL parameters
    (inputs + outputs) in declaration order; ``output_count`` is the trailing
    count of output parameters.
    """
    name: str
    params: tuple[ParamABI, ...]
    output_count: int = 0
    topologies: tuple[str, ...] = ()
    """The levels the program names, outermost first, as a Placement reads them."""

    places: tuple[str, ...] = ()
    """Levels the host places, whose ids lead the call ahead of ``params``.

    A card cannot read which of the mesh's cards it is, so the caller says so.
    These are not parameters the program declares and never appear in a user's
    signature; they are what the entry needs told before its own arguments.
    """

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
    """Implementation base class.

    Implementation base class: an ABI ``type`` plus a subclass-overridden
    ``__call__`` that takes whatever it needs (weights, caches) at
    construction and returns its value(s) directly.
    """

    def __init__(self, type: EntryABI) -> None:
        self.type = type

    def __call__(self, *args):
        raise NotImplementedError(
            f"RuntimeFunction {self.type.name!r}: subclass must implement __call__()"
        )


def param_abi_of(var) -> ParamABI:
    """One ``ParamABI`` for a declared parameter ``Var``: its name and IR ``TensorType``.

    One ``ParamABI`` for a declared parameter ``Var``: its name and IR
    ``TensorType``.
    """
    ty = var.type
    assert isinstance(ty, TensorType), (
        f"param {var.name!r} must be TensorType, got {type(ty).__name__}"
    )
    return ParamABI(name=var.name, type=ty)


def entry_abi_of(fn) -> EntryABI:
    """Derive an ``EntryABI`` for a HIR ``Function``.

    Derive an ``EntryABI`` for a HIR ``Function``: one ``ParamABI`` per
    parameter, ``output_count=0`` (value-returning, not an out-param entry).
    """
    params = tuple(param_abi_of(var) for var in fn.params)
    return EntryABI(name=fn.name, params=params, output_count=0)


_PLACED_LEVELS = ("gpu",)


def places_of(module) -> tuple[str, ...]:
    """The levels *module*'s topology leaves to the host to place.

    A program names a run of levels ending at the finest one the device runs.
    Anything the device cannot read for itself is placed instead, and its id
    has to arrive with the call rather than out of a register.
    """
    declared = tuple(topology.name for topology in module.effective_topologies())
    return tuple(name for name in declared if name in _PLACED_LEVELS)


__all__ = [
    "EntryABI",
    "places_of",
    "ParamABI",
    "RuntimeFunction",
    "entry_abi_of",
    "param_abi_of",
]
