"""What an IR ``Type`` is once its function has been compiled to C++.

An IR function returns a value; in C++ the outputs are trailing parameters and
the call carries ids the IR never declared, so the two layers need separate
vocabulary. One class per IR ``Type``, and one per parameter a convention adds
of its own -- a kind is what a caller dispatches on, so nothing has to read a
name and guess what it stands for.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.target import Target
from tilefoundry.target.facts import TopologyFacts
from tilefoundry.visitor_registry.registries import Role, codegen_registry, spelled


@dataclass(frozen=True)
class Signature:
    """One parameter, or one whole call, as C++ spells it."""


@dataclass(frozen=True)
class TensorSignature(Signature):
    """A declared tensor: its pointer, and the extents its type leaves open.

    Dtype, shape, storage and layout are reached through ``type`` rather than
    restated here.
    """

    name: str
    type: TensorType

    @property
    def dynamic_axes(self) -> tuple[int, ...]:
        """The axes the type states as a ``DimVar``, whose extents travel with the pointer.

        A static extent is a compile-time constant on both sides of the call,
        so only these axes cost the convention anything.
        """
        return tuple(axis for axis, dim in enumerate(self.type.shape) if isinstance(dim, DimVar))

    def extent_name(self, axis: int) -> str:
        """What a declaration calls this tensor's extent along *axis*.

        Invented here and read nowhere else: a caller writes whatever its own
        scope calls that extent, so nothing ever parses this name back.
        """
        return f"{self.name}_shape_{axis}"


@dataclass(frozen=True)
class ProgramIdSignature(Signature):
    """The id of one topology level, told to a program that cannot read it."""

    name: str
    topology_level: str


@dataclass(frozen=True)
class ProgramMetaSignature(Signature):
    """The block of ids a kernel hands to the runtime it was compiled against."""

    name: str


@dataclass(frozen=True)
class LaunchSignature(Signature):
    """One of the geometry arguments a launch is told beyond the kernel's own."""

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
    a fourth convention is a fourth instance, not a fourth subclass. It is a
    list of signatures and nothing more: how many C++ parameters one of them
    becomes is answered where that one is written out.
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
    LaunchSignature("grid_x", "int"),
    LaunchSignature("block_x", "int"),
    LaunchSignature("dynamic_smem", "int"),
)
"""Actual cross-translation-unit scalars on a CPU-to-device Launch edge.

The topology is one-dimensional, so only grid_x and block_x are stated.
Dynamic shared memory remains because its per-Launch value crosses the shim ABI.
"""


def tensor_signature_of(var) -> TensorSignature:
    """The C++ side of one declared parameter ``Var``."""
    ty = var.type
    assert isinstance(ty, TensorType), (
        f"param {var.name!r} must be TensorType, got {type(ty).__name__}"
    )
    return TensorSignature(name=var.name, type=ty)


def program_id_params(module: Module, target: Target) -> tuple[ProgramIdSignature, ...]:
    """One id per level of *module*'s program whose ids *target* states.

    A level the target states has no register the device can read, so its id
    has to arrive with the call. This is the one place the question is asked,
    so a host entry and the shim it calls cannot answer it differently.
    """
    stated = {
        topology_level.name
        for topology_level in target.get_facts(TopologyFacts).topologies
        if topology_level.from_target
    }
    return tuple(
        ProgramIdSignature(
            name=f"tilefoundry_{topology.name}_program_id",
            topology_level=topology.name,
        )
        for topology in module.effective_topologies()
        if topology.name in stated
    )


def called_as(fn, program_ids: tuple[ProgramIdSignature, ...] = ()) -> CallableSignature:
    """How *fn* is called, as *fn*'s own target states it.

    What a symbol is named and what is added around its parameters is the
    target's convention, so the answer comes from the target's handlers and
    a caller of another target reaches it without naming them.
    """
    key = (type(fn.target), Role.CALLEE, PrimFunction)
    convention = codegen_registry.lookup(key)
    if convention is None:
        raise RuntimeError(
            f"codegen: nothing registered for {spelled(key)}, so nothing states "
            f"how {fn.name!r} is called"
        )
    return convention(fn, program_ids)


__all__ = [
    "LAUNCH_ABI",
    "CallableSignature",
    "LaunchSignature",
    "ProgramIdSignature",
    "ProgramMetaSignature",
    "Signature",
    "TensorSignature",
    "TupleSignature",
    "UnitSignature",
    "called_as",
    "program_id_params",
    "tensor_signature_of",
]
