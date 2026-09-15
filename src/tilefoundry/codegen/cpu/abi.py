"""How a host function is called: the entry it exports and what it takes.

The host entry is what the runtime loads and calls, so it is reached by name
with nothing added around its parameters, and it takes runtime tensors rather
than pointers -- a tensor carries its own extents, so no extra parameter does.
"""

from __future__ import annotations

from tilefoundry.codegen.context import EmitContext
from tilefoundry.codegen.signature import (
    CallableSignature,
    ProgramIdSignature,
    TensorSignature,
    tensor_signature_of,
)
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.target import CpuTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen

TENSOR_CTYPE = "tvm::ffi::Tensor"
"""What the runtime hands a host entry for a tensor parameter."""

PROGRAM_ID_CTYPE = "long long"
"""What an id the process supplies travels as, wide enough for any mesh."""


def host_entry(name: str) -> str:
    """The C++ symbol of *name*'s host entry, which ``main`` would collide with."""
    return f"tilefoundry_{name.replace('$', '__')}_host"


@register_codegen(CpuTarget, Role.CALLEE, PrimFunction)
def _called_as_entry(
    fn: PrimFunction, program_ids: tuple[ProgramIdSignature, ...]
) -> CallableSignature:
    """A host entry is called by name, and told first what no device can read.

    An id the target states reaches the card through this entry, so the entry
    is where it enters the program: its caller is the process, which is the
    only thing that knows which program of the mesh it is running.
    """
    return CallableSignature(
        name=host_entry(fn.name),
        params=tuple(tensor_signature_of(var) for var in fn.params),
        output_count=fn.output_count,
        leading=program_ids,
    )


@register_codegen(CpuTarget, Role.CALLEE, TensorSignature)
def _declare_tensor(sig: TensorSignature, ctx: EmitContext) -> tuple[str, ...]:
    """A runtime tensor, which carries its own extents, so nothing else declares them."""
    return (f"{TENSOR_CTYPE} {sig.name}",)


@register_codegen(CpuTarget, Role.CALLEE, ProgramIdSignature)
def _declare_program_id(sig: ProgramIdSignature, ctx: EmitContext) -> tuple[str, ...]:
    """The id of a level no device register answers for, passed in by value."""
    return (f"{PROGRAM_ID_CTYPE} {sig.name}",)


__all__ = ["PROGRAM_ID_CTYPE", "TENSOR_CTYPE", "host_entry"]
