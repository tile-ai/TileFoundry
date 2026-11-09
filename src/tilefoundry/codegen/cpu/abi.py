"""How a host function is called: the entry it exports and what it takes.

The host entry is what the runtime loads and calls, so it is reached by name
with nothing added around its parameters, and it takes runtime tensors rather
than pointers -- a tensor carries its own extents, so no extra parameter does.
"""

from __future__ import annotations

from tilefoundry.codegen import context, registry, signature
from tilefoundry.ir import core
from tilefoundry.target import cpu

TENSOR_CTYPE = "tvm::ffi::Tensor"
"""What the runtime hands a host entry for a tensor parameter."""

PROGRAM_ID_CTYPE = "long long"
"""What an id the process supplies travels as, wide enough for any mesh."""


@registry.register(cpu.Target, registry.Role.CALLEE, core.PrimFunction)
def called_as_entry(
    fn: core.PrimFunction, symbol: str, program_ids: tuple[signature.ProgramIdSignature, ...]
) -> signature.CallableSignature:
    """A host entry is called by name, and told first what no device can read.

    An id the target states reaches the card through this entry, so the entry
    is where it enters the program: its caller is the process, which is the
    only thing that knows which program of the mesh it is running.
    """
    return signature.CallableSignature(
        name=symbol,
        params=tuple(signature.tensor_signature_of(var) for var in fn.params),
        output_count=fn.output_count,
        leading=program_ids,
    )


@registry.register(cpu.Target, registry.Role.CALLEE, signature.TensorSignature)
def declare_tensor(sig: signature.TensorSignature, ctx: context.EmitContext) -> tuple[str, ...]:
    """A runtime tensor, which carries its own extents, so nothing else declares them."""
    return (f"{TENSOR_CTYPE} {sig.name}",)


@registry.register(cpu.Target, registry.Role.CALLEE, signature.ProgramIdSignature)
def declare_program_id(sig: signature.ProgramIdSignature, ctx: context.EmitContext) -> tuple[str, ...]:
    """The id of a level no device register answers for, passed in by value."""
    return (f"{PROGRAM_ID_CTYPE} {sig.name}",)


__all__ = ["PROGRAM_ID_CTYPE", "TENSOR_CTYPE"]
