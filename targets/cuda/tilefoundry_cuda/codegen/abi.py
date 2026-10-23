"""How a CUDA function is called: the symbols it exports and what it takes.

A caller cannot name a device function, only the launcher that launches it, so
that is what CUDA puts in the symbol table. Every spelling here is CUDA's own
and is registered against this target, so a caller of another target asks for
it without naming this module.
"""

from __future__ import annotations

import re

from tilefoundry.codegen.context import EmitContext
from tilefoundry.codegen.signature import (
    CallableSignature,
    LaunchSignature,
    ProgramIdSignature,
    ProgramMetaSignature,
    Signature,
    TensorSignature,
    tensor_signature_of,
)
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_cuda import CudaTarget
from tilefoundry_cuda.codegen.context import CudaCodegenContext
from tilefoundry_cuda.codegen.tir.stmts.mesh_scope import mesh_type

PROGRAM_ID_CTYPE = "long long"
"""What a level's id travels as, wide enough for any count a target states."""

PROGRAM_META_CTYPE = "tilefoundry::ProgramMetaData"
"""The block of ids a kernel hands to the runtime it was compiled against."""

PROGRAM_META = ProgramMetaSignature(name="tilefoundry_meta")
"""The one block of them a launcher fills in and its kernel takes by value."""

_C_ABI = ("void*", "long long")
"""What a buffer and an extent are called where a host unit has to spell them.

A host translation unit compiles without CUDA's headers, so it cannot name
``__nv_bfloat16``; the launcher takes what plain C states and puts the type back
on inside the unit that has it.
"""


CPP_KEYWORDS = frozenset(
    """
    alignas alignof and and_eq asm atomic_cancel atomic_commit atomic_noexcept auto
    bitand bitor bool break case catch char char8_t char16_t char32_t class compl
    concept const consteval constexpr constinit const_cast continue co_await
    co_return co_yield decltype default delete do double dynamic_cast else enum
    explicit export extern false float for friend goto if inline int long mutable
    namespace new noexcept not not_eq nullptr operator or or_eq private protected
    public reflexpr register reinterpret_cast requires return short signed sizeof
    static static_assert static_cast struct switch synchronized template this
    thread_local throw true try typedef typeid typename union unsigned using
    virtual void volatile wchar_t while xor xor_eq
""".split()
)


def check_parameter_names(path, params) -> None:
    """Check parameter spellings and collisions with every generated view name."""
    occupied = {}
    for param in params:
        name = param.name
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) is None
            or name in CPP_KEYWORDS
            or name.startswith("tilefoundry_")
        ):
            raise ValueError(f"codegen: invalid parameter {name!r} in {path!r}")
        signature = tensor_signature_of(param)
        generated = [f"{name}_tensor"]
        generated += [signature.extent_name(axis) for axis in signature.dynamic_axes]
        if isinstance(param.type.layout, ShardLayout):
            generated.append(f"{name}_mesh")
        for spelling in (name, *generated):
            if spelling in occupied:
                raise ValueError(
                    f"codegen: parameter {name!r} in {path!r} collides at "
                    f"{spelling!r} with parameter {occupied[spelling]!r}"
                )
            occupied[spelling] = name


def launcher_symbol(symbol: str) -> str:
    """The C ABI launcher derived from a resolved mangled symbol."""
    return f"{symbol}_launch"


def kernel_symbol(symbol: str) -> str:
    """The global kernel derived from a resolved mangled symbol."""
    return f"{symbol}_kernel"


def kernel_signature(launcher: CallableSignature, name: str) -> CallableSignature:
    """How the kernel compiled from *name* is called, which is nothing outside this unit.

    It takes *launcher*'s own parameters and, where the launcher was told ids no card
    can read, the one block of them it hands its threads. The launch geometry
    stops at the launcher, which is what spends it.
    """
    return CallableSignature(
        name=kernel_symbol(name),
        params=launcher.params,
        output_count=launcher.output_count,
        leading=(PROGRAM_META,) if launcher.leading else (),
    )


def kernel_arguments(kernel: CallableSignature, ctx: CudaCodegenContext) -> str:
    """What the launcher hands the kernel: its C ABI values, typed as the kernel declared.

    The boundary is crossed here and nowhere else -- the launcher holds what any
    unit can spell, and the kernel body indexes device types.
    """
    written: list[str] = []
    for param in kernel.all_params:
        if not isinstance(param, TensorSignature):
            written.append(param.name)
            continue
        element = ctx.dtype_to_cpp(param.type.dtype.name)
        written.append(f"static_cast<{element}*>({param.name})")
        written += [f"static_cast<int>({param.extent_name(axis)})" for axis in param.dynamic_axes]
        layout = param.type.layout
        if isinstance(layout, ShardLayout):
            written.append(f"{mesh_type(layout.mesh)}{{}}")
    return ", ".join(written)


@register_codegen(CudaTarget, Role.CALLEE, PrimFunction)
def called_as_launcher(
    fn: PrimFunction, symbol: str, program_ids: tuple[ProgramIdSignature, ...]
) -> CallableSignature:
    """The one convention of *fn* a caller can reach: its launcher.

    The kernel's own convention is not this one, because nobody outside the
    translation unit can name it. A program that names a level the target
    names such a level is told its id first, and the launch geometry comes last.
    """
    return CallableSignature(
        name=launcher_symbol(symbol),
        params=tuple(tensor_signature_of(var) for var in fn.params),
        output_count=fn.output_count,
        leading=program_ids,
    )


@register_codegen(CudaTarget, Role.CALLEE, Function)
def hir_launcher_signature(fn, symbol, program_ids, params) -> CallableSignature:
    """The TIR pointer and extent ABI over the arguments *fn*'s Launch binds."""
    return CallableSignature(
        name=launcher_symbol(symbol),
        params=tuple(tensor_signature_of(param) for param in params),
        output_count=len(params) - len(fn.params),
        leading=program_ids,
    )


@register_codegen(CudaTarget, Role.CALLEE, TensorSignature)
def _declare_tensor(sig: TensorSignature, ctx: EmitContext) -> tuple[str, ...]:
    """A tensor arrives as a pointer, then one extent per open axis.

    A static extent is a constant on both sides of the call and costs the
    convention nothing; an axis the type leaves open travels with the pointer.
    Where another unit reads the declaration the pointer is untyped, because a
    host unit cannot spell a device dtype.
    """
    if ctx.exported:
        pointer, extent = _C_ABI
    else:
        pointer, extent = f"{ctx.dtype_to_cpp(sig.type.dtype.name)}*", "int"
    declared = [
        f"{pointer} {sig.name}",
        *(f"{extent} {sig.extent_name(axis)}" for axis in sig.dynamic_axes),
    ]
    layout = sig.type.layout
    if not ctx.exported and isinstance(layout, ShardLayout):
        declared.append(f"{mesh_type(layout.mesh)} {sig.name}_mesh")
    return tuple(declared)


@register_codegen(CudaTarget, Role.CALLEE, LaunchSignature)
def _declare_launch(sig: LaunchSignature, ctx: CudaCodegenContext) -> tuple[str, ...]:
    """One geometry argument, in the C type the convention states for it."""
    return (f"{sig.ctype} {sig.name}",)


@register_codegen(CudaTarget, Role.CALLEE, ProgramIdSignature)
def _declare_program_id(sig: ProgramIdSignature, ctx: CudaCodegenContext) -> tuple[str, ...]:
    """The id of a level no card can read for itself, told to it instead."""
    return (f"{PROGRAM_ID_CTYPE} {sig.name}",)


@register_codegen(CudaTarget, Role.CALLEE, ProgramMetaSignature)
def _declare_program_meta(sig: ProgramMetaSignature, ctx: CudaCodegenContext) -> tuple[str, ...]:
    """The whole block of ids, which a kernel takes by value."""
    return (f"{PROGRAM_META_CTYPE} {sig.name}",)


@register_codegen(CudaTarget, Role.CALLER, TensorSignature)
def _pass_tensor(sig: TensorSignature, ctx: EmitContext) -> tuple[str, ...]:
    """The pointer, then the extents, as the calling scope spells them.

    How many arguments there are and in what order is CUDA's, since CUDA
    declared them; what each is called belongs to whoever is calling.
    """
    return (
        ctx.local_value(sig),
        *(ctx.local_extent(sig, axis) for axis in sig.dynamic_axes),
    )


@register_codegen(CudaTarget, Role.CALLER, LaunchSignature)
@register_codegen(CudaTarget, Role.CALLER, ProgramIdSignature)
@register_codegen(CudaTarget, Role.CALLER, ProgramMetaSignature)
def _pass_value(sig: Signature, ctx: EmitContext) -> tuple[str, ...]:
    """One argument the caller already holds, spelled in its own scope."""
    return (ctx.local_value(sig),)


__all__ = [
    "PROGRAM_ID_CTYPE",
    "PROGRAM_META",
    "PROGRAM_META_CTYPE",
    "kernel_symbol",
    "kernel_arguments",
    "kernel_signature",
    "launcher_symbol",
]
