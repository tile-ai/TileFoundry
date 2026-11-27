"""Represent one ANE device translation unit and its C entry points."""

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files

from tilefoundry import codegen, target
from tilefoundry.ir import core, expr, types
from tilefoundry_apple.ane import isa, runtime
from tilefoundry_apple.ane import target as ane_target
from tilefoundry_apple.ane.codegen.tir.ne import conv

from . import hwx, program, tir

PREAMBLE = '#define TVM_FFI_DLL_EXPORT_INCLUDE_METADATA 1\n#include "tilefoundry/runtime/ane/launcher.h"'
LAUNCHER = "ane_launcher.cpp"


@codegen.register(ane_target.Target, codegen.Role.CALLEE, core.PrimFunction)
@codegen.register(ane_target.Target, codegen.Role.CALLEE, core.Function)
def called_as(
    fn: core.PrimFunction | core.Function, symbol: str, unit_ids: tuple[codegen.UnitIdSignature, ...], params=None,
) -> codegen.CallableSignature:
    """Name the ANE entry and retain the logical input and output lists."""
    if unit_ids:
        raise ValueError(f"{fn.name}: ANE unit ids is not decoded yet")
    return codegen.CallableSignature(
        name=symbol,
        params=tuple(
            codegen.TensorSignature(param.name, types.TensorType.from_shape(param.type.shape, dtype=param.type.dtype, storage=expr.StorageKind.HOST))
            for param in (fn.params if params is None else params)
            if not getattr(param, "is_const", False)
        ),
        output_count=getattr(fn, "output_count", 0),
    )


def function_image(
    fn: core.PrimFunction, constants: Mapping[str, bytes] | None = None
) -> hwx.Image:
    """Assemble an entry with supplied weights, or zero weights for its compiled skeleton."""
    ctx = tir.build_context(fn)
    stream = program.to_program(ctx.passes)
    io = tuple(
        hwx.TensorBinding(param.name, ctx.binding_desc(param))
        for param in fn.params
        if not getattr(param, "is_const", False)
    )
    outputs = fn.output_count
    if not 0 < outputs <= len(io):
        raise ValueError(f"{fn.name}: output count {outputs} does not fit {len(io)} parameters")
    if len(io) - outputs > 2 or outputs > 1:
        raise ValueError(f"{fn.name}: more than two inputs or one output is not decoded yet")
    operands = {
        isa.TileDMASrc.tile_dma_src_base_addr_lo.addr: 1,
        isa.TileDMASrc.tile_dma_src2_base_addr_lo.addr: 2,
        isa.TileDMADst.tile_dma_dst_base_addr_lo.addr: 0,
    }
    relocs = []
    for word, (index, register, _) in hwx.base_slots(stream).items():
        if register in isa.COEFFICIENT_RELOCATIONS:
            continue
        operand = ctx.operands[index][operands[register]]
        source = ctx.symbols.get(operand, operand)
        relocs.append(hwx.Reloc(word, source.name))
    coefficients = []
    for weight, layout in ctx.coefficients.items():
        if constants is not None and weight.name not in constants:
            raise ValueError(f"{fn.name}: no bytes supplied for constant {weight.name}")
        values = (
            bytes(layout.lanes * layout.lane_bytes) if constants is None else constants[weight.name]
        )
        coefficients.append(hwx.Coefficient(layout, values))
    consumers = {
        index: tuple(ctx.coefficients).index(weight) for weight, index, _ in ctx.weight_uses
    }
    return hwx.Image(
        stream,
        io,
        tuple(relocs),
        output_count=outputs,
        coefficients=tuple(coefficients),
        coefficient_relocs=hwx.coefficient_relocations(stream, consumers),
    )


def byte_rows(image: bytes) -> str:
    """Render the bytes of one constant array."""
    return "\n".join(
        "  " + ", ".join(f"0x{byte:02x}" for byte in image[off : off + 16]) + ","
        for off in range(0, len(image), 16)
    )


def tensor_row(desc: isa.TensorDesc) -> str:
    """Render one initializer compatible with the launcher header."""
    shape = ", ".join(map(str, desc.shape))
    strides = ", ".join(map(str, desc.strides))
    return f"  {{{json.dumps(desc.dtype)}, {{{shape}}}, {{{strides}}}, {desc.storage_bytes}}},"


def emit_function(
    fn: core.PrimFunction, sig: codegen.CallableSignature, entry_name: str, *, exported: bool
) -> codegen.LinkableFunction:
    """Export a validated skeleton, I/O descriptors and per-weight generated writers."""
    image = function_image(fn)
    payload = image.to_bytes()
    symbol = sig.name
    ctx = tir.build_context(fn)
    writers, weights = [], []
    for index, (weight, layout) in enumerate(ctx.coefficients.items()):
        writer = f"{symbol}_weight_{index}"
        writers.append(conv.weight_writer(writer, layout))
        weights.append(f"  {{{json.dumps(weight.name)}, {writer}}},")
    offset = hwx.coefficient_offset(payload) if weights else 0
    params = ", ".join(f"DLTensor *arg_{index}" for index in range(len(sig.params)))
    declaration = f"void {symbol}(int64_t token, {params})"
    inputs = ", ".join(f"arg_{index}" for index in range(sig.input_count)) or "nullptr"
    outputs = ", ".join(f"arg_{index}" for index in range(sig.input_count, len(sig.params)))
    validation = "\n".join(
        f"  if (!ane_check_tensor(arg_{index}, {symbol}_io[{index}].shape, 16, false))\n"
        f'    TVM_FFI_THROW(ValueError) << "{entry_name}: {param.name} must be a CPU fp16 '
        f'tensor of shape {tuple(param.type.shape)} with nonnegative strides and bounded offsets";'
        for index, param in enumerate(sig.params)
    )
    rows = "\n".join(tensor_row(binding.desc) for binding in image.io)
    definition = (
        f"static const uint8_t {symbol}_image[] = {{\n{byte_rows(payload)}\n}};\n"
        f"static const TensorDesc {symbol}_io[] = {{\n{rows}\n}};\n"
        + "\n".join(writers)
        + (
            f"static const AneWeight {symbol}_weights[] = {{\n" + "\n".join(weights) + "\n};\n"
            if weights
            else ""
        )
        + f"static const AneEntry {symbol}_entry = {{\n"
        f"  {symbol}_image, sizeof({symbol}_image), {offset},\n"
        f"  {symbol}_io, {len(image.io) - image.output_count}, {image.output_count},\n"
        + (f"  {symbol}_weights, {len(weights)}\n" if weights else "  nullptr, 0\n")
        + f"}};\n{declaration} {{\n"
        f"{validation}\n"
        f"  DLTensor *inputs[] = {{{inputs}}};\n"
        f"  DLTensor *outputs[] = {{{outputs}}};\n"
        f"  ane_token_launch(token, inputs, {sig.input_count}, outputs, {sig.output_count});\n}}\n"
    )
    if exported:
        definition += (
            f"static int64_t {symbol}_create() {{ return ane_token_create(&{symbol}_entry); }}\n"
            f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({symbol}, {symbol});\n"
            f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({symbol}_create, {symbol}_create);\n"
        )
        definition += "".join(
            f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({symbol}_{operation}, ane_token_{operation});\n"
            for operation in ("set_weight", "load", "cache_path", "image", "destroy")
        )
    return codegen.LinkableFunction(fn.name, declaration + ";", definition)


@target.bind(ane_target.Target, codegen.LinkableModule)
@dataclass(frozen=True)
class AneLinkableModule(codegen.LinkableModule):
    """The C++ translation unit for an ANE target group."""

    operations: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def create(cls, tgt, owner, functions, ctx) -> "AneLinkableModule":
        """Emit all function images and launcher entry points in group order."""
        if ctx.codegen_context is None:
            raise ValueError("ANE codegen requires a codegen.Context")
        if ctx.codegen_context.options.emit != "full":
            raise ValueError("ANE codegen requires full emission of lowered TIR")
        entries, operations = [], {}
        for fn in functions:
            if not isinstance(fn, core.PrimFunction) or fn.target != tgt:
                raise ValueError(f"{fn.name}: expected a lowered ANE function")
            sig = ctx.signature_of(fn)
            if sig.leading or sig.trailing:
                raise ValueError(f"{fn.name}: ANE launch geometry is not decoded yet")
            root = ctx.codegen_context.module
            exported = fn is root.entry_function()
            entry_name = ctx.codegen_context.symbols.by_function[id(fn)].export_name
            entries.append(emit_function(fn, sig, entry_name, exported=exported))
            if exported:
                operations = {name: f"{sig.name}_{name}" for name in ("create", "set_weight", "load", "cache_path", "image", "destroy")}
        if not entries:
            raise ValueError("ANE codegen requires at least one function")
        return cls(target="ane", language="cpp", preamble=PREAMBLE, functions=tuple(entries), operations=operations)

    def link(self, project: "CMakeProject") -> None:
        """Add the source, launcher headers, and Darwin link options."""
        if shutil.which("clang++") is None:
            raise RuntimeError("ANE link: 'clang++' not on PATH")
        super().link(project)
        project.include(files(__package__).joinpath("include"))
        if LAUNCHER not in project.sources:
            project.add_source(LAUNCHER, files(runtime).joinpath("launcher.cpp").read_text())
        project.append(
            'if(NOT APPLE OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")\n'
            '  message(FATAL_ERROR "ANE linking requires clang++ on macOS")\n'
            "endif()\n"
            'target_link_libraries(${PROJECT_NAME} PRIVATE "-framework CoreFoundation"\n'
            '  "-framework IOSurface" "-framework Security" objc)\n'
        )


@target.bind(ane_target.Target, codegen.LinkedModule)
class AneLinkedModule(codegen.LinkedModule):
    """A linked ANE artifact produced by the shared CMake project."""

    @classmethod
    def create(cls, tgt, project, functions) -> "AneLinkedModule":
        """Let the shared project build the target contributions."""
        linked = project.build(functions)
        return cls(linked.library_path, linked.source, linked.functions)


__all__ = ["AneLinkableModule", "AneLinkedModule", "function_image"]
