"""Represent one ANE device translation unit and its C entry points."""

import hashlib
import json
import shutil
from importlib.resources import files

from tilefoundry import codegen, target
from tilefoundry.ir import core, expr, types
from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane import target as ane_target

from . import emit
from .image import Image, IODesc, Reloc, address_slots

PREAMBLE = '#include "tilefoundry/runtime/ane/launcher.h"'


@codegen.register(ane_target.Target, codegen.Role.CALLEE, core.PrimFunction)
def called_as(
    fn: core.PrimFunction, symbol: str, unit_ids: tuple[codegen.UnitIdSignature, ...]
) -> codegen.CallableSignature:
    """Name the ANE entry and retain the logical input and output lists."""
    if unit_ids:
        raise ValueError(f"{fn.name}: the ANE entry takes no unit IDs")
    return codegen.CallableSignature(
        name=symbol,
        params=tuple(codegen.tensor_signature_of(param) for param in fn.params),
        output_count=fn.output_count,
    )


def describe(param: expr.Var) -> IODesc:
    """Read the logical geometry of one static unary feature map."""
    type_ = param.type
    if not isinstance(type_, types.TensorType) or not isinstance(type_.layout, types.Layout):
        raise ValueError(f"{param.name}: ANE I/O needs a flat tensor layout")
    shape = tuple(expr.static_dim_value(n) for n in type_.shape)
    strides = tuple(expr.static_dim_value(n) for n in type_.layout.strides)
    return IODesc(param.name, type_.dtype.name, shape, strides)


def function_image(fn: core.PrimFunction) -> Image:
    """Assemble the TD stream and bind its two placeholders to I/O symbols."""
    program = emit.build_program(fn)
    io = tuple(describe(param) for param in fn.params)
    slots = address_slots(program)
    relocs = []
    for addr, desc in zip(
        (
            isa.TileDMASrc.tile_dma_src_base_addr_lo.addr,
            isa.TileDMADst.tile_dma_dst_base_addr_lo.addr,
        ),
        io,
        strict=True,
    ):
        places = slots[addr]
        if len(places) != 1:
            raise ValueError(f"{fn.name}: expected one address slot for {desc.name}")
        relocs.append(Reloc(places[0], desc.name))
    return Image(program.to_bytes(), b"", io, tuple(relocs))


def byte_rows(image: bytes) -> str:
    """Render the bytes of one constant array."""
    return "\n".join(
        "  " + ", ".join(f"0x{byte:02x}" for byte in image[off : off + 16]) + ","
        for off in range(0, len(image), 16)
    )


def io_row(desc: IODesc) -> str:
    """Render one initializer compatible with the launcher header."""
    shape = ", ".join(map(str, desc.shape))
    strides = ", ".join(map(str, desc.strides))
    return f"  {{{json.dumps(desc.name)}, {json.dumps(desc.dtype)}, {{{shape}}}, {{{strides}}}}},"


def emit_function(
    fn: core.PrimFunction, sig: codegen.CallableSignature
) -> codegen.LinkableFunction:
    """Emit a C ABI entry that forwards the embedded image and I/O arrays."""
    image = function_image(fn)
    payload = image.to_bytes()
    symbol = sig.name
    image_name = f"{symbol}_image"
    io_name = f"{symbol}_io"
    digest = hashlib.sha256(payload).hexdigest()
    declaration = f'extern "C" int {symbol}(DLTensor** inputs, DLTensor** outputs)'
    rows = "\n".join(map(io_row, image.io))
    definition = (
        f"static const uint8_t {image_name}[] = {{\n{byte_rows(payload)}\n}};\n"
        f"static const IODesc {io_name}[] = {{\n{rows}\n}};\n\n"
        f"{declaration} {{\n"
        f"  return ane_launch({image_name}, sizeof({image_name}),\n"
        f'                    "{digest}",\n'
        f"                    {io_name}, {len(image.io)}, inputs, outputs);\n"
        "}\n"
    )
    return codegen.LinkableFunction(fn.name, declaration + ";", definition)


@target.bind(ane_target.Target, codegen.LinkableModule)
class AneLinkableModule(codegen.LinkableModule):
    """The C++ translation unit for an ANE target group."""

    @classmethod
    def create(cls, tgt, owner, functions, ctx) -> "AneLinkableModule":
        """Emit all function images and launcher entry points in group order."""
        if ctx.codegen_context is not None and ctx.codegen_context.options.emit != "full":
            raise ValueError("ANE codegen requires full emission of lowered TIR")
        entries = []
        for fn in functions:
            if not isinstance(fn, core.PrimFunction) or fn.target != tgt:
                raise ValueError(f"{fn.name}: expected a lowered ANE function")
            sig = ctx.signature_of(fn)
            if sig.leading or sig.trailing:
                raise ValueError(f"{fn.name}: ANE entry does not take launch geometry")
            entries.append(emit_function(fn, sig))
        if not entries:
            raise ValueError("ANE codegen requires at least one function")
        return cls(target="ane", language="cpp", preamble=PREAMBLE, functions=tuple(entries))

    def link(self, project: "CMakeProject") -> None:
        """Add the source, launcher headers, and Darwin link options."""
        if shutil.which("clang++") is None:
            raise RuntimeError("ANE link: 'clang++' not on PATH")
        super().link(project)
        project.include(files(__package__).joinpath("include"))
        project.append(
            'if(NOT APPLE OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")\n'
            '  message(FATAL_ERROR "ANE linking requires clang++ on macOS")\n'
            "endif()\n"
            'set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")\n'
            "target_link_options(${PROJECT_NAME} PRIVATE -F/System/Library/PrivateFrameworks)\n"
            'target_link_libraries(${PROJECT_NAME} PRIVATE "-framework CoreFoundation"\n'
            '  "-framework IOSurface" "-framework AppleNeuralEngine" objc)\n'
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
