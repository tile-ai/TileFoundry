"""H14 HWX records encoding."""

import struct

from tilefoundry_apple.ane.isa import arith

from .bindings import TensorBinding

CMD_BUFFER_REFERENCE = 0x40
CMD_OPERATION = 0x04
CMD_COMPILER_METADATA = 0x08
CMD_SOURCE_MODEL_INFO = 0x31
SOURCE_MODEL_TAG = b"src model info\0\0"
BUFFER_REFERENCE_SIZE = 0x20
SEGMENT_RECORD_SIZE = 0x8A0
SEGMENT_RECORD_LAYOUT = 0x222
SEGMENT_RECORD_PROCEDURE_OFFSET = 0x898
BINDING_RECORD_MIN_SIZE = 0xD48
BINDING_RECORD_LAYOUT = 0x34A
BINDING_RECORD_PROCEDURE_OFFSET = 0xD38
BINDING_RECORD_SYMBOL_OFFSET = 0xD3D
RESOURCE_SLOTS = 5
DEFAULT_RECORD_COUNT = 17
TOOL_NAME = "tilefoundry ane codegen v1"
TARGET_PROFILE = (
    ("-t", "h14g"),
    ("--fno-fold-scale", "true"),
    ("--fdram-allocator", "ffreuse"),
    ("--fdram-tensor-priority", "sizebyliverange"),
    ("--fl2-allocator", "ffreuse"),
    ("--fl3-allocator", "ffreuse"),
    ("--fl2-cache-mode", "resident"),
    ("--fsignature", "ident"),
    ("--fdisable-bonded-networks", "true"),
    ("--enable-param-and-map-rtgraph-refactor", "false"),
    ("--memcache-size", "4194304"),
    ("--fspatial-split", "disabled"),
    ("--fenable-circular-buffer-in-spatial-split", "-1"),
    ("--fkernel-rewind", "enabled"),
    ("--fdump-status-dictionary-to-file", None),
    ("--max-td-latency", "10000.000000"),
    ("--ne-frequency", "-1.000000"),
    ("--pstate-dcs", "-1"),
    ("--pstate-soc", "-1"),
    ("--bss-limit", "-1"),
    ("--foptimize-ne-utilization", "true"),
    ("--disable-cache-prefetch-mask", "1"),
    ("--optimize-mutable-kernel-section", "true"),
    ("--enable-summary-performance-stats", "false"),
    ("--split-kernel-section", "true"),
    ("--max-kernel-section-size", "134217728"),
    ("--enable-context-switch-events", "false"),
    ("--fdisable-ne-width-slicing", "false"),
    ("--disable-hard-swish-opt", "false"),
    ("--preserve-texture-fraction", "false"),
    ("--max-td-count", "-1"),
    ("--max-segment-size", "-1"),
    ("--e4m3-overflow-setting", "Saturate"),
    ("--enable-dram-inplace-allocation", "false"),
    ("--enable-l2-batch-splitting", "true"),
    ("--enable-global-cw-optimization", "true"),
    ("--suppress-math-exception", "false"),
    ("--enable-l2-cached-buffer", "true"),
    ("--enable-low-effort-cp-allocation", "false"),
    ("--enable-noise-reduction-tiling", "false"),
    ("--dump-parallel-score", "false"),
    ("--enable-function-inlining", "true"),
    ("--fdisable-weight-file-size-check", "false"),
    ("--compile_time_mutable", "false"),
    ("--enable-afm-mlir-features", "true"),
    ("--enable-stable-precision", "false"),
    ("--enable-kernel-split-for-multi-palette-lut", "false"),
    ("--enable-non-self-replaceable-persistence", "false"),
    ("--cost-model-cluster-threshold", "85"),
    ("--dma-buffer-allocation-prioritization", "false"),
    ("--enable-matmul-kernel-pad-insertion", "false"),
    ("--Wl-undefined", "fvmlib"),
)


def validate_names(procedure: str, names: tuple[str, ...]) -> None:
    """Reject overlapping fixed strings and colliding eight-byte I/O references."""
    if len(procedure) + 1 > BINDING_RECORD_SYMBOL_OFFSET - BINDING_RECORD_PROCEDURE_OFFSET:
        raise ValueError("procedure name overlaps the binding record symbol region")
    short = [name[:8] for name in names]
    if len(set(short)) != len(short):
        raise ValueError("I/O names collide after truncation to eight bytes")


def buffer_reference(binding: TensorBinding) -> bytes:
    """Encode the 0x40 load command that names one surface and its address."""
    name = binding.name.encode("ascii")
    return (
        struct.pack("<IIQ", CMD_BUFFER_REFERENCE, BUFFER_REFERENCE_SIZE, 0x18)
        + struct.pack("<Q", binding.address)
        + name[:8].ljust(8, b"\0")
    )


def segment_record(
    text_address: int,
    text_const_address: int,
    resources: tuple[int, ...],
    task_words: int,
    task_count: int,
    procedure: str,
    record_count: int,
    format_code: int,
) -> bytes:
    """Encode the 0x04 segment record: the procedure's VM entry points."""
    if len(resources) > RESOURCE_SLOTS:
        raise ValueError(f"at most {RESOURCE_SLOTS} resources fit the segment record")
    name = procedure.encode("ascii") + b"\0"
    size = max(SEGMENT_RECORD_SIZE, arith.align_up(SEGMENT_RECORD_PROCEDURE_OFFSET + len(name), 8))
    out = bytearray(size)
    struct.pack_into("<IIII", out, 0x000, CMD_OPERATION, size, 0x4, SEGMENT_RECORD_LAYOUT)
    struct.pack_into("<Q", out, 0x010, text_address)
    struct.pack_into("<Q", out, 0x020, text_const_address)
    for index, address in enumerate(resources):
        struct.pack_into("<Q", out, 0x050 + 0x10 * index, address)
    struct.pack_into("<Q", out, 0x810, text_address)
    struct.pack_into("<I", out, 0x820, 4)
    struct.pack_into("<I", out, 0x824, task_words)
    struct.pack_into("<I", out, 0x830, task_count)
    struct.pack_into("<II", out, 0x838, 0xFFFFFFFF, 0xFFFFFFFF)
    struct.pack_into("<I", out, 0x844, 0xFFFF)
    struct.pack_into("<I", out, 0x848, SEGMENT_RECORD_PROCEDURE_OFFSET)
    struct.pack_into("<I", out, 0x860, record_count)
    struct.pack_into("<II", out, 0x868, 3, 0)
    struct.pack_into("<II", out, 0x870, 2, 0)
    struct.pack_into("<III", out, 0x878, 7, 8, 0)
    struct.pack_into("<II", out, 0x880, 0x461C4000, 0)
    struct.pack_into("<II", out, 0x888, 9, 8)
    struct.pack_into("<I", out, 0x890, format_code)
    struct.pack_into("<I", out, 0x894, 0)
    out[SEGMENT_RECORD_PROCEDURE_OFFSET : SEGMENT_RECORD_PROCEDURE_OFFSET + len(name)] = name
    return bytes(out)


def binding_record(binding: TensorBinding, procedure: str, declared: bool = False) -> bytes:
    """Encode one 0x04 binding record: the shape and strides of a surface."""
    desc = binding.desc
    symbol = binding.tensor_symbol.encode("ascii") + b"\0"
    short = binding.name.encode("ascii") + b"\0"
    symbol_offset = BINDING_RECORD_SYMBOL_OFFSET
    short_offset = symbol_offset + len(symbol)
    size = max(BINDING_RECORD_MIN_SIZE, arith.align_up(short_offset + len(short), 8))
    row, plane, batch, total = desc.declared_geometry if declared else desc.canonical_geometry
    element = desc.element_bytes
    out = bytearray(size)
    struct.pack_into("<IIII", out, 0x000, CMD_OPERATION, size, 3, BINDING_RECORD_LAYOUT)
    struct.pack_into("<II", out, 0x010, 3, binding.role)
    struct.pack_into("<I", out, 0x018, BINDING_RECORD_PROCEDURE_OFFSET)
    struct.pack_into("<I", out, 0x020, symbol_offset)
    struct.pack_into("<I", out, 0x024, desc.element_code)
    struct.pack_into("<IIII", out, 0x028, *desc.shape)
    struct.pack_into("<I", out, 0x038, 1)
    struct.pack_into("<QQQQQ", out, 0x050, batch, plane, row, element, total)
    struct.pack_into("<I", out, 0x078, 1)
    struct.pack_into("<I", out, 0x07C, short_offset)
    struct.pack_into("<Q", out, 0x080, total)
    procedure_bytes = procedure.encode("ascii") + b"\0"
    out[
        BINDING_RECORD_PROCEDURE_OFFSET : BINDING_RECORD_PROCEDURE_OFFSET + len(procedure_bytes)
    ] = procedure_bytes
    out[symbol_offset:short_offset] = symbol
    out[short_offset : short_offset + len(short)] = short
    return bytes(out)


def text_record(command: int, payload: bytes) -> bytes:
    """Encode an ANEC command whose body is an opaque byte payload."""
    size = arith.align_up(8 + len(payload), 8)
    return struct.pack("<II", command, size) + payload.ljust(size - 8, b"\0")


def compiler_metadata(
    tool: str,
    version: str,
    kernel_bytes: int,
    binding_bytes: int,
    task_count: int,
    flags: tuple[tuple[str, str | None], ...] = TARGET_PROFILE,
) -> bytes:
    """Encode the compiler banner and target options as the metadata command."""
    lines = [
        "ANEC v1",
        tool,
        "",
        "coremltools-component-milinternal: ",
        "coremltools-version: %s" % version,
        "coremlc-component-MIL: %s" % version,
        "coremlc-version: %s" % version,
        "",
        " Module ANEC: ",
        "\t ModuleBundleName: %s" % tool.replace(" ", "."),
        "\t ModuleVersion: %s" % version,
        "\t ModuleCompilationFlags: ",
    ]
    for name, value in flags:
        lines.append(f"\t{name}" if value is None else f"\t{name}={value}")
    lines.extend(
        [
            "\t--kernels=%d" % kernel_bytes,
            "\t--surfaces=%d" % binding_bytes,
            "\t--tasks=%d" % task_count,
        ]
    )
    text = "\n".join(lines) + "\n"
    return text_record(CMD_COMPILER_METADATA, text.encode("ascii") + b"\0")


def source_model_info(tool: str, version: str, procedure: str, task_count: int) -> bytes:
    """Encode the 0x31 command: the tagged source-model provenance blob.

    The record is ``{u32 cmd, u32 size, char tag[16], u64 text_offset,
    u64 text_length, text}`` with the text at offset 0x28 from the command.  The
    reference text is a short block of ``label: value`` lines naming the
    front-end that produced the source model.
    """
    body = (
        f"\n{tool}\n"
        f"tilefoundry-version: {version}\n"
        f"ane-procedure: {procedure}\n"
        f"ane-task-count: {task_count}\n"
    )
    text = body.encode("ascii") + b"\0"
    payload = bytearray(SOURCE_MODEL_TAG)
    payload += struct.pack("<QQ", 0x28, len(text))
    payload += text
    return text_record(CMD_SOURCE_MODEL_INFO, bytes(payload))


def runtime_version() -> bytes:
    """Encode the runtime-version command: a four-byte 1.1 payload."""
    payload = bytearray(SOURCE_MODEL_TAG[:0] + b"rt.version".ljust(16, b"\0"))
    text = struct.pack("<4B", 1, 0, 1, 0)
    payload += struct.pack("<QQ", 0x28, len(text))
    payload += text
    return text_record(CMD_SOURCE_MODEL_INFO, bytes(payload))


__all__ = [
    "DEFAULT_RECORD_COUNT",
    "RESOURCE_SLOTS",
    "TOOL_NAME",
    "binding_record",
    "buffer_reference",
    "compiler_metadata",
    "runtime_version",
    "segment_record",
    "source_model_info",
    "validate_names",
]
