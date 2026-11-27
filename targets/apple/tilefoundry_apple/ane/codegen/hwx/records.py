"""The ANE records of an HWX container: the load commands and symbols its loader reads."""

from collections.abc import Iterator

from .bindings import BINDING_ROLE_INPUT, BINDING_ROLE_OUTPUT, TensorBinding
from .macho import COMMAND, N_EXT, N_GSYM, N_LSYM, N_SECT, Field, Record, load_command

RESOURCE_SLOTS = 5
NAME_BYTES = 8
PROCEDURE_BYTES = 5
SEGMENT_PROCEDURE = 0x898
BINDING_PROCEDURE = 0xD38
DEFAULT_RECORD_COUNT = 17
TOOL_NAME = "tilefoundry ane codegen v1"
BUFFER_REFERENCE = Record(
    Field(0x00, "I", "command", 0x40),
    Field(0x04, "I", "size"),
    Field(0x08, "Q", default=0x18),
    Field(0x10, "Q", "address"),
    Field(0x18, f"{NAME_BYTES}s", "name"),
)
SEGMENT_RECORD = Record(
    Field(0x000, "I", "command", 0x04),
    Field(0x004, "I", "size"),
    Field(0x008, "I", default=0x4),
    Field(0x00C, "I", default=0x222),
    Field(0x010, "Q", "text"),
    Field(0x020, "Q", "text_const"),
    Field(0x050, "Q8x" * RESOURCE_SLOTS, "resources", (0,) * RESOURCE_SLOTS),
    Field(0x810, "Q", "text"),
    Field(0x820, "I", default=4),
    Field(0x824, "I", "task_words"),
    Field(0x830, "I", "task_count"),
    Field(0x838, "II", default=(0xFFFFFFFF, 0xFFFFFFFF)),
    Field(0x844, "I", "traits", 0xFFFF),
    Field(0x848, "I", "procedure_offset", SEGMENT_PROCEDURE),
    Field(0x860, "I", "record_count"),
    Field(0x868, "I", default=3),
    Field(0x870, "I", default=2),
    Field(0x878, "II", default=(7, 8)),
    Field(0x880, "I", default=0x461C4000),
    Field(0x888, "II", default=(9, 8)),
    Field(0x890, "I", "format_code"),
    Field(SEGMENT_PROCEDURE, "8s", "procedure"),
)
BINDING_RECORD = Record(
    Field(0x000, "I", "command", 0x04),
    Field(0x004, "I", "size"),
    Field(0x008, "I", default=3),
    Field(0x00C, "I", default=0x34A),
    Field(0x010, "I", default=3),
    Field(0x014, "I", "role"),
    Field(0x018, "I", "procedure_offset", BINDING_PROCEDURE),
    Field(0x020, "I", "symbol_offset"),
    Field(0x024, "I", "element_code"),
    Field(0x028, "4I", "shape"),
    Field(0x038, "I", default=1),
    Field(0x050, "4Q", "strides"),
    Field(0x070, "Q", "total"),
    Field(0x078, "I", default=1),
    Field(0x07C, "I", "name_offset"),
    Field(0x080, "Q", "total"),
    Field(BINDING_PROCEDURE, f"{PROCEDURE_BYTES}s", "procedure"),
)
TAGGED = Record(
    Field(0x00, "I", "command", 0x31),
    Field(0x04, "I", "size"),
    Field(0x08, "16s", "tag"),
    Field(0x18, "Q", "offset"),
    Field(0x20, "Q", "length"),
)
TARGET_PROFILE = tuple("""
    -t=h14g --fno-fold-scale=true --fdram-allocator=ffreuse
    --fdram-tensor-priority=sizebyliverange --fl2-allocator=ffreuse --fl3-allocator=ffreuse
    --fl2-cache-mode=resident --fsignature=ident --fdisable-bonded-networks=true
    --enable-param-and-map-rtgraph-refactor=false --memcache-size=4194304
    --fspatial-split=disabled --fenable-circular-buffer-in-spatial-split=-1
    --fkernel-rewind=enabled --fdump-status-dictionary-to-file --max-td-latency=10000.000000
    --ne-frequency=-1.000000 --pstate-dcs=-1 --pstate-soc=-1 --bss-limit=-1
    --foptimize-ne-utilization=true --disable-cache-prefetch-mask=1
    --optimize-mutable-kernel-section=true --enable-summary-performance-stats=false
    --split-kernel-section=true --max-kernel-section-size=134217728
    --enable-context-switch-events=false --fdisable-ne-width-slicing=false
    --disable-hard-swish-opt=false --preserve-texture-fraction=false --max-td-count=-1
    --max-segment-size=-1 --e4m3-overflow-setting=Saturate
    --enable-dram-inplace-allocation=false --enable-l2-batch-splitting=true
    --enable-global-cw-optimization=true --suppress-math-exception=false
    --enable-l2-cached-buffer=true --enable-low-effort-cp-allocation=false
    --enable-noise-reduction-tiling=false --dump-parallel-score=false
    --enable-function-inlining=true --fdisable-weight-file-size-check=false
    --compile_time_mutable=false --enable-afm-mlir-features=true --enable-stable-precision=false
    --enable-kernel-split-for-multi-palette-lut=false
    --enable-non-self-replaceable-persistence=false --cost-model-cluster-threshold=85
    --dma-buffer-allocation-prioritization=false --enable-matmul-kernel-pad-insertion=false
    --Wl-undefined=fvmlib
""".split())
ELEMENT_CATALOG = tuple("""
    void:t1=1 int8:t2=r2;0;127 uint8:t3=r1;0;255 int16:t4=r1;-32768;32767 float16:t5=r1;2;0
    float:t6=r1;4;0 raw10:t7=r1;-512;511 lut:t8= uint4:t9=r1;0;15 uint6:t10=r1;0;63
    uint1:t11=r1;0,1 uint2:t12=r1;0;3 packed10:t13=r1;-512;511 y10:t14=r1;-512;511
    uint16:t15=r1;0;65535 e4m3:t16=r1;1;0 int32:t17=r1;-2147483648;2147483647
    uint32:t18=r1;0;4294967295 int64:t19=r1;-9223372036854775808;9223372036854775807
    uint64:t20=r1;0;18446744073709551615 raw12:t21=r1;-512;511 y12:t22=r1;-512;511
    int4:t23=r1;-8;7
""".split())


def commands(
    image: "Image", bindings: tuple[TensorBinding, ...], text: bytes, kernel: bytes,
    text_address: int, const_address: int, kernel_address: int,
) -> tuple[bytes, ...]:
    """Encode the ANE load commands of *image* in loader order.

    Buffer references follow address order, the segment record lists the surfaces and
    then the kernel as resources, and binding records list inputs before outputs.
    """
    resources = tuple(binding.address for binding in bindings)
    if kernel and len(resources) < RESOURCE_SLOTS:
        resources += (kernel_address,)
    if len(resources) > RESOURCE_SLOTS:
        raise ValueError(f"at most {RESOURCE_SLOTS} resources fit the segment record")
    tasks = len(image.program.tds)
    format_code = image.format_code
    if format_code is None:
        format_code = max(descriptor.header.get(0x04, 0) for descriptor in image.program.tds)
    source = (
        f"\n{image.tool}\ntilefoundry-version: {image.version}\n"
        f"ane-procedure: {image.procedure}\nane-task-count: {tasks}\n\0"
    )
    references = tuple(
        load_command(BUFFER_REFERENCE, address=binding.address, name=binding.name.encode("ascii"))
        for binding in bindings
    )
    segment = load_command(
        SEGMENT_RECORD, text=text_address, text_const=const_address,
        resources=resources + (0,) * (RESOURCE_SLOTS - len(resources)),
        task_words=len(text) // 4, task_count=tasks, record_count=image.record_count,
        format_code=format_code, procedure=image.procedure.encode("ascii"),
    )
    records = tuple(
        binding_record(binding, image.procedure, image.layout == "declared")
        for role in (BINDING_ROLE_INPUT, BINDING_ROLE_OUTPUT)
        for binding in bindings
        if binding.role == role
    )
    surfaces = sum(binding.size for binding in bindings)
    return (
        *references,
        segment,
        *records,
        compiler_metadata(image.tool, image.version, len(kernel), surfaces, tasks),
        tagged(b"src model info", source.encode("ascii")),
        tagged(b"rt.version", bytes((1, 0, 1, 0))),
    )


def symbols(
    image: "Image", bindings: tuple[TensorBinding, ...], kernel_address: int, kernel_section: int
) -> Iterator[tuple[str, int, int, int, int]]:
    """Name the surfaces, every coefficient lane, the element catalog and the surface shapes.

    Surface and lane names are external section symbols. Shape type ids continue past
    the catalog, five per surface.
    """
    for index, binding in enumerate(bindings):
        yield binding.name, N_SECT | N_EXT, index + 1, 0, binding.address
    token = image.kernel_digest or image.kernel_symbol
    for index, coefficient in enumerate(image.coefficients):
        prefix = f"{token}_{index}" if len(image.coefficients) > 1 else token
        layout = coefficient.layout
        for lane in range(layout.lanes):
            address = kernel_address + layout.base + lane * layout.stride
            desc = 2 if lane == 0 else 0
            yield f"{prefix}_ne_{lane}", N_SECT | N_EXT, kernel_section, desc, address
    for entry in ELEMENT_CATALOG:
        yield entry, N_LSYM, 0, 0, 0
    for index, binding in enumerate(bindings):
        base = 1 + len(ELEMENT_CATALOG) + 5 * index
        yield shape_stab(binding, base), N_GSYM, 0, base & 0x1F, 0


def binding_record(binding: TensorBinding, procedure: str, declared: bool) -> bytes:
    """Encode one binding record; the tensor symbol and the short name follow its fields."""
    desc = binding.desc
    row, plane, batch, total = desc.declared_geometry if declared else desc.canonical_geometry
    symbol = binding.tensor_symbol.encode("ascii") + b"\0"
    return load_command(
        BINDING_RECORD, symbol + binding.name.encode("ascii") + b"\0", role=binding.role,
        symbol_offset=BINDING_RECORD.size, element_code=desc.element_code, shape=desc.shape,
        strides=(batch, plane, row, desc.element_bytes), total=total,
        name_offset=BINDING_RECORD.size + len(symbol), procedure=procedure.encode("ascii"),
    )


def compiler_metadata(tool: str, version: str, kernels: int, surfaces: int, tasks: int) -> bytes:
    """Encode the 0x08 command: the compiler banner, the target profile and the image sizes."""
    lines = (
        "ANEC v1", tool, "", "coremltools-component-milinternal: ",
        f"coremltools-version: {version}", f"coremlc-component-MIL: {version}",
        f"coremlc-version: {version}", "", " Module ANEC: ",
        f"\t ModuleBundleName: {tool.replace(' ', '.')}", f"\t ModuleVersion: {version}",
        "\t ModuleCompilationFlags: ", *(f"\t{flag}" for flag in TARGET_PROFILE),
        f"\t--kernels={kernels}", f"\t--surfaces={surfaces}", f"\t--tasks={tasks}",
    )
    return load_command(COMMAND, "\n".join(lines).encode("ascii") + b"\n\0", command=0x08)


def tagged(tag: bytes, payload: bytes) -> bytes:
    """Encode a 0x31 command: a 16-byte tag, then the offset and length of *payload*."""
    return load_command(TAGGED, payload, tag=tag, offset=TAGGED.size, length=len(payload))


def shape_stab(binding: TensorBinding, base: int) -> str:
    """Encode a surface's element type and per-axis byte strides as type ``base``.

    The stab is ``name:t<base>=`` followed by one ``ar1;0;<extent>;<id>=s<stride><axis>``
    group per axis (batch, channel, height, width) and the element code last.
    """
    desc = binding.desc
    row, plane, batch, _ = desc.canonical_geometry
    strides = (batch, plane, row, desc.element_bytes)
    groups = (
        f"ar1;0;{extent};{base + 1 + axis}=s{stride}{name}"
        for axis, (name, extent, stride) in enumerate(zip("nchw", desc.shape, strides))
    )
    return f"{binding.name}:t{base}=" + ":".join(groups) + f":{desc.element_code}"


__all__ = [
    "DEFAULT_RECORD_COUNT", "NAME_BYTES", "PROCEDURE_BYTES", "TOOL_NAME", "commands", "symbols",
]
