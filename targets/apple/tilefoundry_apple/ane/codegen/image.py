"""Write a complete H14 ``.hwx`` image container.

An H14 image is a Mach-O variant (``0xBEEFFACE``, cputype ``0x80``, cpusubtype 5)
whose load commands carry seven ANEC extensions on top of the two segment
commands a Mach-O reader understands.  The layout encoded here was recovered by
diffing reference containers and is verified byte-for-byte against them:

    header                     8 x u32
    LC_SEGMENT_64   __PAGEZERO  no sections, VM only, 16 KiB, prot 0/0
    LC_SEGMENT_64   __FVMLIB    one section per I/O surface
                                  outputs: __data,  prot 2/2, section flags 0x23
                                  inputs:  __const, prot 1/1, section flags 0x21
                                fileoff = filesize = 0 (bound at load)
    LC_SEGMENT_64   __TEXT      2 sections, prot 5/5, vmsize = filesize = 0x8000
                                  __text   flags 0x28 (0x128 with relocations)
                                  __const  flags 0x26, 0x4000 bytes, zeros
    LC_SEGMENT_64   __KERN_0    only when the image carries a coefficient blob
                                  1 section __kern_0, prot 1/1, flags 0x4
                                  segname __kern_0, section flags 0x26
    0x40 x N                    buffer reference: {u32 0x40, u32 0x20, u64 0x18,
                                  u64 vmAddress, char name[8]}
    0x04 x 1                    segment record, 0x8a0 bytes (see _segment_record)
    0x04 x N                    binding record, >= 0xd48 bytes (_binding_record)
    0x08                        compiler metadata, UTF-8 text
    0x31                        source model info, tagged UTF-8 text
    LC_SYMTAB                   one N_SECT|N_EXT symbol per I/O surface

The virtual address space starts at 0x30000000.  Each surface reserves
``align_up(storage_bytes, 0x4000)``, then ``__TEXT`` takes 0x8000, then
``__KERN_0`` takes ``align_up(kernel_bytes, 0x4000)``.  On the file side the
commands (which carry the ANEC payloads inline), the symbol table, the string
table and the __text relocation table all live below the first segment, which
starts at 0x4000 when they fit there and at 0x8000 otherwise.

The segment record carries the text address, the word and task counts, a trait
mask at +0x844, and a trailing block of per-descriptor bookkeeping from +0x860:
two statistics words, one unsigned argument, and the eight-word
``{7, 8, 10000.0f, 0, 9, 8, <code>, 0}`` group.  That bookkeeping is produced by
the producer of the image, not by the descriptor, so ``record_count`` (the +0x860
word) and ``format_code`` (the last word of the group) are explicit parameters,
defaulted to the values the reference containers use most often.

Surfaces follow one canonical geometry.  Across the reference containers 768 of
770 binding records declare ``row = align64(width * element)``,
``plane = height * row``, ``batch = channels * plane`` and an extent of
``planes`` reached from surface index 0.  ``layout="canonical"`` (the default)
requires the descriptor to say exactly that, because that is the window the DMA
programmed in the same image writes; ``layout="declared"`` keeps the caller's
strides instead.
"""

import struct
from dataclasses import dataclass

from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.isa._arith import align_pow2, align_up

HEADER = struct.Struct("<8I")
SEGMENT = struct.Struct("<II16sQQQQiiII")
SECTION = struct.Struct("<16s16sQQ8I")
SYMTAB = struct.Struct("<6I")
SYMBOL = struct.Struct("<IBBHQ")
RELOCATION = struct.Struct("<iI")
IO_GEOMETRY = struct.Struct("<II4I4Q")

MAGIC = 0xBEEFFACE
CPU = 0x80
H14 = 5
FILE_TYPE = 2

LC_SEGMENT_64 = 0x19
LC_SYMTAB = 2

CMD_BUFFER_REFERENCE = 0x40
CMD_OPERATION = 0x04
CMD_COMPILER_METADATA = 0x08
CMD_SOURCE_MODEL_INFO = 0x31

SURFACE_ROW_ALIGN = 64
VM_BASE = 0x30000000
PAGE = 0x4000
TEXT_VM_SIZE = 0x8000
TEXT_CONST_SIZE = 0x4000
TEXT_CONST_ALIGN = 0x40

SEG_PAGEZERO_FLAGS = 0x4
SEG_FVMLIB_FLAGS = 0x6
SEG_TEXT_FLAGS_PLAIN = 0x4
SEG_TEXT_FLAGS_KERNEL = 0x0
SEG_KERNEL_FLAGS = 0x4

SECT_DATA_FLAGS = 0x23
SECT_CONST_FLAGS = 0x21
SECT_TEXT_FLAGS = 0x28
SECT_TEXT_FLAGS_RELOC = 0x128
SECT_TEXT_CONST_FLAGS = 0x26
SECT_KERNEL_FLAGS = 0x26

SECT_ALIGN_IO = 14
SECT_ALIGN_TEXT = 14
SECT_ALIGN_TEXT_CONST = 6
SECT_ALIGN_KERNEL = 6

SOURCE_MODEL_TAG = b"src model info\0\0"
N_SECT = 0x0E
N_EXT = 0x01

BUFFER_REFERENCE_SIZE = 0x20
SEGMENT_RECORD_SIZE = 0x8A0
SEGMENT_RECORD_LAYOUT = 0x222
SEGMENT_RECORD_PROCEDURE_OFFSET = 0x898
BINDING_RECORD_MIN_SIZE = 0xD48
BINDING_RECORD_LAYOUT = 0x34A
BINDING_RECORD_PROCEDURE_OFFSET = 0xD38
BINDING_RECORD_SYMBOL_OFFSET = 0xD3D
BINDING_ROLE_INPUT = 1
BINDING_ROLE_OUTPUT = 2
RESOURCE_SLOTS = 5

ELEMENT_BYTES = {"f16": 2, "f32": 4}
ELEMENT_CODES = {"f16": 5, "f32": 6}

DEFAULT_RECORD_COUNT = 17
DEFAULT_FORMAT_CODE = 0

TOOL_NAME = "tilefoundry ane codegen v1"

# The H14 compiler records the option set it ran with in the 0x08 banner.  These
# are the settings the descriptor stream emitted by this backend assumes: an
# L2-resident cache mode, reuse allocators, one segment, unbounded TD and
# segment counts.  The banner is provenance text, but a complete one keeps the
# record the same shape as the reference containers.
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


def name16(name: bytes, limit: int = 16) -> bytes:
    """Encode a fixed-width Mach-O name field."""
    return name[: limit - 1].ljust(limit, b"\0")


@dataclass(frozen=True)
class IODesc:
    """A named f16 feature map, with shape and strides in elements."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    @property
    def element_bytes(self) -> int:
        """Bytes per element of the declared dtype."""
        if self.dtype not in ELEMENT_BYTES:
            raise ValueError(f"{self.name}: unsupported I/O dtype {self.dtype!r}")
        return ELEMENT_BYTES[self.dtype]

    @property
    def element_code(self) -> int:
        """Element code used by the binding record."""
        if self.dtype not in ELEMENT_CODES:
            raise ValueError(f"{self.name}: unsupported I/O dtype {self.dtype!r}")
        return ELEMENT_CODES[self.dtype]

    @property
    def byte_strides(self) -> tuple[int, int, int]:
        """The batch, plane and row strides in bytes."""
        element = self.element_bytes
        return (self.strides[0] * element, self.strides[1] * element, self.strides[2] * element)

    @property
    def declared_geometry(self) -> tuple[int, int, int, int]:
        """The surface the caller declared: row, plane, batch, total in bytes."""
        batch, plane, row = self.byte_strides
        return row, plane, batch, self.storage_bytes

    @property
    def canonical_geometry(self) -> tuple[int, int, int, int]:
        """The surface size the compiler always declares: row, plane, batch, total.

        Every reference container describes its surfaces this way: rows are
        padded to 64 bytes, the plane step is ``height`` rows, the batch step is
        ``channels`` planes, and the extent is ``extent0 - 1`` batch steps plus
        one plane of channels.  A descriptor whose strides ask for more than
        this describes a buffer the runtime never allocates.
        """
        element = self.element_bytes
        row = align_up(self.shape[3] * element, SURFACE_ROW_ALIGN)
        plane = self.shape[2] * row
        batch = self.shape[1] * plane
        total = (self.shape[0] - 1) * batch + batch
        return row, plane, batch, total

    @property
    def storage_bytes(self) -> int:
        """Bytes spanned by the surface, including any stride padding.

        A layout may give a singleton outer dimension a zero stride, so the
        extent comes from the largest inner block plus the batch steps rather
        than from ``shape[0] * batch``.
        """
        batch, plane, _ = self.byte_strides
        inner = self.shape[1] * plane
        return (self.shape[0] - 1) * batch + inner

    def validate(self, layout: str = "canonical") -> None:
        """Reject I/O descriptors this container cannot describe.

        ``layout="canonical"`` requires the descriptor to describe exactly the
        surface the compiler declares (see ``canonical_geometry``).
        ``layout="declared"`` keeps whatever the caller declared, which is only
        correct when the descriptor and the DMA window already agree.
        """
        if not self.name or not self.name.isascii() or not self.name.isprintable():
            raise ValueError(f"invalid I/O symbol {self.name!r}")
        if self.dtype not in ELEMENT_BYTES:
            raise ValueError(f"{self.name}: unsupported I/O dtype {self.dtype!r}")
        if len(self.shape) != 4 or any(type(n) is not int or not 0 < n < 2**32 for n in self.shape):
            raise ValueError(f"{self.name}: expected four positive I/O extents")
        if len(self.strides) != 4 or any(
            type(n) is not int or not 0 <= n < 2**63 for n in self.strides
        ):
            raise ValueError(f"{self.name}: expected four nonnegative I/O strides")
        if self.storage_bytes <= 0 or self.storage_bytes >= 2**63:
            raise ValueError(f"{self.name}: I/O surface is empty or overflows")
        if layout == "canonical":
            self.require_canonical()
        elif layout != "declared":
            raise ValueError(f"unknown layout policy {layout!r}")
        check = self.shape[3] * self.element_bytes
        cover = self.shape[2] * self.byte_strides[2]
        if self.byte_strides[2] < check or self.byte_strides[1] < cover:
            raise ValueError(f"{self.name}: I/O strides do not cover the tensor")
        if self.shape[0] > 1 and self.byte_strides[0] < self.shape[1] * self.byte_strides[1]:
            raise ValueError(f"{self.name}: batch stride does not cover the channels")

    def require_canonical(self) -> None:
        """Reject a descriptor that asks for more than the canonical surface."""
        row, plane, _batch, total = self.canonical_geometry
        _batch_stride, plane_stride, row_stride = self.byte_strides
        if (row_stride, plane_stride) == (row, plane) and self.storage_bytes == total:
            return
        element = self.element_bytes
        wanted = (
            total // element // self.shape[1] if self.shape[0] > 1 else 0,
            plane // element,
            row // element,
            1,
        )
        raise ValueError(
            f"{self.name}: I/O layout declares a "
            f"{(row_stride, plane_stride, self.storage_bytes)} byte surface, but an "
            f"ANE surface of {self.shape} is {(row, plane, total)} "
            f"(row = align64(width x element), plane = height x row, "
            f"total = channels x plane). Declare strides {wanted} instead: the "
            f"descriptor must match the window the DMA writes."
        )


@dataclass(frozen=True)
class Reloc:
    """A 64-bit placeholder at a word offset from the start of the TD stream."""

    word_offset: int
    symbol: str


@dataclass(frozen=True)
class Section:
    """One section of one segment command."""

    name: bytes
    segment: bytes
    address: int
    size: int
    offset: int
    alignment: int
    flags: int
    reloc_offset: int = 0
    reloc_count: int = 0


@dataclass(frozen=True)
class Segment:
    """One LC_SEGMENT_64 command."""

    name: bytes
    address: int
    size: int
    file_offset: int
    file_size: int
    max_protection: int
    initial_protection: int
    flags: int
    sections: tuple[Section, ...] = ()

    def command(self) -> bytes:
        """Encode the segment command followed by its section commands."""
        out = bytearray(
            SEGMENT.pack(
                LC_SEGMENT_64,
                SEGMENT.size + SECTION.size * len(self.sections),
                name16(self.name),
                self.address,
                self.size,
                self.file_offset,
                self.file_size,
                self.max_protection,
                self.initial_protection,
                len(self.sections),
                self.flags,
            )
        )
        for section in self.sections:
            out += SECTION.pack(
                name16(section.name),
                name16(section.segment),
                section.address,
                section.size,
                section.offset,
                section.alignment,
                section.reloc_offset,
                section.reloc_count,
                section.flags,
                0,
                0,
                0,
            )
        return bytes(out)


@dataclass(frozen=True)
class Surface:
    """One I/O surface with its role, virtual address and section."""

    desc: IODesc
    role: int
    address: int
    size: int

    @property
    def section_name(self) -> bytes:
        """The __FVMLIB section that reserves this surface."""
        return b"__data" if self.role == BINDING_ROLE_OUTPUT else b"__const"

    @property
    def symbol_name(self) -> str:
        """The symbol the __text relocations resolve against."""
        return self.desc.name

    @property
    def tensor_symbol(self) -> str:
        """The label recorded in the binding record."""
        if self.role == BINDING_ROLE_OUTPUT:
            return f"{self.desc.name}@output"
        return self.desc.name


#: Registers that hold a 64-bit coefficient base address and the width of the
#: relocation record that patches them (8 bytes for the paired low/high words,
#: 4 bytes for the single-word forms).
COEFF_SLOT_ADDRS = frozenset({0x1908, 0x19F8, 0x1A00})
COEFF_SINGLE_ADDRS = frozenset({0x19E8, 0x19EC, 0x19F4})
RELOCATION_WIDTH = {addr: 8 for addr in COEFF_SLOT_ADDRS}
RELOCATION_WIDTH.update({addr: 4 for addr in COEFF_SINGLE_ADDRS})


def coeff_slots(program: isa.Program) -> dict[int, int]:
    """Map each coefficient base-address slot's text word offset to its register."""
    slots: dict[int, int] = {}
    start = len(isa.PREAMBLE) // 4
    for descriptor in program.tds:
        first = start + descriptor.header_size_words()
        for packet in descriptor.packets:
            for index, (addr, _) in enumerate(packet.writes):
                if addr in RELOCATION_WIDTH:
                    slots[first + 1 + index] = addr
            first += 1 + len(packet.writes)
        start = align_pow2(start + descriptor.encoded_words(start * 4), 2)
    return slots


#: Coefficient lanes: at most one per engine node.
MAX_COEFF_LANES = 16
#: Every lane starts on a 64-byte boundary.
COEFF_LANE_ALIGN = 64


def coefficient_lanes(shape: tuple[int, int, int, int]) -> int:
    """Count the coefficient lanes a weight of ``shape`` needs."""
    out_channels = shape[0]
    return max(1, min(MAX_COEFF_LANES, out_channels))


def coefficient_layout(
    shape: tuple[int, int, int, int], *, lut_bytes: int = 0
) -> tuple[int, int, int]:
    """Return ``(lanes, per_lane, stride)`` for one coefficient tensor.

    ``per_lane`` output channels go to each lane; a lane's stream walks the input
    channels, then the kernel taps, then the lane's output channels, and the next
    lane starts on a 64-byte boundary.
    """
    out_channels, in_channels, kh, kw = shape
    lanes = coefficient_lanes(shape)
    per_lane = out_channels // lanes
    if per_lane * lanes != out_channels:
        raise ValueError(f"output channels {out_channels} do not split into {lanes} lanes")
    lane_bytes = per_lane * in_channels * kh * kw * 2 + lut_bytes
    stride = align_up(lane_bytes, COEFF_LANE_ALIGN)
    return lanes, per_lane, stride


def coefficient_section(image) -> tuple[bytes, int, tuple[int, ...]]:
    """Pack every coefficient tensor of an image into one section.

    Each tensor is lane-tiled on its own (a tensor whose pass also applies a
    lookup table carries the table behind its weights), and the tensors follow
    each other, so the section offset of each tensor is the extended-packet
    operand of the pass that consumes it.
    """
    coefficients = image.coefficients
    if coefficients is None:
        if image.kernel_shape is None:
            return image.kern, 0, (0,)
        blob, stride = coefficient_blob(image.kernel_shape, image.kern)
        return blob, stride, (0,)
    out = bytearray()
    bases: list[int] = []
    stride = 0
    for shape, values, lut_bytes in coefficients:
        bases.append(len(out))
        packed, stride = coefficient_blob(shape, values, lut_bytes=lut_bytes)
        out += packed
    return bytes(out), stride, tuple(bases)


#: One activated lane: a fixed header, the activation table, then the weights.
#: Offsets are bytes inside the 256-byte lane block.
LUT_SLOTS = (0x00, 0x02, 0x06) + tuple(range(0x08, 0x4E, 2)) + (0x4E, 0x52, 0x54)
LUT_HEADER = {0x00: 0xC91A, 0x02: 0x7C00, 0x04: 0x0000, 0x06: 0x7C00}
LUT_TRAILER = {0x4E: 0x3C00, 0x50: 0x9C00, 0x52: 0x0001, 0x54: 0x0001}
WEIGHT_SLOTS = tuple(range(0x80, 0x100, 2))


def coefficient_blob(
    shape: tuple[int, int, int, int], values: bytes, *, lut_bytes: int = 0,
    lut: bytes | None = None,
) -> tuple[bytes, int]:
    """Pack one coefficient tensor into its per-lane stream, returning it and the stride.

    ``values`` holds the tensor in output-channel major order as little-endian
    16-bit elements.
    """
    out_channels, in_channels, kh, kw = shape
    if len(values) != out_channels * in_channels * kh * kw * 2:
        raise ValueError("coefficient bytes do not match the declared shape")
    lanes, per_lane, stride = coefficient_layout(shape, lut_bytes=lut_bytes)
    word = 2
    if lut is not None and lut_bytes:
        if len(lut) != len(LUT_SLOTS) * 2:
            raise ValueError(f"a lookup table must fill {len(LUT_SLOTS)} slots")
        # The first half of every lane is identical, so it is built once.
        head = bytearray(0x80)
        for offset, value in LUT_HEADER.items():
            struct.pack_into("<H", head, offset, value)
        for offset, value in LUT_TRAILER.items():
            struct.pack_into("<H", head, offset, value)
        for slot, offset in enumerate(LUT_SLOTS):
            head[offset : offset + word] = lut[slot * word : slot * word + word]
        out = bytearray()
        for lane in range(lanes):
            block = bytearray(head)
            block.extend(bytes(stride - len(head)))
            weights = bytearray()
            for ci in range(in_channels):
                for kh_i in range(kh):
                    for kw_i in range(kw):
                        for j in range(per_lane):
                            index = (((lane * per_lane + j) * in_channels + ci) * kh + kh_i) * kw + kw_i
                            weights += values[index * word : index * word + word]
            for slot, offset in enumerate(WEIGHT_SLOTS):
                if slot * word < len(weights):
                    block[offset : offset + word] = weights[slot * word : slot * word + word]
            out += block
        return bytes(out), stride
    out = bytearray()
    for lane in range(lanes):
        lane_bytes = bytearray()
        for ci in range(in_channels):
            for kh_i in range(kh):
                for kw_i in range(kw):
                    for j in range(per_lane):
                        index = (((lane * per_lane + j) * in_channels + ci) * kh + kh_i) * kw + kw_i
                        lane_bytes += values[index * word : index * word + word]
        out += lane_bytes.ljust(stride, b"\0")
    return bytes(out), stride


def address_slots(program: isa.Program) -> dict[int, tuple[int, ...]]:
    """Locate DRAM base pairs in descriptor payloads by register address."""
    addresses = (
        isa.TileDMASrc.tile_dma_src_base_addr_lo.addr,
        isa.TileDMASrc.tile_dma_src2_base_addr_lo.addr,
        isa.TileDMADst.tile_dma_dst_base_addr_lo.addr,
    )
    slots: dict[int, list[int]] = {addr: [] for addr in addresses}
    start = len(isa.PREAMBLE) // 4
    for descriptor in program.tds:
        first = start + descriptor.header_size_words()
        for packet in descriptor.packets:
            for index, ((addr, _), (following, _)) in enumerate(
                zip(packet.writes, packet.writes[1:])
            ):
                if addr in slots and following == addr + 4:
                    slots[addr].append(first + 1 + index)
            first += 1 + len(packet.writes)
        start = align_pow2(start + descriptor.encoded_words(start * 4), 2)
    return {addr: tuple(places) for addr, places in slots.items()}


def slot_roles() -> dict[int, int]:
    """Map each base-address register to the binding role it binds."""
    return {
        isa.TileDMADst.tile_dma_dst_base_addr_lo.addr: BINDING_ROLE_OUTPUT,
        isa.TileDMASrc.tile_dma_src_base_addr_lo.addr: BINDING_ROLE_INPUT,
        isa.TileDMASrc.tile_dma_src2_base_addr_lo.addr: BINDING_ROLE_INPUT,
    }


def buffer_reference(surface: Surface) -> bytes:
    """Encode the 0x40 load command that names one surface and its address."""
    name = surface.desc.name.encode("ascii")
    return (
        struct.pack("<IIQ", CMD_BUFFER_REFERENCE, BUFFER_REFERENCE_SIZE, 0x18)
        + struct.pack("<Q", surface.address)
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
    scratch_address: int = 0,
    scratch_bytes: int = 0,
) -> bytes:
    """Encode the 0x04 segment record: the procedure's VM entry points."""
    if len(resources) > RESOURCE_SLOTS:
        raise ValueError(f"at most {RESOURCE_SLOTS} resources fit the segment record")
    name = procedure.encode("ascii") + b"\0"
    size = max(SEGMENT_RECORD_SIZE, align_up(SEGMENT_RECORD_PROCEDURE_OFFSET + len(name), 8))
    out = bytearray(size)
    struct.pack_into("<IIII", out, 0x000, CMD_OPERATION, size, 0x4, SEGMENT_RECORD_LAYOUT)
    struct.pack_into("<Q", out, 0x010, text_address)
    struct.pack_into("<Q", out, 0x020, text_const_address)
    if scratch_address or scratch_bytes:
        struct.pack_into("<Q", out, 0x040, scratch_address)
    for index, address in enumerate(resources):
        struct.pack_into("<Q", out, 0x050 + 0x10 * index, address)
    struct.pack_into("<Q", out, 0x810, text_address)
    struct.pack_into("<I", out, 0x820, 4)
    struct.pack_into("<I", out, 0x824, task_words)
    struct.pack_into("<I", out, 0x830, task_count)
    struct.pack_into("<II", out, 0x838, 0xFFFFFFFF, 0xFFFFFFFF)
    struct.pack_into("<I", out, 0x844, 0xFFFF)
    struct.pack_into("<I", out, 0x848, SEGMENT_RECORD_PROCEDURE_OFFSET)
    struct.pack_into("<I", out, 0x858, scratch_bytes)
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


def binding_record(surface: Surface, procedure: str, declared: bool = False) -> bytes:
    """Encode one 0x04 binding record: the shape and strides of a surface."""
    desc = surface.desc
    symbol = surface.tensor_symbol.encode("ascii") + b"\0"
    short = desc.name.encode("ascii") + b"\0"
    symbol_offset = BINDING_RECORD_SYMBOL_OFFSET
    short_offset = symbol_offset + len(symbol)
    size = max(BINDING_RECORD_MIN_SIZE, align_up(short_offset + len(short), 8))
    row, plane, batch, total = desc.declared_geometry if declared else desc.canonical_geometry
    element = desc.element_bytes
    out = bytearray(size)
    struct.pack_into("<IIII", out, 0x000, CMD_OPERATION, size, 3, BINDING_RECORD_LAYOUT)
    struct.pack_into("<II", out, 0x010, 3, surface.role)
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
    size = align_up(8 + len(payload), 8)
    return struct.pack("<II", command, size) + payload.ljust(size - 8, b"\0")


def compiler_metadata(
    tool: str,
    version: str,
    kernel_bytes: int,
    surface_bytes: int,
    task_count: int,
    flags: tuple[tuple[str, str | None], ...] = TARGET_PROFILE,
) -> bytes:
    """Encode the 0x08 command: the compiler banner.

    The reference containers carry a plain UTF-8 banner here: the compiler and
    front-end versions, then the option set the compiler ran with, then the
    source and output paths.  Nothing in it is read back by the loader, but a
    complete banner keeps the record the same shape as the reference files.
    """
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
            "\t--surfaces=%d" % surface_bytes,
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


#: Element-type catalog every compiled image carries, in symbol-table order.
ELEMENT_CATALOG = (
    "void:t1=1",
    "int8:t2=r2;0;127",
    "uint8:t3=r1;0;255",
    "int16:t4=r1;-32768;32767",
    "float16:t5=r1;2;0",
    "float:t6=r1;4;0",
    "raw10:t7=r1;-512;511",
    "lut:t8=",
    "uint4:t9=r1;0;15",
    "uint6:t10=r1;0;63",
    "uint1:t11=r1;0,1",
    "uint2:t12=r1;0;3",
    "packed10:t13=r1;-512;511",
    "y10:t14=r1;-512;511",
    "uint16:t15=r1;0;65535",
    "e4m3:t16=r1;1;0",
    "int32:t17=r1;-2147483648;2147483647",
    "uint32:t18=r1;0;4294967295",
    "int64:t19=r1;-9223372036854775808;9223372036854775807",
    "uint64:t20=r1;0;18446744073709551615",
    "raw12:t21=r1;-512;511",
    "y12:t22=r1;-512;511",
    "int4:t23=r1;-8;7",
)

#: Type ids past the catalog: five per surface (four axis strides and the shape).
FIRST_SURFACE_TYPE = 1 + len(ELEMENT_CATALOG)
AXES = "nchw"
#: Private symbol types the compiler uses for the catalog and the shape records.
CATALOG_TYPE = 0x80
SHAPE_TYPE = 0x20


def shape_stab(desc: "IODesc", base: int) -> str:
    """Encode one surface's element type and its per-axis byte strides.

    The record is ``name:t<base>=`` followed by one ``ar1;0;<extent>;<id>=s<stride><axis>``
    group per axis (batch, channel, height, width) and the element code last.
    """
    row, plane, batch, _total = desc.canonical_geometry
    shape = desc.shape
    strides = (batch, plane, row, desc.element_bytes)
    groups = []
    for axis, extent, stride in zip(AXES, shape, strides):
        groups.append(f"ar1;0;{extent};{base + 1 + len(groups)}=s{stride}{axis}")
    return f"{desc.name}:t{base}=" + ":".join(groups) + f":{desc.element_code}"


def runtime_version() -> bytes:
    """Encode the runtime-version command: a four-byte 1.1 payload."""
    payload = bytearray(SOURCE_MODEL_TAG[:0] + b"rt.version".ljust(16, b"\0"))
    text = struct.pack("<4B", 1, 0, 1, 0)
    payload += struct.pack("<QQ", 0x28, len(text))
    payload += text
    return text_record(CMD_SOURCE_MODEL_INFO, bytes(payload))


@dataclass(frozen=True)
class Image:
    """A complete H14 image: descriptor text, kernel blob, I/O, relocations."""

    text: bytes
    kern: bytes
    io: tuple[IODesc, ...]
    relocs: tuple[Reloc, ...]
    procedure: str = "main"
    output_count: int | None = None
    kernel_reloc: tuple[int, ...] = ()
    scratch_bytes: int = 0
    record_count: int = DEFAULT_RECORD_COUNT
    format_code: int | None = None
    kernel_symbol: str = "coeff"
    kernel_digest: str | None = None
    kernel_shape: tuple[int, int, int, int] | None = None
    coefficients: tuple[tuple[tuple[int, int, int, int], bytes, int], ...] | None = None
    tool: str = TOOL_NAME
    version: str = "1.0"
    layout: str = "canonical"

    def surfaces(self, program: isa.Program) -> tuple[Surface, ...]:
        """Order the I/O descriptors outputs first, then inputs, with addresses.

        Roles come from the TD stream where it binds a symbol to a DMA base
        register.  A caller that passes no relocations must say how many of the
        trailing descriptors are outputs.
        """
        outputs: tuple[IODesc, ...]
        inputs: tuple[IODesc, ...]
        if self.output_count is None:
            roles = slot_roles()
            by_word: dict[int, int] = {
                word: roles[addr]
                for addr, words in address_slots(program).items()
                for word in words
            }
            declared: dict[str, int] = {
                reloc.symbol: by_word.get(reloc.word_offset, BINDING_ROLE_INPUT)
                for reloc in self.relocs
            }
            outputs = tuple(
                item for item in self.io if declared.get(item.name) == BINDING_ROLE_OUTPUT
            )
            inputs = tuple(
                item for item in self.io if declared.get(item.name) != BINDING_ROLE_OUTPUT
            )
        else:
            count = self.output_count
            if type(count) is not int or not 0 <= count <= len(self.io):
                raise ValueError(f"output count {count!r} does not fit {len(self.io)} surfaces")
            split = len(self.io) - count
            inputs = tuple(self.io[:split])
            outputs = tuple(self.io[split:])
        surface_list: list[Surface] = []
        address = VM_BASE + align_up(self.scratch_bytes, PAGE)
        for role, group in ((BINDING_ROLE_OUTPUT, outputs), (BINDING_ROLE_INPUT, inputs)):
            for item in group:
                size = align_up(item.storage_bytes, PAGE)
                surface_list.append(Surface(item, role, address, size))
                address += size
        return tuple(surface_list)

    def relocation_table(
        self, text: bytes, program: isa.Program, kernel_section: int | None
    ) -> bytes:
        """Encode the __text relocation records for the coefficient blob only.

        An I/O surface address is *not* relocated here: the descriptor states the
        surface base as a flagged two-word packet whose operands stay zero, and
        the loader binds them.  A coefficient blob is different: it is resolved
        by a section-relative reference into __kern_0, which is the only kind of
        Mach-O relocation a compiled image carries.
        """
        records = bytearray()
        if self.kernel_reloc and kernel_section is None:
            raise ValueError("kernel relocations need a kernel section")
        slots = coeff_slots(program)
        # A multi-tensor image derives its records from the parsed stream, whose
        # word offsets already include the alignment gaps between descriptors.
        words = sorted(slots) if self.coefficients is not None else self.kernel_reloc
        used = {record * 4 for record in (reloc.word_offset for reloc in self.relocs)}
        for word in words:
            if type(word) is not int or word < 0:
                raise ValueError(f"kernel relocation word {word!r} is not a word offset")
            off = word * 4
            width = RELOCATION_WIDTH.get(slots.get(word))
            if width is None:
                raise ValueError(f"kernel relocation word {word} does not address a coefficient slot")
            if off + width > len(text):
                raise ValueError(f"kernel relocation at word {word} leaves the descriptor")
            if off in used:
                raise ValueError(f"overlapping kernel relocation at word {word}")
            # A section-relative relocation keeps its addend in place, so the
            # operand may name an offset inside the coefficient section.
            used.add(off)
            flags = (1 << 24) | ((width.bit_length() - 1) << 25) | kernel_section
            records.extend(RELOCATION.pack(off, flags))
        return bytes(records)

    def to_bytes(self) -> bytes:
        """Write the little-endian container without binding buffer addresses."""
        if len(self.text) % 4:
            raise ValueError("image text must be a whole number of words")
        if not self.procedure or not self.procedure.isascii() or not self.procedure.isprintable():
            raise ValueError(f"invalid procedure name {self.procedure!r}")
        if type(self.record_count) is not int or not 0 <= self.record_count < 2**32:
            raise ValueError(f"record count {self.record_count!r} is not a u32")
        if self.format_code is not None and (
            type(self.format_code) is not int or not 0 <= self.format_code < 2**32
        ):
            raise ValueError(f"format code {self.format_code!r} is not a u32")
        if type(self.scratch_bytes) is not int or not 0 <= self.scratch_bytes < 2**32:
            raise ValueError(f"scratch size {self.scratch_bytes!r} is not a u32")
        has_kern = bool(self.kern) or self.coefficients is not None
        program = isa.Program.from_bytes(self.text)
        if not program.tds:
            raise ValueError("image text must contain a descriptor")
        # The record states the largest per-descriptor cycle estimate.
        format_code = self.format_code
        if format_code is None:
            format_code = max(
                (descriptor.header.get(0x04, 0) for descriptor in program.tds), default=0
            )
        if len({item.name for item in self.io}) != len(self.io):
            raise ValueError("duplicate image I/O symbol")
        for item in self.io:
            item.validate(self.layout)

        surfaces = self.surfaces(program)
        text_address = (
            VM_BASE + align_up(self.scratch_bytes, PAGE) + sum(surface.size for surface in surfaces)
        )
        text_const_address = align_up(text_address + len(self.text), TEXT_CONST_ALIGN)
        if text_const_address + TEXT_CONST_SIZE > text_address + TEXT_VM_SIZE:
            raise ValueError("__TEXT cannot hold its const section")
        kernel_address = text_address + TEXT_VM_SIZE
        blob, lane_stride, tensor_bases = coefficient_section(self)
        # A coefficient section keeps at least two pages so a small blob cannot
        # be addressed past the end of the segment.
        kernel_size = max(2 * PAGE, align_up(len(blob), PAGE))
        # The resource table lists the surfaces in section order, then the
        # coefficient blob; the blob is dropped when the table is full.
        resources = [surface.address for surface in surfaces]
        if has_kern and len(resources) < RESOURCE_SLOTS:
            resources.append(kernel_address)

        command_payloads = tuple(buffer_reference(surface) for surface in surfaces)
        segment_payload = segment_record(
            text_address,
            text_const_address,
            tuple(resources),
            len(self.text) // 4,
            len(program.tds),
            self.procedure,
            self.record_count,
            format_code,
            VM_BASE if self.scratch_bytes else 0,
            self.scratch_bytes,
        )
        binding_surfaces = tuple(
            surface for surface in surfaces if surface.role == BINDING_ROLE_INPUT
        ) + tuple(surface for surface in surfaces if surface.role == BINDING_ROLE_OUTPUT)
        binding_payloads = tuple(
            binding_record(surface, self.procedure, self.layout == "declared")
            for surface in binding_surfaces
        )
        surface_bytes = sum(surface.size for surface in surfaces)
        metadata_payload = compiler_metadata(
            self.tool, self.version, len(blob), surface_bytes, len(program.tds)
        )
        source_payload = source_model_info(
            self.tool, self.version, self.procedure, len(program.tds)
        )
        runtime_payload = runtime_version()

        symtab_size = SYMTAB.size
        command_size = (
            SEGMENT.size
            + len(surfaces) * (SEGMENT.size + SECTION.size)
            + SEGMENT.size
            + 2 * SECTION.size
            + (SEGMENT.size + SECTION.size if has_kern else 0)
            + sum(len(payload) for payload in command_payloads)
            + len(segment_payload)
            + sum(len(payload) for payload in binding_payloads)
            + len(metadata_payload)
            + len(source_payload)
            + len(runtime_payload)
            + symtab_size
        )
        command_count = (
            1
            + len(surfaces)
            + 1
            + (1 if has_kern else 0)
            + len(command_payloads)
            + 1
            + len(binding_payloads)
            + 1
            + 1
            + 1
            + 1
        )
        kernel_section = len(surfaces) + 3 if has_kern else None
        symbols, strings = self.symbol_table(surfaces, kernel_section)
        reloc_data = self.relocation_table(self.text, program, kernel_section)
        reloc_count = len(reloc_data) // RELOCATION.size

        text_file_offset = self.file_offset(
            HEADER.size + command_size + len(symbols) + len(strings) + len(reloc_data),
            len(surfaces),
        )
        if text_file_offset is None:
            raise ValueError("image metadata does not fit below the __TEXT segment")
        symbol_offset = HEADER.size + command_size
        string_offset = symbol_offset + len(symbols)
        reloc_offset = string_offset + len(strings) if reloc_data else 0
        if reloc_offset + len(reloc_data) > text_file_offset:
            raise ValueError("image symbols and relocations cross the __TEXT segment")
        text_const_file_offset = text_file_offset + align_up(len(self.text), TEXT_CONST_ALIGN)
        if text_const_file_offset + TEXT_CONST_SIZE > text_file_offset + TEXT_VM_SIZE:
            raise ValueError("__TEXT cannot hold its const section on disk")
        kernel_file_offset = text_file_offset + TEXT_VM_SIZE

        segments = [self.pagezero()]
        for surface in surfaces:
            segments.append(
                Segment(
                    b"__FVMLIB",
                    surface.address,
                    surface.size,
                    0,
                    0,
                    2 if surface.role == BINDING_ROLE_OUTPUT else 1,
                    2 if surface.role == BINDING_ROLE_OUTPUT else 1,
                    SEG_FVMLIB_FLAGS,
                    (
                        Section(
                            surface.section_name,
                            b"__FVMLIB",
                            surface.address,
                            surface.desc.storage_bytes,
                            0,
                            SECT_ALIGN_IO,
                            SECT_DATA_FLAGS
                            if surface.role == BINDING_ROLE_OUTPUT
                            else SECT_CONST_FLAGS,
                        ),
                    ),
                )
            )
        segments.append(
            Segment(
                b"__TEXT",
                text_address,
                TEXT_VM_SIZE,
                text_file_offset,
                TEXT_VM_SIZE,
                5,
                5,
                SEG_TEXT_FLAGS_KERNEL if has_kern else SEG_TEXT_FLAGS_PLAIN,
                (
                    Section(
                        b"__text",
                        b"__TEXT",
                        text_address,
                        len(self.text),
                        text_file_offset,
                        SECT_ALIGN_TEXT,
                        SECT_TEXT_FLAGS_RELOC if reloc_count else SECT_TEXT_FLAGS,
                        reloc_offset,
                        reloc_count,
                    ),
                    Section(
                        b"__const",
                        b"__TEXT",
                        text_const_address,
                        TEXT_CONST_SIZE,
                        text_const_file_offset,
                        SECT_ALIGN_TEXT_CONST,
                        SECT_TEXT_CONST_FLAGS,
                    ),
                ),
            )
        )
        if has_kern:
            segments.append(
                Segment(
                    b"__KERN_0",
                    kernel_address,
                    kernel_size,
                    kernel_file_offset,
                    kernel_size,
                    1,
                    1,
                    SEG_KERNEL_FLAGS,
                    (
                        Section(
                            b"__kern_0",
                            b"__KERN_0",
                            kernel_address,
                            len(blob),
                            kernel_file_offset,
                            SECT_ALIGN_KERNEL,
                            SECT_KERNEL_FLAGS,
                        ),
                    ),
                )
            )

        commands = bytearray()
        for segment in segments:
            commands += segment.command()
        for payload in command_payloads:
            commands += payload
        commands += segment_payload
        for payload in binding_payloads:
            commands += payload
        commands += metadata_payload
        commands += source_payload
        commands += runtime_payload
        commands += SYMTAB.pack(
            LC_SYMTAB,
            SYMTAB.size,
            symbol_offset,
            len(symbols) // SYMBOL.size,
            string_offset,
            len(strings),
        )
        if len(commands) != command_size:
            raise ValueError("command area size disagrees with its plan")

        out = bytearray(text_file_offset + TEXT_VM_SIZE + kernel_size)
        out[: HEADER.size] = HEADER.pack(
            MAGIC, CPU, H14, FILE_TYPE, command_count, command_size, 0x00200000, 0
        )
        if command_count != len(segments) + len(command_payloads) + len(binding_payloads) + 5:
            raise ValueError("command count disagrees with its plan")
        out[HEADER.size : HEADER.size + command_size] = commands
        out[symbol_offset : symbol_offset + len(symbols)] = symbols
        out[string_offset : string_offset + len(strings)] = strings
        out[reloc_offset : reloc_offset + len(reloc_data)] = reloc_data
        out[text_file_offset : text_file_offset + len(self.text)] = self.text
        if has_kern:
            out[kernel_file_offset : kernel_file_offset + len(blob)] = blob
        return bytes(out)

    @staticmethod
    def pagezero() -> Segment:
        """Encode the guard page Mach-O requires below the image."""
        return Segment(b"__PAGEZERO", 0, PAGE, 0, 0, 0, 0, SEG_PAGEZERO_FLAGS)

    @staticmethod
    def file_offset(metadata_end: int, surface_count: int) -> int | None:
        """Pick the first segment file offset that clears the metadata.

        The compiler places __TEXT at 0x4000 when a program has at most two
        surfaces and at 0x8000 otherwise, then steps further out only when the
        metadata would not clear it.
        """
        candidates = (PAGE, 2 * PAGE, 4 * PAGE, 8 * PAGE)
        first = 0 if surface_count <= 2 else 1
        for candidate in candidates[first:]:
            if metadata_end <= candidate:
                return candidate
        return None

    def symbol_table(
        self, surfaces: tuple[Surface, ...], kernel_section: int | None = None
    ) -> tuple[bytes, bytes]:
        """Encode the surface symbols, the element catalog and the shape records.

        Three groups follow each other: one external symbol per surface in section
        order, the element-type catalog shared by every image, and one shape record
        per surface whose type ids continue past the catalog.
        """
        names = bytearray(b"\0")
        symbols = bytearray()
        for index, surface in enumerate(surfaces):
            symbols.extend(SYMBOL.pack(len(names), N_SECT | N_EXT, index + 1, 0, surface.address))
            names.extend(surface.symbol_name.encode("ascii") + b"\0")
        if self.kern or self.coefficients:
            if kernel_section is None:
                raise ValueError("a coefficient blob needs its section ordinal")
            address = VM_BASE + align_up(self.scratch_bytes, PAGE) + sum(
                surface.size for surface in surfaces
            ) + TEXT_VM_SIZE
            if self.coefficients is None:
                groups = ((self.kernel_shape, 0, 0),)
            else:
                _blob, _stride, bases = coefficient_section(self)
                groups = tuple(
                    (shape, bases[index], lut) for index, (shape, _bytes, lut) in enumerate(self.coefficients)
                )
            token = self.kernel_digest or self.kernel_symbol
            for index, (shape, base, lut) in enumerate(groups):
                lanes, _per_lane, stride = coefficient_layout(shape, lut_bytes=lut)
                prefix = f"{token}_{index}" if len(groups) > 1 else token
                for lane in range(lanes):
                    symbols.extend(
                        SYMBOL.pack(
                            len(names),
                            N_SECT | N_EXT,
                            kernel_section,
                            0x0002 if lane == 0 else 0,
                            address + base + lane * stride,
                        )
                    )
                    names.extend(f"{prefix}_ne_{lane}".encode("ascii") + b"\0")
        for entry in ELEMENT_CATALOG:
            symbols.extend(SYMBOL.pack(len(names), CATALOG_TYPE, 0, 0, 0))
            names.extend(entry.encode("ascii") + b"\0")
        for index, surface in enumerate(surfaces):
            base = FIRST_SURFACE_TYPE + 5 * index
            symbols.extend(SYMBOL.pack(len(names), SHAPE_TYPE, 0, base & 0x1F, 0))
            names.extend(shape_stab(surface.desc, base).encode("ascii") + b"\0")
        return bytes(symbols), bytes(names)


__all__ = [
    "IODesc",
    "Image",
    "Reloc",
    "Section",
    "Segment",
    "Surface",
    "address_slots",
]
