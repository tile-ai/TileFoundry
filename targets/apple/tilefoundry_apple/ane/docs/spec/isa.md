# TileFoundry ANE Spec — ISA

The H14 byte layer represents configuration writes as task descriptors (TDs).
It encodes words and shares feature-map geometry with HWX and runtime bindings.

## 1. Fields and blocks

```python
class BitField:
    """An inclusive bit range in a word at a byte address."""

    addr: int
    hi: int
    lo: int

    def get(self, word: int) -> int: ...
    def put(self, word: int, val: int) -> int: ...


class Block:
    """A contiguous range of configuration words with named fields."""

    base: ClassVar[int]
    end: ClassVar[int]
    fields: ClassVar[dict[str, BitField]]

    def pack(self, **name_val: int) -> Writes: ...
```

- constraints:
  - `get` MUST return unshifted bits. `put` MUST preserve bits outside the field.
  - A field value MUST fit its bit range and satisfy any declared value restrictions.
  - A field for a floating-point operand takes its integer bit pattern.
  - A block covers aligned word addresses in the half-open byte range `[base, end)`.
  - `Block.pack` MUST combine named fields into words that start at zero.
  - `Writes` MUST retain the mask of every explicitly assigned field, including zero values.

## 2. Packets

```python
def serialize(image: Mapping[int, int]) -> list[Packet]: ...
```

- constraints:
  - `serialize` MUST reject addresses outside the aligned configuration words.
  - `serialize` MUST keep each packet inside one block and order its values by increasing address.
  - Direct `Packet` encoding MUST reject addresses outside a declared block or crossing blocks.
  - RUN carries up to 64 consecutive words. MASK selects words within 16 words after its first word.
  - At each lowest pending address, compare the RUN and MASK candidates. Choose RUN on a tie.

## 3. Intrinsics

```python
class TensorDesc:
    """A feature map's memory layout as the DMA and the L2 buffer address it."""

    dtype: str
    shape: tuple[int, int, int, int]
    strides: tuple[int, int, int, int]

    def from_shape(cls, dtype: str, shape: tuple[int, int, int, int]) -> "TensorDesc": ...
    def validate(self, layout: str = "canonical") -> None: ...
    def validate_feature(self) -> None: ...


def load(source: TensorDesc) -> dict[int, int]: ...
def store(destination: TensorDesc) -> dict[int, int]: ...
def compute_pe(kind: TaskType, output: TensorDesc, **operands) -> dict[int, int]: ...
def compute_ne(op: NEOp, **operands) -> dict[int, int]: ...
def splice(*parts: Mapping[int, int]) -> Writes: ...
```

- constraints:
  - `isa/tensor_desc.py` MUST own the single feature-map description: dtype (`f16` or `f32`),
    NCHW shape and strides in elements. Names and addresses belong to I/O bindings.
  - `from_shape` MUST derive canonical strides: rows align to 64 bytes, a channel plane spans
    one height of rows, and a batch spans all channel planes.
  - Canonical validation MUST compare only axes with extent greater than one. Byte strides
    and storage size MUST fill singleton axes from the padded inner geometry.
  - Declared layouts MUST have unit W stride for non-singleton W, non-overlapping rows,
    planes and batches, and a positive storage extent below 2**63 bytes.
  - Tile staging MUST derive L2 pitch from the scheduled width without changing the logical
    shape or DRAM strides stored in `TensorDesc`. Intrinsics MUST receive fp16, N=1 feature maps.
  - `TensorDesc.validate_feature` MUST own this dtype/batch precondition for intrinsics,
    TIR tensor conversion and template entry points. General descriptor geometry still admits f32
    and multiple batches.
  - `load` MUST program the staged source buffer and the source DMA of one feature map.
  - `store` MUST program the staged result buffer and the result DMA of one feature map.
  - `compute_pe` MUST program the planar engine options and declare the task kind.
  - `compute_ne` MUST program the neural engine op and MUST arm the kernel DMA path.
  - A primitive MUST build its words from declared fields only. It MUST NOT shift a value into
    position by hand.
  - DRAM bases MUST remain relocatable addends; L2 bases MUST use the planned byte offsets.
  - `splice` MUST combine disjoint field writes and accept repeated equal assignments.
    Different values on overlapping masks MUST raise `ValueError`, including explicit zero
    assignments. Nested splices MUST preserve masks. A plain mapping assigns complete words.

## 4. Default image

```python
RESET_IMAGE: Mapping[int, int]
```

- constraints:
  - `isa.fields.RESET_IMAGE` MUST own reset words; codegen `DEFAULT_IMAGE` aliases it.
  - The image MUST hold the value every descriptor starts from, so a word that still holds it is
    not restated.
  - The image MUST NOT hold the node id, which every descriptor writes even at its default.

## 5. Task descriptors

```python
class TD:
    """A task descriptor: a header followed by an ordered packet sequence."""

    def from_values(
        cls, values: Mapping[BitField | int, int], *, base: Mapping[int, int] | None = None, tid: int = 0
    ) -> "TD": ...
    def to_bytes(self) -> bytes: ...
    def writes(self) -> dict[int, int]: ...
    def size_words(self) -> int: ...
```

- constraints:
  - A TD MUST retain its task ID and header as immutable construction values.
  - The header carries the task ID and word count. A dependency flag adds a dependency ID word.
  - `from_values` MUST apply raw words before fields and reject invalid configuration word addresses.
  - Raw words MUST satisfy every named field domain, including before reset-value elision.
  - Nonzero reserved bits MUST be rejected as not decoded yet.
  - `validate` MUST check every packet and the effective register map from `RESET_IMAGE` plus
    this TD's writes, including flagged address addends. No preceding TD supplies reset values.
  - For field-only writes, `from_values` MUST start from the base word or zero if it is absent.
  - `from_values` MUST omit resulting words equal to words present in `base`.
  - `writes` MUST report literal writes. A later literal write to the same address takes priority.
  - `size_words` MUST count the header and packet words, not alignment padding.
  - `to_bytes` MUST encode little-endian words without stream framing.

## 6. Program stream

```python
class Program:
    """One stream of descriptors with a preamble and alignment padding."""

    tds: tuple[TD, ...]

    def to_bytes(self) -> bytes: ...
```

- constraints:
  - The stream MUST start with one preamble. Each descriptor MUST start at a 16-byte boundary.
  - The stream MUST fill gaps between descriptors with zero bytes and end at the last descriptor word.

## 7. Headers, reset image and packet framing

A task header uses an address space separate from the configuration registers.
Its first word carries task ID, task size and flags. Byte offsets `0x04`, `0x08`,
`0x10`, `0x18` and `0x1C` hold estimated cycles, event mask, debug-event mask,
DRAM event mask and control flags. A dependency sets control bit 1 and adds a
ninth header word with the dependency ID. Codegen preserves that dependency
when replacing a descriptor's header.

The default control word enables task status. Codegen leaves cycle and DRAM event
estimates at zero; they need network cost information beyond stream position.
Convolution tasks set engine-enable bits 16–18. For active node counts 1, 2,
3–4, 5–8 and above 8, the codes are respectively 1, 2, 3, 4 and 5; zero nodes
leaves the code clear. Codegen defaults to 16 nodes for neural-engine tasks.

| Position | Event mask | Debug mask | DRAM word |
|---|---|---|---|
| Single | `0x2A` | `0x00FFF868` | 0 |
| First | `0x08` | `0x00FFF868` | 0 |
| Middle | 0 | `0x00FFF800` | 0 |
| Last | `0x22` | `0x00FFF868` | 0 |

The debug-mask variant `0x68` selects event groups and remains outside the
positional default. Each TD starts from the reset configuration independently;
word elision never inherits a prior TD's writes. The node-ID word is absent from
the reset image so it is always stated.

A packet's 32-bit header holds a base word address in bits 0–14. RUN uses bits
15–20 for count minus one and bits 21–30 for relocation flags. MASK sets bit 31
and uses bits 15–30 as the window after its first word. The emission pipeline
orders coefficient DMA, common configuration, source DMA, L2, PE, NE and
result DMA. Extended coefficient address pairs precede source, second-source
and destination pairs; single-word extended slots follow the pairs.

The task-size field holds at most 2047 words. Streams begin with the 16-byte
preamble `01000000000000000000000000000000`. A descriptor near a 64-byte boundary
can receive marker packets before its extended address packets. If its end,
rounded to 16 bytes, is on that boundary, a gap shorter than 8 bytes gets one
8-byte marker packet, a gap of 8–15 bytes gets two, and a larger gap gets none.
Each marker writes `0xDEAD` through the destination marker field.

Coefficient register addresses are declared once as `KernelDMA` fields.
`COEFFICIENT_RELOCATIONS` maps their low-word addresses to patch widths: three
64-bit pairs and three 32-bit singles. Extended-packet flags are a separate
encoding choice, so the serializer uses the applicable subset of these slots.
`serialize` takes the flags per base register; `RELOCATION_SLOTS` is the default.

## 8. Field formulas and staging

`rules.channel_groups` computes the log2 number of 16-channel groups, returning
zero for one group. `active_nodes` stores node count minus one. `shift` clamps
ceil-log2 of the work dimension to 0–4. `unicast_channels` rounds the number of
16-channel groups up to a power of two and caps it at 8; `rules.check` requires it,
and fill-lower-nodes exactly when it exceeds 1, on a transposing bypass. These
formulas are supported by compiler disassembly or descriptor comparisons.

With channel step `ch` and channel count `channels`, the staged depth is
`channels × ch`. The L2 row stride is that depth on both engines: the device
computes identical results for every row stride that covers the depth, so no
guard band is programmed.

Feature-map DRAM rows align to the shared `SURFACE_ROW_ALIGN` of 64 bytes;
L2 rows align to 16. A singleton extent causes the DMA row-stride field to
remain zero. A narrower staged tile keeps the logical feature-map geometry
and DRAM strides while changing its L2 pitch. Pitch generation and
effective-register validation share `TensorDesc.staging_row` and the constraints
in [§9](#9-verified-transfer-validation).

The neural width code is a power-of-two subdivision of 256 bytes. The dense row
MUST divide 256 exactly before its ratio is tested for a power of two. An explicit
code MUST lie in 0–4 and equal the shape-derived code. Other row subdivisions are
rejected as not decoded yet; contradictory explicit codes are errors. A transposing
bypass programs the preferred, minimum and maximum shift slices of `0x0038` to the
code; an identity bypass programs only the maximum slice.
A convolution reading from DRAM programs its kernel height and width, horizontal
and vertical stride, and left/top padding through the declared kernel-window
fields. Its output offsets are both one. A bias selects `0x0D04[4] = 1` and is read
from the coefficient lanes ahead of the weights. ReLU selects the native nonlinear mode
and reserves no lookup-table bytes in the coefficient stream. A lookup-table
nonlinear selects the listed mode `0x0D04[17:16] = 2`, leaves `0x0D00`, the LUT
enable `0x19E4` and its size and bases at reset, and reads the 128-byte table at
the start of every coefficient lane, so the lane offsets and buffer sizes cover
the table and the weights.

## 9. Verified transfer validation

`isa/dma.py` decodes enabled tile DMA from access types and routing, independently
of base-word presence. It accepts DMA_READ source/result, an enabled destination,
source routing 1 for unary or 3 for binary, and second-stream plus PE second-source
2 (add, multiply, min, max) or 3 (subtract: first minus second source) for binary.
Complete-height unary and neural-engine continuation additionally accepts L2_READ
source, result or both, and so does a binary whose two sources are both L2_READ,
with the second stream off and source routing 1. A descriptor with neither DMA nor
a second stream that programs an L2 channel or row pitch keeps both in L2; its
geometry comes from the common block alone.
An L2 result MUST disable destination DMA and buffer mode; an L2 source MUST omit
source DMA. Its channel and row pitches MUST be explicit; depth/group pitches
and source DMA format MUST stay at reset zero, matching the decoded
`load_chained` form. Other access/routing combinations are rejected as not decoded yet.

The verified transfer format is fp16, N=1, depth=1, common formats 2 and tile format
`(format_mode, trunc, shift, memory_format, offset_ch, interleave, cmp_vec)` equal to
`(1, 3, 0, 2, 0, 1, 0)`. Destination FIFO and both zero-pad flags are one.
Depth/group DMA strides are zero. Other formats, transpose controls and padded
footprints other than the convolution and pools below are rejected as not decoded
yet. The transfer extent covers effective elements; it does not generalize to other
compressed or tiled encodings.

Source and second-source geometry is `(cin, hin, win)`. Destination geometry is
`(cout, hout, wout)`, except spatial reduction writes `(cout, 1, 1)` while declaring
the input spatial geometry. The verified NE bypass C/W transpose declares source
geometry at both ends, uses the unicast count and fill-lower-nodes that
`rules.check` relates to its channels, and writes
`(wout, hout, cout)`. DMA stride fields are in 64-byte units. A zero row stride in
a singleton geometry uses its canonical padded row. Rows MUST cover width × 2;
channel strides MUST cover height × row. `tile_height` MUST lie in `[1, hout]`.
Elementwise declared geometries MUST agree; pool/conv output dimensions MUST
match their positive kernel window, strides and padding. Only top and left padding
are encoded; the output dimensions imply equal bottom and right padding. The decoded
padded windows are an NE convolution with a 3×3 kernel, stride 1 and padding 1, and
the pools that the Pool emitter admits; the source DMA reads the unpadded
`(cin, hin, win)` map. Other padded windows are rejected as not decoded yet.

`L2_CAPACITY` is defined once in `isa/dma.py` as 2 MiB. For each pass, source,
second-source and result intervals MUST end within that capacity and MUST be
disjoint. L2 byte bases and strides are the fields multiplied by 16. The checked
envelope reserves complete staged rows. A row stride MUST cover the staged depth
`channels × align16(width × 2)`; channel, depth and group strides MUST equal the
staged layout. Pool source rows include the kernel halo, up to the source height;
reduction reserves one result row. A DMA result that leaves its row stride zero occupies the depth, or
one source row for a reduction. An L2-only result MUST explicitly program its
channel and row pitches to the staging geometry that its consumer uses. Depth
and group extents remain one.
Every overlapping live interval MUST be rejected, including equal-geometry 1×1
convolution source/result intervals with alias bits set. Alias bits do not establish
permission to overlap buffers. Their existing encoding is retained.
The result address field is `L2.l2_result_address` at `0x0534[20:4]` in 16-byte units.
`validation.validate` MUST also
require each L2 source to match an earlier result interval and shape that no
intervening DMA staging or result write has overwritten.

Coefficient lane count MUST equal the number of enabled coefficient DMA slots.
The emitter checks its planned count and the container checks the effective slot
enable fields against the bound coefficient layout. Unequal counts are rejected
as not decoded yet; packing and slot programming for equal counts stay unchanged.


## 10. Runtime storage and host tensors

The C ABI `TensorDesc.storage_bytes` MUST be emitted from the Python descriptor's
`storage_bytes`: `(N - 1) * byte_batch_stride + C * byte_channel_stride`, including
complete padded channel planes. The launcher rounds that reservation to 16 KiB
and MUST verify `IOSurfaceGetAllocSize` covers it. It MUST NOT substitute the last
logical element's span for this reservation.

Host I/O tensors MUST be CPU device 0 with matching fp16 dtype, four-dimensional
shape and nonnegative strides. Positive noncontiguous strides and byte offsets
are supported. Pointer, count, shape and index-overflow checks precede host copies.
Weight writers require contiguous OIHW CPU fp16 tensors; Python makes a contiguous
snapshot after checking device and dtype.
