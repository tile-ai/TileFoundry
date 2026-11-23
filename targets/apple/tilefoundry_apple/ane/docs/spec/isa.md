# TileFoundry ANE Spec — ISA

The H14 byte layer represents configuration writes as task descriptors (TDs).
It operates on words and does not interpret tensor layouts or execute tasks.

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

    def pack(self, **name_val: int) -> dict[int, int]: ...
```

- constraints:
  - `get` MUST return unshifted bits. `put` MUST preserve bits outside the field.
  - A field value MUST fit its bit range and satisfy any declared value restrictions.
  - A field for a floating-point operand takes its integer bit pattern.
  - A block covers aligned word addresses in the half-open byte range `[base, end)`.
  - `Block.pack` MUST combine named fields into words that start at zero.

## 2. Packets

```python
def serialize(image: Mapping[int, int]) -> list[Packet]: ...
```

- constraints:
  - `serialize` MUST reject addresses outside the aligned configuration words.
  - `serialize` MUST keep each packet inside one block and order its values by increasing address.
  - RUN carries up to 64 consecutive words. MASK selects words within 16 words after its first word.
  - At each lowest pending address, compare the RUN and MASK candidates. Choose RUN on a tie.

## 3. Intrinsics

```python
@dataclass(frozen=True)
class Surface:
    """One feature map as the DMA and the L2 buffer address it."""

    channels: int
    height: int
    width: int
    row: int
    channel: int

def surface(channels: int, height: int, width: int) -> Surface: ...
def load(source: Surface) -> dict[int, int]: ...
def store(destination: Surface) -> dict[int, int]: ...
def compute_pe(kind: TaskType, output: Surface, **operands) -> dict[int, int]: ...
def compute_ne(op: NEOp, **operands) -> dict[int, int]: ...
def splice(*parts: Mapping[int, int]) -> dict[int, int]: ...
```

- constraints:
  - A surface MUST pad its rows so the DMA walks 64-byte steps, and MUST space its channels one
    height of rows apart.
  - `load` MUST program the staged source buffer and the source DMA of one feature map.
  - `store` MUST program the staged result buffer and the result DMA of one feature map.
  - `compute_pe` MUST program the planar engine options and declare the task kind.
  - `compute_ne` MUST program the neural engine op and MUST arm the kernel DMA path.
  - A primitive MUST build its words from declared fields only. It MUST NOT shift a value into
    position by hand.
  - A primitive MUST leave the memory base addresses at zero; the stream does not bind memory.
  - `splice` MUST OR words that share an address. Field ranges never overlap, so the result is
    the same as programming each field in turn.

## 4. Default image

```python
DEFAULT_IMAGE: dict[int, int]
```

- constraints:
  - The image MUST hold the value every descriptor starts from, so a word that still holds it is
    not restated.
  - The image MUST NOT hold the node id, which every descriptor writes even at its default.

## 5. Task descriptors

```python
class TD:
    """A task descriptor: a header followed by an ordered packet sequence."""

    def build(
        cls, values: Mapping[BitField | int, int], *, base: Mapping[int, int] | None = None
    ) -> "TD": ...
    def to_bytes(self) -> bytes: ...
    def writes(self) -> dict[int, int]: ...
    def size_words(self) -> int: ...
    def decode(cls, buf: bytes, off: int = 0) -> tuple["TD", int]: ...
```

- constraints:
  - The header carries the task ID and word count. A dependency flag adds a dependency ID word.
  - `build` MUST apply raw words before fields and reject invalid configuration word addresses.
  - For field-only writes, `build` MUST start from the base word or zero if it is absent.
  - `build` MUST omit resulting words equal to words present in `base`.
  - `writes` MUST report literal writes. A later literal write to the same address takes priority.
  - `size_words` MUST count the header and packet words, not alignment padding.
  - `to_bytes` MUST encode little-endian words without stream framing.
  - `decode` MUST reject truncated descriptors and return the next aligned offset.

## 6. Program stream

```python
class Program:
    """One stream of descriptors with a preamble and alignment padding."""

    tds: tuple[TD, ...]

    def to_bytes(self) -> bytes: ...
    def from_bytes(cls, buf: bytes) -> "Program": ...
```

- constraints:
  - The stream MUST start with one preamble. Each descriptor MUST start at a 16-byte boundary.
  - The stream MUST pad each descriptor to the next boundary with zero bytes.
  - `from_bytes` MUST reject an invalid preamble and missing or nonzero alignment bytes.
