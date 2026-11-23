# TileFoundry ANE Spec — Codegen

The ANE builder emits a [descriptor stream](./isa.md) from a lowered TIR function.
The emitter registry is local to ANE and does not emit source text.

## 1. Program builder

```python
def build_program(fn: core.PrimFunction) -> isa.Program: ...
```

- constraints:
  - The builder MUST accept lowered H14 functions that issue one unary task.
  - The builder MUST traverse straight-line statements and engine mesh scopes in source order.
  - Each `Evaluate` MUST dispatch by its Op class through `EMITTERS`.
  - A missing emitter MUST report the target and Op.
  - Register each emitter with `emitter(op_type)` in the file for that Op.
  - An emitter MUST return the [intrinsic](./isa.md#3-intrinsics) sequence of one engine pass and
    MUST NOT name a bit position or a block word.

## 2. Serializer

```python
class State:
    """The configuration image the next descriptor starts from."""

    VOLATILE: frozenset[int]
    image: dict[int, int]

    def emit(self, image: Mapping[int, int]) -> isa.TD: ...


def to_program(parts: Iterable[Mapping[int, int]]) -> isa.Program: ...
```

- constraints:
  - The serializer MUST splice the intrinsic words of one engine pass into one descriptor.
  - An emission MUST start from the [default image](./isa.md#4-default-image) overlaid with the
    sticky words the previous descriptor left behind.
  - An emission MUST write every supplied volatile word, even if its value did not change.
  - An emission MUST omit a sticky word if its value did not change.
  - Only words known to be sticky can leave the volatile set.
  - The serializer MUST reject task sizes that exceed the descriptor header range.

## 3. Unary emitter

- constraints:
  - The emitter MUST use the processing engine for [unary](./tir.md#2-ops).
  - The emitter MUST emit an explicit load, one planar engine pass and an explicit store.
  - The emitter MUST pass the affine and clamp operands to the planar engine primitive as floats.
  - Both feature maps MUST be direct `gmem` parameters. The emitter MUST encode each operand's H and C strides.
  - The emitter MUST reject out-of-range extents and unrepresentable strides instead of truncating them.
  - The emitter MUST leave the memory addresses as placeholders. The stream does not bind memory.

## 4. Image

```python
class IODesc:
    """A named f16 feature map, with shape and strides in elements."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]


class Reloc:
    """A 64-bit placeholder at a word offset from the start of the TD stream."""

    word_offset: int
    symbol: str


class Image:
    """An H14 image with descriptor text, coefficients, I/O, and relocations."""

    text: bytes
    kern: bytes
    io: tuple[IODesc, ...]
    relocs: tuple[Reloc, ...]

    def to_bytes(self) -> bytes: ...
```

- constraints:
  - The image MUST preserve the descriptor stream in `__TEXT,__text`.
  - The image MUST carry coefficients in `__KERN_0`. A unary image has no coefficient bytes.
  - The offline image keeps I/O records in `__FVMLIB,__io`, with input records before output records.
  - Each relocation MUST name an I/O symbol and refer to a zeroed address pair in the descriptor stream.
  - The image MUST reject misaligned text, duplicate I/O names, and unresolved relocation symbols.

## 5. Translation unit

- constraints:
  - The ANE target MUST contribute one C++ translation unit to the shared link project.
  - Each function MUST embed its image and I/O records and expose a C entry point.
  - The C entry point MUST pass its two DLTensor arrays and the embedded records to the runtime declaration.
  - The runtime implementation MUST be supplied separately from the generated translation unit.
