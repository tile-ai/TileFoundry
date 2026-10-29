# AMX TIR instruction contracts

AMX instructions are effect-form Ops. An invocation has operand order given
below, returns `Unit`, and writes only its declared output fragments. Attributes
are compile-time `ParamDef` values; operand shape, dtype, storage and layout must
match the declaration's Patterns together. A logical shape alone does not prove
that a tensor occupies the required register fragment.

## 1. Common constraints

- Each AMX instruction operates on one CPU thread's private fragments. A
  `thread` mesh may contain multiple independent invocations; its extent does
  not multiply the physical AMX engine count or join their register files.
- Register operands use `rmem`. Transfer memory operands use `gmem`.
- Layout strides are measured in elements of the operand dtype. Byte offsets
  and register numbers are separate attributes; they are not tensor axes.
- `R`, `W` and `RW` below mean READ, WRITE and READ|WRITE memory effects.
  Accumulators and partial-write destinations are `RW`; unchanged lanes retain
  their incoming values.
- X and Y each contain eight 64-byte registers; Z contains 64 64-byte rows.
  X/Y byte offsets wrap within their 512-byte bank. Register-bank roles are
  part of the instruction contract, even when two operand types are equal.
- A nested layout shape preserves its physical mode decomposition. It must not
  be replaced by an arbitrary contiguous layout with the same element count.
- `bits(d)` denotes the bit width of dtype `d`; `lanes(d) = 512 / bits(d)`.
  All divisions in the formulas below are exact for admitted types.

`ProductMode` is `MATRIX` or `VECTOR`. `Repeat` and `TransferCount` have members
`ONE`, `TWO`, `FOUR`; only the members admitted by a particular Op are legal.
`Shuffle` is `NONE`, `PAIRS`, `QUADS`, or `OCTETS`. Integer signedness is carried
by the dtype, not inferred from the register bank.

### 1.1 Static instruction names

The catalogue has 24 mnemonic families and 39 registered static Op classes.
The suffix is an instruction-contract version, not a chip generation number.
Each version fixes its parameter domain and computation semantics. There is no
public `version`, `generation`, or Target constructor attribute.

| Family | Classes and registered DSL names | First semantic generation |
| --- | --- | --- |
| LDX | `LdxV1` / `T.amx.ldx_v1`; `LdxV2` / `T.amx.ldx_v2`; `LdxV3` / `T.amx.ldx_v3` | 1, 2, 3 |
| LDY | `LdyV1` / `T.amx.ldy_v1`; `LdyV2` / `T.amx.ldy_v2`; `LdyV3` / `T.amx.ldy_v3` | 1, 2, 3 |
| EXTRX | `ExtrxV1` / `T.amx.extrx_v1`; `ExtrxV2` / `T.amx.extrx_v2`; `ExtrxV3` / `T.amx.extrx_v3` | 1, 2, 4 |
| EXTRY | `ExtryV1` / `T.amx.extry_v1`; `ExtryV2` / `T.amx.extry_v2`; `ExtryV3` / `T.amx.extry_v3` | 1, 2, 4 |
| MATFP | `MatfpV1` / `T.amx.matfp_v1`; `MatfpV2` / `T.amx.matfp_v2` | 1, 2 |
| VECFP | `VecfpV1` / `T.amx.vecfp_v1`; `VecfpV2` / `T.amx.vecfp_v2`; `VecfpV3` / `T.amx.vecfp_v3` | 1, 2, 4 |
| MATINT | `MatintV1` / `T.amx.matint_v1`; `MatintV2` / `T.amx.matint_v2` | 1, 3 |
| VECINT | `VecintV1` / `T.amx.vecint_v1`; `VecintV2` / `T.amx.vecint_v2`; `VecintV3` / `T.amx.vecint_v3` | 1, 2, 4 |
| GENLUT | `GenlutV1` / `T.amx.genlut_v1`; `GenlutV2` / `T.amx.genlut_v2` | 1, 2 |

The remaining classes retain unsuffixed names:
`Set`, `Clr`, `Ldz`, `Ldzi`, `Stx`, `Sty`, `Stz`, `Stzi`,
`Fma16`, `Fma32`, `Fma64`, `Fms16`, `Fms32`, `Fms64`, and `Mac16`.
Their DSL names are `T.amx.set`, `T.amx.clr`, `T.amx.ldz`, `T.amx.ldzi`,
`T.amx.stx`, `T.amx.sty`, `T.amx.stz`, `T.amx.stzi`, `T.amx.fma16`,
`T.amx.fma32`, `T.amx.fma64`, `T.amx.fms16`, `T.amx.fms32`,
`T.amx.fms64`, and `T.amx.mac16`.

The compilation Target admits registered capabilities rather than changing an
Op's meaning. `apple.m2_pro` admits the nine versioned names ending in `_v2`
above except MATINT, for which it admits `matint_v1`; it also admits all 15
unsuffixed instructions. Each capability is the DSL leaf name prefixed by
`amx_`, for example `amx_extrx_v2`.

## 2. Lifecycle: Set and Clr

```python
class Set(Op):
    token: Tensor  # RW

class Clr(Op):
    token: Tensor  # RW
```

- constraints:
  - `token` is `u8[1]`, `rmem`, `Layout((1,), (1,))`.
  - No public attributes.
- semantics:
  - `Set` writes an enabled token, value 1.
  - `Clr` writes a disabled token, value 0. Register contents after disabling
    are not defined by the token contract.

## 3. Transfers: Ldx, Ldy, Ldz, Ldzi, Stx, Sty, Stz, Stzi

```python
class Transfer(Op):  # common signature of the classes in this section
    count: TransferCount = TransferCount.ONE
    nonconsecutive: bool = False
    dtype: DType = DType.u8
    register: int = 0
    memory_offset: int = 0
    src: Tensor  # R
    dst: Tensor  # W
```

- constraints:
  - LDX/LDY select X/Y; LDZ and LDZI select Z. STX/STY/STZ/STZI reverse
    the corresponding load direction. Loads are `gmem → rmem`; stores are
    `rmem → gmem`. Source and destination dtypes are equal.
  - Noninterleaved dtypes are `i8`, `u8`, `i16`, `u16`, `i32`, `u32`,
    `f16`, `bf16`, `f32`, or `f64`. LDZI/STZI admit `u32`, `i32`, or `f32`, default `u32`.
  - `memory_offset ∈ [0,511]`. X/Y `register ∈ [0,7]`; Z register numbers
    are in `[0,63]`.
  - `LdxV1`/`LdyV1` admit count 1 or 2. V2 additionally admits 4.
    V3 additionally admits `nonconsecutive=True` for count 2 or 4.
  - LDZ/STX/STY/STZ admit count 1 or 2; LDZI/STZI require count 1.
    All unsuffixed transfers require `nonconsecutive=False`.
  - For noninterleaved transfers, let `c=int(count)`, `n=lanes(dtype)`.
    Operand shape is `(c,n)`. Memory layout is `Layout((c,n),(n,1))`.
    Register layout has the same strides unless nonconsecutive: its row stride
    is `4*n` for count 2 and `2*n` for count 4.
  - LDZI/STZI have logical shape `(1,16)`. Memory layout is
    `Layout((1,16),(16,1))`; Z layout is
    `Layout((1,(2,8)),(0,(16,1)))`. The nested lane mode selects the
    interleaved physical Z words.
- semantics:
  - Copy the selected 64-byte register fragments without numeric conversion.
    Multiple-register addressing and bank wrap follow the selected transfer
    contract; only the destination fragments are written.

## 4. Fma16/32/64 and Fms16/32/64

```python
class FloatingProduct(Op):
    mode: ProductMode = ProductMode.MATRIX
    acc_dtype: DType
    x_dtype: DType
    y_dtype: DType
    out_dtype: DType | None = None
    x_offset: int = 0
    y_offset: int = 0
    z_row: int = 0
    skip_x: bool = False
    skip_y: bool = False
    skip_z: bool = False
    x_mask: int = 0
    y_mask: int = 0
    acc: Tensor  # RW, Z
    lhs: Tensor  # R, Y
    rhs: Tensor  # R, X
```

| Classes | Input dtypes and defaults | Accumulator dtype and default | Lanes |
| --- | --- | --- | --- |
| `Fma16`, `Fms16` | X/Y `f16` | MATRIX `f16` or `f32`, default `f32`; VECTOR `f16` | 32 |
| `Fma32`, `Fms32` | X/Y independently `f16` or `f32`, default `f32` | `f32` | 16 |
| `Fma64`, `Fms64` | X/Y `f64` | `f64` | 8 |

- constraints:
  - `out_dtype`, when supplied, equals `acc_dtype`.
  - `x_offset,y_offset ∈ [0,511]`; `z_row ∈ [0,63]`.
    `x_mask ∈ [0,127]`. MATRIX `y_mask ∈ [0,127]`; VECTOR `y_mask=0`.
  - For lane count `n`, MATRIX shapes are `acc[n,n]`, `lhs[n,1]`,
    `rhs[1,n]`; VECTOR shapes are all `[1,n]`.
  - X/Y input strides are 1, except FMA32/FMS32 `f16` inputs, which use
    stride 2 to select the low half of each 32-bit input slot.
  - Equal-width matrix accumulators use `Layout((n,n),(64,1))`.
    Widening FMA16/FMS16 use
    `Layout((32,(2,16)),(32,(16,1)))`; VECTOR accumulators are contiguous.
- semantics:
  - MATRIX applies an outer product; VECTOR applies a lane-wise product.
    FMA adds the product to Z, FMS subtracts the product from Z, with fused
    floating arithmetic at the declared accumulator precision.
  - The skip controls select the following arithmetic, where `s=+1` for FMA
    and `s=-1` for FMS. Masks select lanes/rows; unselected outputs persist.

| skip_x | skip_y | skip_z | Selected result |
| --- | --- | --- | --- |
| false | false | false | z + s*x*y |
| false | false | true | s*x*y |
| false | true | false | z + s*x |
| false | true | true | s*x |
| true | false | false | z + s*y |
| true | false | true | s*y |
| true | true | false | z |
| true | true | true | signed zero selected by FMA/FMS |

## 5. Mac16

`Mac16` has the FMA control attributes except `out_dtype`, and operand order
`(acc: RW, lhs: R, rhs: R)` with Z/Y/X bank roles.

- constraints:
  - `mode=MATRIX` by default. X/Y dtypes independently admit signed `i8` or
    `i16`, default `i16`. An `i8` input has element stride 2.
  - MATRIX accumulator dtypes are `i16`, `u16`, `i32`, `u32`, default `i32`;
    VECTOR admits `i16` or `u16`.
  - MATRIX shapes are `[32,32]`, `[32,1]`, `[1,32]`; VECTOR shapes are
    `[1,32]`. Equal-width accumulator layout is `(64,1)` in MATRIX mode;
    widened layout is `Layout((32,(2,16)),(32,(16,1)))`.
  - Offset, row, mask and skip domains match §4.
- semantics:
  - Multiply integer inputs and add into the declared accumulator lanes,
    preserving the instruction's finite-width integer result. MATRIX is an
    outer product; VECTOR is lane-wise. Skip controls and masks select the
    corresponding accumulation/additive forms and written lanes.

## 6. MatfpV1/V2 and VecfpV1/V2/V3

```python
class FloatingALU(Op):
    fp_mode: FPMode = FPMode.F32
    dtype: DType = DType.f32
    acc_dtype: DType = DType.f32
    alu: FPALU = FPALU.ADD_PRODUCT
    index_mode: IndexMode = IndexMode.NONE
    index_register: int = 0
    x_offset: int = 0
    y_offset: int = 64
    z_row: int = 0
    mask: int = 0
    y_mask: int = 0
    shuffle_x: Shuffle = Shuffle.NONE
    shuffle_y: Shuffle = Shuffle.NONE
    repeat: Repeat = Repeat.ONE
    broadcast_mode: int = 0
    acc: Tensor  # RW, Z
    lhs: Tensor  # R, Y data or packed indices
    rhs: Tensor  # R, X data or packed indices
    table: Tensor | None = None  # R, indexed bank
```

| `fp_mode` | `dtype` | `acc_dtype` |
| --- | --- | --- |
| F16 | f16 | f16 |
| F16_F32 | f16 | f32 |
| F32 | f32 | f32 |
| F64 | f64 | f64 |
| BF16 | bf16 | bf16 |
| BF16_F32 | bf16 | f32 |

- constraints:
  - V1 excludes BF16 modes. MATFP admits `ADD_PRODUCT`, `SUB_PRODUCT`,
    `POSITIVE_SELECT`. VECFP additionally admits `MIN`, `MAX`; V2/V3 also
    admit `PRODUCT`, `ADD_X`, `ADD_Y`.
  - Indexed addressing requires `alu=ADD_PRODUCT`. `IndexMode` is `NONE`,
    `X2`, `Y2`, `X4`, or `Y4`; the suffix denotes bits per packed index.
    `index_register ∈ [0,7]`, and is zero when indexing is disabled.
  - MATFP requires `repeat=ONE`, `broadcast_mode=0`, `z_row ∈ [0,7]`.
    VECFP has `z_row ∈ [0,63]`; V1 requires repeat 1, V2/V3 admit 1, 2, 4.
    `broadcast_mode ∈ [0,7]` and is zero when repeat is 1.
  - `x_offset,y_offset ∈ [0,511]`; both masks are in `[0,511]`.
  - Let `n=lanes(dtype)`, `q=bits(acc_dtype)/bits(dtype)`.
    MATFP shapes are `acc[n,n]`, `lhs[n,1]`, `rhs[1,n]`. Equal-width Z
    layout is `Layout((n,n),(64,1))`; widened Z layout is
    `Layout((n,(q,n/q)),(n,(lanes(acc_dtype),1)))`.
  - VECFP shapes are `acc[r,n]`, `lhs[r,n]`, `rhs[r,n]`, where
    `r=int(repeat)`. Equal-width Z row stride is
    `(64/r)*lanes(acc_dtype)`; widening splits the lane mode as `(q,n/q)`
    with strides `(lanes(acc_dtype),1)`.
  - Indexed X or Y is a packed `u8` operand with byte count
    `ceil(n*index_bits/8)` per row; the other input retains `dtype`.
    `table` is absent for NONE and contiguous `dtype[n]` otherwise.
  - Vector broadcast modes 2/6 give X row stride zero; modes 3/7 give Y row
    stride zero. Other input row strides equal their physical row byte count
    for packed operands, or lane count for typed operands.
- semantics:
  - With X=`x`, Y=`y`, Z=`z`: ADD_PRODUCT is `z+x*y`, SUB_PRODUCT is
    `z-x*y`, PRODUCT is `x*y`, ADD_X is `z+x`, ADD_Y is `z+y`.
    MIN/MAX compare X with Z. POSITIVE_SELECT returns zero for nonpositive X
    and Y otherwise, according to the selected floating format.
  - MATFP applies the operation to matrix fragments; VECFP applies it to
    vector rows. Shuffles permute X/Y lanes before arithmetic. Masks preserve
    unselected destinations and encode the instruction's zero/broadcast forms.
  - VECFP V3 aligns repeated X/Y origins to register boundaries, with the
    lane/index alignment exceptions selected by broadcast and index mode;
    V2 retains byte-offset origins.

## 7. MatintV1/V2 and VecintV1/V2/V3

These Ops use operand order `(acc: RW, lhs: R, rhs: R, table: optional R)`
with the same Z/Y/X/indexed-bank roles as §6.

```python
class IntegerALU(Op):
    alu: IntALU = IntALU.ADD_PRODUCT
    lane_mode: IntLane = IntLane.NARROW16
    x_dtype: DType = DType.i16
    y_dtype: DType = DType.i16
    acc_dtype: DType = DType.i16
    index_mode: IndexMode = IndexMode.NONE
    index_register: int = 0
    x_offset: int = 0
    y_offset: int = 64
    z_row: int = 0
    shift: int = 0
    mask: int = 0
    mask_y: bool = False
    saturate: bool = False
    rounding: bool = False
    signed_output: bool = False
    shuffle_x: Shuffle = Shuffle.NONE
    shuffle_y: Shuffle = Shuffle.NONE
    repeat: Repeat = Repeat.ONE
    broadcast_mode: int = 0
```

- constraints:
  - Both families admit ADD_PRODUCT, SUB_PRODUCT, ADD_SUM, SUB_SUM, SHIFT,
    SQRDMLAH and SQRDMLSH. MATINT additionally admits WIDE_PRODUCT and POPCOUNT.
    VECINT V2/V3 additionally admit PRODUCT, ADD_X and ADD_Y; VECINT V1
    excludes these three. VECINT does not admit WIDE_PRODUCT or POPCOUNT.
  - MATINT ordinary arithmetic uses NARROW16 or WIDE16 lanes. SHIFT also
    admits I32, I8_I8_I32 and I8_I8_I16. WIDE_PRODUCT admits NARROW16,
    I8_I8_I32, I8_I8_I16 and I8_I16_I32; POPCOUNT admits NARROW16,
    WIDE16 or I32. VECINT ordinary arithmetic admits NARROW16, WIDE16,
    I8_I8_I32, I8_I8_I16, I8_I16_I32 and I16_I8_I32; its SHIFT also
    admits I8. Both families restrict SQRDMLAH/SQRDMLSH to NARROW16.
  - Dtypes are signed or unsigned integers with the following X/Y/Z widths:
    NARROW16 16/16/16, WIDE16 16/16/32, I32 32/32/32,
    I8 8/8/8, I8_I8_I32 8/8/32, I8_I8_I16 8/8/16,
    I8_I16_I32 8/16/32, I16_I8_I32 16/8/32.
    MATINT WIDE_PRODUCT uses 8-bit X/Y except I8_I16_I32: that mode is
    8/8/16 on V1 and 8/16/32 on V2. NARROW16 WIDE_PRODUCT is 8/8/16.
  - SHIFT consumes only the accumulator, not X/Y/table. Its dtype placeholders
    are `u8`; `saturate`, `rounding`, and `signed_output` are available only
    for SHIFT. SHIFT requires both shuffles NONE.
  - Indexing is allowed only for ADD_PRODUCT or WIDE_PRODUCT; packed operand
    counts and table presence follow §6, using each side's actual lane width.
  - `shift ∈ [0,31]`, `mask ∈ [0,511]`, X/Y offsets in `[0,511]`.
    `mask_y` chooses whether the mask applies on the Y dimension.
    MATINT `z_row ∈ [0,3]`, repeat 1 and broadcast 0.
    VECINT `z_row ∈ [0,63]`; V1 repeat 1, V2/V3 repeat 1/2/4,
    broadcast in `[0,7]`, zero for repeat 1.
  - Matrix and vector fragments use the declared nested Z lane decomposition:
    widening ratio `q=max(1,bits(acc_dtype)/bits(input_lane_dtype))` splits
    the lane axis into `(q,lanes/q)` with inner stride 1 and outer stride
    `lanes(acc_dtype)`. For matrix arithmetic let `xb=bytes(x_dtype)`,
    `zb=bytes(acc_dtype)` and `step=zb` when `xb=1`, otherwise `step=xb`:
    accumulator shape is `(64/step,64/xb)` and row stride is `step*64/zb`.
    Vector lane count is `64/min(bytes(x_dtype),bytes(y_dtype))`, row count
    equals repeat, and row stride is `(64/repeat)*64/zb`. SHIFT uses its
    separate accumulator-only layout at the width selected by lane mode.
    Typed inputs are contiguous within a row; packed-index inputs are
    contiguous `u8` bytes.
- semantics:
  - ADD/SUB_PRODUCT accumulate a product; ADD/SUB_SUM accumulate the input
    sum with the selected sign. PRODUCT, ADD_X and ADD_Y use the corresponding
    product/addend without inventing floating arithmetic.
  - SHIFT shifts accumulator lanes with the declared rounding, saturation and
    signed-output controls. SQRDMLAH/SQRDMLSH perform the signed saturating
    rounded doubling multiply-accumulate/subtract operation. WIDE_PRODUCT
    widens products; POPCOUNT uses the integer bit-count operation.
  - VECINT V3 changes repeated X/Y origin alignment in the same manner as
    VECFP V3; earlier versions preserve their byte-offset origins.

## 8. ExtrxV1/V2/V3 and ExtryV1/V2/V3

```python
class Extract(Op):
    mode: ExtractMode = ExtractMode.LEGACY
    src_dtype: DType = DType.f32
    dst_dtype: DType = DType.f32
    destination: Bank  # default X for EXTRX, Y for EXTRY
    z_selector: int = 0
    destination_register: int = 0
    destination_offset: int = 0
    repeat: Repeat = Repeat.ONE
    alternating: bool = False
    low_byte_only: bool = False
    shift: int = 0
    rounding: bool = False
    saturate: bool = False
    mask: int = 0
    src: Tensor  # R, Z
    dst: Tensor  # RW, selected X/Y bank
```

- constraints:
  - COPY admits the ten noninterleaved transfer dtypes in §3; LEGACY admits
    `i16`, `u16`, `f16`, `i32`, `u32`, `f32`, `f64`. Both require
    `dst_dtype=src_dtype` and the family's default destination bank.
  - INTEGER admits signed/unsigned 8/16/32-bit dtypes, with
    `bits(dst_dtype) <= bits(src_dtype)`. FLOAT admits f16→f16,
    f32→f32/f16/bf16, f64→f64; V1 FLOAT requires equal source/destination dtype.
  - INTEGER/FLOAT allow destination X or Y. V1 requires repeat 1; V2/V3
    allow repeat 1/2/4 in these modes. COPY/LEGACY require repeat 1.
  - COPY has `z_selector ∈ [0,7]`, `destination_register ∈ [0,7]`,
    `destination_offset=0`, mask 0. Other modes have `z_selector ∈ [0,63]`,
    `destination_register=0`, `destination_offset ∈ [0,511]`.
  - LEGACY mask is in `[0,127]`; INTEGER/FLOAT mask is in `[0,511]`.
    `alternating=True` requires INTEGER/FLOAT with 32-bit source.
    `low_byte_only=True` requires LEGACY with 16-bit source.
    `shift ∈ [0,31]` and rounding/saturation are available only in INTEGER;
    other modes require shift 0 and both flags false.
  - COPY operands have shape `(1,lanes(src_dtype))` and contiguous row layout.
    Conversion modes have source and destination logical shape
    `(int(repeat),lanes(dst_dtype))`; destination is contiguous by row.
  - Let `q=bits(src_dtype)/bits(dst_dtype)` and `n=lanes(src_dtype)`.
    For narrowing, source layout splits its lane axis into `(n,q)`.
    EXTRX inner strides are `(1,n)`; EXTRY strides are `(64,n)`.
    Alternating selection with 16-bit output doubles the second stride.
    Row stride is `(64/repeat)*n` for EXTRX and `64/repeat/bytes(src_dtype)`
    for EXTRY. Equal-width conversion uses column stride 1 or 64 respectively.
    LEGACY retains its separate horizontal/vertical Z fragment layouts.
- semantics:
  - EXTRX reads horizontal Z fragments; EXTRY reads vertical fragments.
    COPY/LEGACY preserve element representation, with legacy byte selection
    controlled by `low_byte_only`. INTEGER applies narrowing/shift/rounding/
    saturation controls; FLOAT converts to the selected floating format.
  - Masks preserve untouched destination bytes. V3 aligns a repeated
    destination origin down to a 64-byte register boundary; V2 preserves the
    requested byte offset.

## 9. GenlutV1/V2

```python
class GenerateLookup(Op):
    mode: LutMode  # required
    bf16: bool = False
    dtype: DType  # required; must satisfy mode
    source_bank: Bank = Bank.X
    table_bank: Bank = Bank.Y
    destination: Bank = Bank.X
    source_offset: int = 0
    table_register: int = 1
    destination_register: int = 2
    source: Tensor  # R
    table: Tensor   # R
    dst: Tensor     # W
```

- constraints:
  - Source/table banks are X or Y. Generation modes write X or Y; lookup
    modes may additionally write Z. X/Y destination register is in `[0,7]`,
    Z register in `[0,63]`; table register is in `[0,7]`, source offset `[0,511]`.
  - F32/F16/F64/I32/I16/U32/U16 select the corresponding dtype. F16 with
    `bf16=True` requires V2 and dtype bf16. All other modes require bf16 false.
  - LOOKUP32_2/_4 admit i32/u32/f32; LOOKUP16_2/_4/_5 admit
    i16/u16/f16/bf16; LOOKUP8_2/_4/_5 admit i8/u8;
    LOOKUP64_4 requires f64.
  - Table is contiguous `dtype[lanes(dtype)]`.
    Generation source has the same type; destination is contiguous `u8[64]`.
    Lookup source is packed contiguous u8, and destination has the table dtype
    with `lanes(dtype)` elements. Source byte counts are:

| Lookup mode | Source bytes |
| --- | --- |
| LOOKUP32_2 | 4 |
| LOOKUP16_2 | 8 |
| LOOKUP8_2 | 16 |
| LOOKUP64_4 | 4 |
| LOOKUP32_4 | 8 |
| LOOKUP16_4 | 16 |
| LOOKUP8_4 | 32 |
| LOOKUP16_5 | 20 |
| LOOKUP8_5 | 40 |

- semantics:
  - Generation scans table entries until the first value greater than each
    source value and packs the preceding index using the selected index width;
    unused destination bytes are zero. The comparison uses the mode's signed,
    unsigned or floating interpretation.
  - Lookup unpacks 2-, 4-, or 5-bit indices and copies the indexed table values
    to the destination lanes. It changes neither the table nor source bytes.
