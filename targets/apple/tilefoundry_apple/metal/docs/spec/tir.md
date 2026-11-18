# TileFoundry Metal Spec — TIR

The Metal TIR Ops are the cooperative functions that an Apple GPU provides to a
SIMD group. Each Op is one family of native Metal Shading Language (MSL)
functions. The target is specified in [target](./target.md).

## 1. Common constraints

- Every Op is an instruction ([terms](../../../../../../docs/spec/terms.md)) in effect form. It writes its `dst` operand and returns `TupleType.Void`.
- `R` and `W` below mean the READ and WRITE memory effects. Every Op has one `W` operand, `dst`.
- The execution mesh is G groups of `thread` units. One call covers G groups. G is in `[1, 32]`.
- A **quad group** is a Metal term: four threads that the MSL `quad_` functions combine. It is not a term of [terms](../../../../../../docs/spec/terms.md).
- A matrix Op executes in SIMD groups of 32 lanes. A lane Op executes in SIMD groups of 32 lanes or in quad groups of 4 lanes.
- A group of 32 lanes corresponds to the `simd_` function, and a group of 4 lanes to the `quad_` function.
- An Op is admitted when the target lists any one of its capability names.
  Admission does not check the capability of the selected kind or of the `quad_` form.
- A `dtype` attribute MUST equal the dtype of every data operand. The default is `f32`.
- All lanes of a group are assumed active. Behaviour with inactive lanes is not modelled.

### 1.1 Layout constants

| Constant | Value | Meaning |
|---|---|---|
| `SIMDGROUPS` | `Mesh(("thread",), Layout(((G, 2, 4, 2, 2),), ((32, 1, 2, 8, 16),)))` | G SIMD groups. The lane digits are `lane = d0 + 2*d1 + 8*d2 + 16*d3`. |
| `FRAGMENT` | `ShardLayout(Layout((G, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (S(0), S(4), S(2), S(3), S(1)), SIMDGROUPS)` | G 8 × 8 matrices in registers. Rows are `(d3, d1)`; columns are `(d2, d0, slot)`. |
| `LANE_GROUPS` | `Mesh(("thread",), Layout(((G, W),), ((W, 1),)))`, `W` in `{4, 32}`, `G * W <= 1024` | G SIMD groups or quad groups. |
| `LANES` | `ShardLayout(Layout((G, W), (0, 0)), (S(0), S(1)), LANE_GROUPS)` | Shape `(G, W)`: one scalar per lane. |
| `SHARED` | `ShardLayout(Layout((G, 1), (0, 0)), (S(0), B()), LANE_GROUPS)` | Shape `(G, 1)`: one scalar per group, the same in every lane. |

A fragment operand has shape `(G, 8, 8)`, storage `rmem` and layout `FRAGMENT`.
A memory operand has shape `(G, 8, 8)`, storage `gmem` or `smem`, and layout
`Layout((G, 8, 8), (group_stride, row_stride, 1))` with `row_stride >= 8`.
`row_stride` is the `elements_per_row` argument of the native function.
`group_stride` is not constrained.
A lane operand has storage `rmem` and layout `LANES`. A shared operand has
storage `rmem` and layout `SHARED`.

### 1.2 Matrix lane map

Lane `t` of a SIMD group holds the elements `(row, col)` and `(row, col + 1)` of the 8 × 8 matrix:

```text
row = ((t >> 1) & 3) + ((t >> 4) & 1) * 4
col = (t & 1) * 2 + ((t >> 3) & 1) * 4
```

- constraints:
  - `FRAGMENT` MUST give this lane map. The 64 elements MUST each occur in exactly one lane.
  - All three matrix element types use the same lane map.
  - The Metal Shading Language does not specify this lane map.
    It was measured on one device of Apple GPU family 8, `apple.m2_pro.gpu`. It is not stated for the whole family.

### 1.3 Element types

| Op | `kind` | Element types |
|---|---|---|
| `ldmatrix`, `stmatrix`, `matmul`, `mma` | — | f16, bf16, f32 |
| `reduce` | SUM, PRODUCT, MIN, MAX | f16, f32, i8, u8, i16, u16, i32, u32 |
| `reduce` | AND, OR, XOR | i8, u8, i16, u16, i32, u32 |
| `reduce` | ALL, ANY | bool |
| `prefix`, `shuffle`, `shuffle_fill` | all | f16, f32, i8, u8, i16, u16, i32, u32 |
| `ballot` | — | `src` bool, `dst` u32 |
| `shuffle.sel`, `shuffle_fill.delta` | all | u16 |

## 2. Ops

| DSL name | Class | Capability names | Execution mesh |
|---|---|---|---|
| `T.metal.ldmatrix` | `LdMatrix` | `simdgroup_load` | `SIMDGROUPS` |
| `T.metal.stmatrix` | `StMatrix` | `simdgroup_store` | `SIMDGROUPS` |
| `T.metal.matmul` | `MatMul` | `simdgroup_multiply` | `SIMDGROUPS` |
| `T.metal.mma` | `Mma` | `simdgroup_multiply_accumulate` | `SIMDGROUPS` |
| `T.metal.reduce` | `Reduce` | `simd_sum`, `simd_product`, `simd_min`, `simd_max`, `simd_and`, `simd_or`, `simd_xor`, `simd_all`, `simd_any` | `LANE_GROUPS` |
| `T.metal.prefix` | `Prefix` | `simd_prefix_inclusive_sum`, `simd_prefix_inclusive_product`, `simd_prefix_exclusive_sum`, `simd_prefix_exclusive_product` | `LANE_GROUPS` |
| `T.metal.shuffle` | `Shuffle` | `simd_shuffle`, `simd_broadcast`, `simd_shuffle_xor`, `simd_shuffle_up`, `simd_shuffle_down`, `simd_shuffle_rotate_up`, `simd_shuffle_rotate_down` | `LANE_GROUPS` |
| `T.metal.shuffle_fill` | `ShuffleFill` | `simd_shuffle_and_fill_up`, `simd_shuffle_and_fill_down` | `LANE_GROUPS` |
| `T.metal.ballot` | `Ballot` | `simd_ballot` | `LANE_GROUPS` |

### 2.1 Memory Ops (`ir/tir/memory`)

#### LdMatrix

```python
class LdMatrix(Op):
    """Effect form; load one 8 x 8 matrix per SIMD group from memory.

    Attributes:
        transpose: attribute; load the transpose of each matrix.
        dtype: attribute; element type.
        dst: input; W; fragment operand.
        src: input; R; memory operand.
    """

    transpose: bool = False
    dtype: DType = DType.f32
    dst: Tensor
    src: Tensor
```

- constraints:
  - `dst` MUST equal `src`, or its transpose when `transpose` is true.

#### StMatrix

```python
class StMatrix(Op):
    """Effect form; store one 8 x 8 matrix per SIMD group to memory.

    Attributes:
        transpose: attribute; store the transpose of each matrix.
        dtype: attribute; element type.
        dst: input; W; memory operand.
        src: input; R; fragment operand.
    """

    transpose: bool = False
    dtype: DType = DType.f32
    dst: Tensor
    src: Tensor
```

- constraints:
  - `dst` MUST equal `src`, or its transpose when `transpose` is true.
  - The windows of the G groups in `dst` can overlap. The Op does not reject an overlap.

### 2.2 NN Ops (`ir/tir/nn`)

#### MatMul

```python
class MatMul(Op):
    """Effect form; dst = lhs @ rhs for each SIMD group.

    Attributes:
        dtype: attribute; element type.
        dst: input; W; fragment operand.
        lhs: input; R; fragment operand.
        rhs: input; R; fragment operand.
    """

    dtype: DType = DType.f32
    dst: Tensor
    lhs: Tensor
    rhs: Tensor
```

- constraints:
  - `dst` MUST equal `lhs @ rhs` for each group.

#### Mma

```python
class Mma(Op):
    """Effect form; dst = lhs @ rhs + acc for each SIMD group.

    Attributes:
        dtype: attribute; element type.
        dst: input; W; fragment operand.
        lhs: input; R; fragment operand.
        rhs: input; R; fragment operand.
        acc: input; R; fragment operand.
    """

    dtype: DType = DType.f32
    dst: Tensor
    lhs: Tensor
    rhs: Tensor
    acc: Tensor
```

- constraints:
  - `acc` MAY be the same tensor as `dst`.

### 2.3 Lane Ops (`ir/tir/arith`)

#### ReduceKind / PrefixKind / ShuffleKind

```python
class ReduceKind(Enum):
    """How Reduce combines the lanes of a group."""

    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    AND = "and"
    OR = "or"
    XOR = "xor"
    ALL = "all"
    ANY = "any"


class PrefixKind(Enum):
    """The running combination Prefix computes."""

    SUM = "sum"
    PRODUCT = "product"


class ShuffleKind(Enum):
    """Which lane each lane reads in Shuffle and ShuffleFill."""

    INDEX = "index"
    BROADCAST = "broadcast"
    XOR = "xor"
    UP = "up"
    DOWN = "down"
    ROTATE_UP = "rotate_up"
    ROTATE_DOWN = "rotate_down"
```

- constraints:
  - The DSL names are `T.metal.ReduceKind`, `T.metal.PrefixKind` and `T.metal.ShuffleKind`.
  - A Reduce member corresponds to `simd_<value>`. A Prefix member corresponds to `simd_prefix_<inclusive|exclusive>_<value>`.
  - For Shuffle, INDEX corresponds to `simd_shuffle` and BROADCAST to `simd_broadcast`. Every other member corresponds to `simd_shuffle_<value>`.
  - For ShuffleFill, UP and DOWN correspond to `simd_shuffle_and_fill_<value>`.

#### Reduce

```python
class Reduce(Op):
    """Effect form; combine the lanes of each group and give every lane the result.

    Attributes:
        kind: attribute; the combination.
        dtype: attribute; element type.
        dst: input; W; shared operand.
        src: input; R; lane operand, or a partial operand.
    """

    kind: ReduceKind = ReduceKind.SUM
    dtype: DType = DType.f32
    dst: Tensor
    src: Tensor
```

- constraints:
  - `src` MUST be a lane operand, or, for SUM, MIN and MAX, a partial operand.
  - A partial operand has shape `(G, 1)`, storage `rmem` and the shard attrs `(S(0), Partial(kind))` on `LANE_GROUPS`.
    Each lane holds one partial value of its group.
  - `dst` MUST hold the combination of all lanes of each group.

#### Prefix

```python
class Prefix(Op):
    """Effect form; running sum or product over the lanes of each group.

    Attributes:
        kind: attribute; the combination.
        inclusive: attribute; include the lane itself.
        dtype: attribute; element type.
        dst: input; W; lane operand.
        src: input; R; lane operand.
    """

    kind: PrefixKind = PrefixKind.SUM
    inclusive: bool = False
    dtype: DType = DType.f32
    dst: Tensor
    src: Tensor
```

- constraints:
  - When `inclusive` is true, lane `l` MUST receive the combination of lanes `0..l`.
  - When `inclusive` is false, lane `l` MUST receive the combination of lanes `0..l-1`.
  - In the exclusive form, lane 0 receives 0 for SUM and 1 for PRODUCT.

#### Shuffle

```python
class Shuffle(Op):
    """Effect form; each lane reads the value of another lane of its group.

    Attributes:
        kind: attribute; how each lane selects its source lane.
        dtype: attribute; element type.
        dst: input; W; lane operand.
        src: input; R; lane operand.
        sel: input; R; the selector.
    """

    kind: ShuffleKind = ShuffleKind.INDEX
    dtype: DType = DType.f32
    dst: Tensor
    src: Tensor
    sel: Tensor
```

- constraints:
  - For INDEX, `sel` MUST be a lane operand. Every other kind MUST have one scalar `sel` for the whole call.
  - Lane `l` reads lane `s` of `src`: INDEX `sel[l]`, BROADCAST `sel`, XOR `l ^ sel`, UP `l - sel`, DOWN `l + sel`.
  - ROTATE_UP reads `(l - sel) mod W`. ROTATE_DOWN reads `(l + sel) mod W`.
  - For UP and DOWN, a lane whose source is outside the group keeps its own value.
  - For INDEX, BROADCAST and XOR, MSL does not define the result for a source outside the group. The evaluator raises `EvalError`.

#### ShuffleFill

```python
class ShuffleFill(Op):
    """Effect form; shift each sub-vector of modulo lanes and take the vacated lanes from fill.

    Attributes:
        kind: attribute; UP or DOWN.
        modulo: attribute; lanes in one sub-vector.
        dtype: attribute; element type.
        dst: input; W; lane operand.
        src: input; R; lane operand.
        fill: input; R; lane operand.
        delta: input; R; scalar shift.
    """

    kind: ShuffleKind = ShuffleKind.UP
    modulo: int = 32
    dtype: DType = DType.f32
    dst: Tensor
    src: Tensor
    fill: Tensor
    delta: Tensor
```

- constraints:
  - `kind` MUST be UP or DOWN. `modulo` MUST be in `{2, 4, 8, 16, 32}` and MUST NOT exceed `W`. Verification checks both.
  - Let `p` be the position of a lane in its sub-vector, and `q = p - delta` for UP or `p + delta` for DOWN.
    The lane reads `src` at `q` when `0 <= q < modulo`, and `fill` at `q mod modulo` otherwise, both in its sub-vector.
  - MSL does not define the result for `delta > modulo`. The evaluator raises `EvalError`.

#### Ballot

```python
class Ballot(Op):
    """Effect form; one bit per lane of a bool, as a u32 that every lane of the group reads.

    Attributes:
        dst: input; W; shared operand.
        src: input; R; lane operand.
    """

    dst: Tensor
    src: Tensor
```

- constraints:
  - Bit `l` of `dst` MUST be `src[l]`. Only the low 32 bits of the native vote are kept; a quad group sets the low 4 bits.

## 3. Evaluation

- The evaluator is torch. A matrix product is `a @ b`, and a product with accumulation is `a @ b + c`.
- The floating-point rounding order is not modelled. This applies to `matmul`, `mma`, and floating-point `reduce` and `prefix` with SUM or PRODUCT.
  The torch order of a reduction can differ from hardware. The evaluator agrees with hardware only on exactly representable values.
- The evaluator computes u16 and u32 lane data in i64. It writes a result back to the operand dtype with wrap-around.

## 4. HIR matches

| HIR Op | Condition | Instruction |
|---|---|---|
| `nn.MatMul` | `a_layout = "MK"`, `b_layout = "KN"`, fragment operands | `T.metal.matmul` |
| `sharding.Reshard` | memory operand to fragment operand | `T.metal.ldmatrix`, `transpose=False` |
| `sharding.Reshard` | fragment operand to memory operand | `T.metal.stmatrix`, `transpose=False` |
| `sharding.Reshard` | partial operand with `Partial(sum)`, `Partial(min)` or `Partial(max)` to shared operand | `T.metal.reduce` of that kind |

- constraints:
  - `T.metal.mma` has no HIR match. A program selects it with `tf.schedule` ([schedule §1](../../../../../../docs/spec/schedule.md#1-authored-form)) or writes it in authored TIR.
  - The other Ops have no HIR match.
  - Take a HIR `MatMul` whose `lhs` and `rhs` are fragment operands of one mesh in MK × KN order.
    Its result MUST be a fragment operand with the layout of `lhs` and storage `rmem`.
    The result dtype follows the core `MatMul` rule ([hir §1.3](../../../../../../docs/spec/hir.md#13-op)).
    The product is one cooperative instruction, and all three operands share one lane map.
  - For other operands, the core shape and layout apply. The result is in `rmem` when an operand is in `smem` or `rmem`.
  - The candidate report lists an Op once for each of its capability names. This repetition has no effect on selection.

## 5. Lowering

`metal.Target` binds its own `ConvertHIRToTIR` ([passes §7.1](../../../../../../docs/spec/passes.md#71-converthirtotir)).

- constraints:
  - Lowering issues each `T.metal.*` instruction once, in the mesh that owns its operands. One issue covers all G groups.
  - A `T.metal.*` call outside every mesh `with` MUST raise `LoweringError` with "instruction issue has no lexical mesh".
  - Lowering MUST leave every other Op to the core lowering.
  - Lowering does not tile. An operand that one instruction does not cover, such as a 16 × 16 matrix, is refused.

## 6. Barriers

A barrier is the core `T.sync(m)` ([tir §1.5](../../../../../../docs/spec/tir.md#15-sync)).

The table is a correspondence. No code emits these functions.

| `T.sync` mesh | MSL function |
|---|---|
| The whole `thread` mesh | `threadgroup_barrier` |
| A slice of exactly one SIMD group of 32 threads | `simdgroup_barrier` |

- constraints:
  - `T.sync` can only occur in an authored TIR `prim_func`. HIR has no barrier.
  - Lowering from HIR inserts no barrier. Metal has no sync analysis.
  - A HIR program that stores to threadgroup memory and then reads it back is therefore not synchronized. This is a known limit.
  - Metal has no barrier for any other unit set. Examples are a lane subset under 32 and a range of several groups that is not the whole mesh.
  - The core rule requires the target verifier to reject such a unit set ([tir §1.5](../../../../../../docs/spec/tir.md#15-sync)).
    Metal does not meet this rule. It accepts these unit sets and does not reject them.
