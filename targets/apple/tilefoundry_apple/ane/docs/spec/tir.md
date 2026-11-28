# TileFoundry ANE Spec — TIR

The ANE TIR Ops are the tasks that one Apple Neural Engine runs. Each Op is one instruction form. The target is specified in [target](./target.md).

## 1. Common constraints

- Every Op is an instruction ([terms](../../../../../../docs/spec/terms.md)) in effect form. It writes its `dst` operand and returns `TupleType.Void`.
- `R` and `W` below mean the READ and WRITE memory effects. Every Op has one `W` operand, `dst`.
- The execution mesh is the one `engine` unit (`ENGINE`). One call is one issue.
- The mesh scope of a call MUST match `ENGINE`.
- Every operand MUST be f16 (`FLOATS`).
- An Op is admitted when `Architecture.capabilities` lists its capability name ([target §1](./target.md#1-anearchitecture)).
  The names are `ne.conv`, `ne.bypass`, `pe.pool`, `pe.unary`, `pe.binary` and `pe.reduce`.
- A feature map is `[1, C, H, W]` in `gmem` or `smem`, with `C <= 65536`, `H <= 16384` and `W <= 16384`.
  Its W stride MUST be 1 when `W` is above 1. The engine addresses a feature map by row, channel, depth and group
  strides and has no element stride. An Op may further constrain the layout.
- `weight` and `bias` are in `gmem`. The kernel height and width of a `weight` are at most 32.
- The public nonlinear modes are `NONE`, `RELU`, `SIGMOID` and `SILU` ([Nonlinear](#nonlinear)).
  Pairing gives SiLU, sigmoid and tanh no `T.ane` candidate.
- Single-operator runs on `apple.m2_pro.ane` checked the value ranges of this page. Its architecture is `apple.h14`.
  No other ANE generation is checked.

### 1.1 Layout constants

| Constant | Value | Meaning |
|---|---|---|
| `ENGINE` | `Mesh(("engine",), Layout(((1,),), ((1,),)))` | The one `engine` unit. |
| `FLOATS` | `(f16,)` | The dtype of every operand. |
| `FEATURE_STORAGE` | `gmem` or `smem` | The storage of a feature map and of an activation matrix. |
| `GMEM_ONLY` | `gmem` | The storage of `weight` and `bias`. |
| `STRIDES` | `((1, 1), (1, 2), (2, 1), (2, 2))` | The admitted `stride` of `conv` and `pool`, as (height, width). |
| `MAX_CHANNELS` | `65536` | The largest channel extent. |
| `MAX_HEIGHT` | `16384` | The largest height extent. |
| `MAX_WIDTH` | `16384` | The largest width extent. |
| `MAX_KERNEL` | `32` | The largest kernel height or width of a `conv` weight. |
| `ACTIVATION_LIMITS` | `(MAX_WIDTH, MAX_CHANNELS)` | The largest `(M, K)` of an activation matrix. |
| `WEIGHT_LIMITS` | `(MAX_CHANNELS, MAX_CHANNELS)` | The largest `(K, N)` of a weight matrix. |

| Form | Shape | Storage |
|---|---|---|
| Feature map | `(1, C, H, W)`; `C` at most `MAX_CHANNELS`, `H` at most `MAX_HEIGHT`, `W` at most `MAX_WIDTH` | `FEATURE_STORAGE` |
| Kernel | `(O, I, KH, KW)`; `O` and `I` at most `MAX_CHANNELS`, `KH` and `KW` at most `MAX_KERNEL` | `GMEM_ONLY` |
| Activation matrix | `(rows, columns)`; the extents are at most `ACTIVATION_LIMITS` | `FEATURE_STORAGE` |
| Weight matrix | `(rows, columns)`; the extents are at most `WEIGHT_LIMITS` | `GMEM_ONLY` |
| Bias | `(O,)`; `O` at most `MAX_CHANNELS` | `GMEM_ONLY` |

### 1.2 Matrix forms

- A channel-outer matrix `[M, K]` has strides `(1, M)`. K is the channel axis and M is the width axis.
- A row-major `[M, K]` has strides `(K, 1)`.
- A mode of extent 1 has any stride.
- `conv` takes a channel-outer `[M, K]` as `src` and a row-major `[K, N]` as `weight`.
  Its `dst` is `[M, N]` with strides `(1, M)` or `(N, 1)`.
- One `bypass` with `perm=(2, 1, 0)` turns a row-major `[M, K]` into a channel-outer `[M, K]`.

### 1.3 Rounding

- `evaluation.f16` rounds a value to the nearest fp16 value. A tie takes the value farther from zero.
- The evaluator forms each step in fp32. `f16(...)` in a formula below marks one rounding.
- A step without `f16(...)` does not round.

## 2. Ops

A HIR pairing offers an Op as a candidate for a HIR call. An explicit `tf.schedule` ([schedule §1](../../../../../../docs/spec/schedule.md#1-authored-form)) selects an Op without pairing.

| DSL name | Class | Capability | Task | HIR pairing |
|---|---|---|---|---|
| `T.ane.conv` | `Conv` | `ne.conv` | neural-engine convolution | `nn.Conv2D`; `nn.MatMul` with `a_layout="MK"` and `b_layout="KN"` |
| `T.ane.bypass` | `Bypass` | `ne.bypass` | neural-engine pass-through | `nn.ReLU`; `tensor.Transpose` |
| `T.ane.pool` | `Pool` | `pe.pool` | processing-engine pooling | `nn.Pool2D` |
| `T.ane.unary` | `Unary` | `pe.unary` | processing-engine element-wise unary | `math.Unary` of kind ABS, SQUARE or NEG |
| `T.ane.binary` | `Binary` | `pe.binary` | processing-engine element-wise binary | `math.Binary` |
| `T.ane.reduce` | `Reduce` | `pe.reduce` | processing-engine reduction | `tensor.Reduce` |

The capability name is the mnemonic of the instruction.

### 2.1 Neural-engine Ops (`ir/tir/ne`)

#### Nonlinear

```python
class Nonlinear(Enum):
    """What a neural-engine task applies after its arithmetic."""

    NONE = "none"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SILU = "silu"
```

- constraints:
  - The DSL name is `T.ane.Nonlinear`.
  - NONE keeps the value. RELU gives `max(x, 0)`. Neither rounds.
  - SIGMOID and SILU look `x` up in the 128-byte table the engine holds for each (`TABLES` in `ir/tir/ne/nonlinear.py`).
    The table has 33 fp16 samples at `x * 2 + 16 == i` for `i` in 0..32. Between two samples the value is
    interpolated linearly and rounded once to fp16. Below and above the samples a line gives
    `f16(f16(slope * x) + offset)`. At or below `x_lo`, and at or above `x_hi`, the table's fixed outputs apply.
  - A table approximates its function: SILU is within `2**-6` of `x * sigmoid(x)` and gives `-0.0017` at 0;
    SIGMOID is within `2**-8` of `sigmoid(x)`.
  - The last bit of a value interpolated between two samples is not decoded. Samples, lines and fixed outputs are.

#### Conv

```python
class Conv(Op):
    """Effect form; dst = nonlinear(conv2d(src, weight) + bias).

    Attributes:
        stride: attribute; (height, width) step of the window.
        padding: attribute; (height, width) zero padding on both sides.
        dilation: attribute; (height, width) spacing of the kernel taps.
        groups: attribute; number of channel groups.
        nonlinear: attribute; the step after the sum.
        dst: input; W; feature map, or matrix.
        src: input; R; feature map, or channel-outer matrix.
        weight: input; R; kernel, or row-major matrix.
        bias: input; R; one value for each output channel, or each column.
    """

    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1
    nonlinear: Nonlinear = Nonlinear.NONE
    dst: Tensor
    src: Tensor
    weight: Tensor
    bias: Tensor | None = None
```

- constraints:
  - The feature form is `dst = nonlinear(f16(conv2d(src, weight) + bias))`. The matrix form is `dst = nonlinear(f16(src @ weight + bias))`.
  - The sum is held in signed 16.16 fixed point: each product is rounded to a multiple of `2**-16`, a tie away from zero, and the rounded products add exactly. A product whose operands' binary exponents sum below -16 contributes nothing; an fp16 subnormal carries the exponent -14. The products add in channel order; the first partial sum outside `[-2**15, 2**15)` makes the result infinite with that sum's sign. A single product of magnitude `2**15` or more MUST be rejected by the evaluator as not decoded yet.
  - The Op rounds once, after it adds `bias`. `nonlinear` applies to the rounded value; a table rounds again.
  - `bias` is optional and part of the instruction. Its shape MUST be `(O,)` in the feature form and `(N,)` in the matrix form.
  - In the feature form, `src` is `[1, C, H, W]`, `weight` is `[O, C / groups, KH, KW]` and `dst` is `[1, O, H', W']`.
  - `H'` is `(H + 2 * padding[0] - KH) // stride[0] + 1`. `W'` is the same rule with the width values.
  - `stride` MUST be in `STRIDES`. Each element is 1 or 2.
  - Each `padding` element MUST be at least 0 and less than the kernel extent of its axis.
  - `dilation` MUST be `(1, 1)`.
  - A dilated convolution and a stride of 3 or more are not one instruction. Pairing refuses them.
  - `groups` MUST be at least 1. It MUST divide `C` and `O`. The input channels of `weight` MUST equal `C / groups`.
  - In the matrix form, `src` is `[M, K]`, `weight` is `[K, N]` and `dst` is `[M, N]`. The strides follow [§1.2](#12-matrix-forms).
  - The matrix form MUST have `stride=(1, 1)`, `padding=(0, 0)` and `groups=1`.
  - `dst`, `src` and `weight` MUST be all feature maps or all matrices.
  - Schedules MUST declare intermediate storage through tensor types. SMEM L2 addresses
    MUST come from the framework's memory analysis and transient staging from codegen,
    with disjoint source and result ranges.
  - Codegen accepts a complete-height feature-map L2 source or result; other SMEM
    requests are not decoded yet and MUST be rejected. The evaluator retains the
    reference semantics for SMEM operands.

#### Bypass

```python
class Bypass(Op):
    """Effect form; dst = nonlinear(src permuted over (channel, height, width) by perm).

    Attributes:
        perm: attribute; the source axis of each (channel, height, width) axis of dst.
        nonlinear: attribute; the step after the permutation.
        dst: input; W; feature map, or channel-outer matrix.
        src: input; R; feature map, or row-major matrix.
    """

    perm: tuple[int, int, int] = (0, 1, 2)
    nonlinear: Nonlinear = Nonlinear.NONE
    dst: Tensor
    src: Tensor
```

- constraints:
  - `perm` MUST be a permutation of `(0, 1, 2)`. The default is the identity.
  - In the feature form, `dst.shape[1 + i]` MUST equal `src.shape[1 + perm[i]]`. The batch axis stays first.
  - `dst` and `src` MUST have the same rank.
  - The Op moves values and does not round. `nonlinear` applies to the moved values.
  - The matrix form relays the values unchanged. It MUST have `perm=(2, 1, 0)`.
  - In the matrix form, `dst` has the shape of `src`. `src` is row-major and `dst` is channel-outer ([§1.2](#12-matrix-forms)).
  - A transpose that moves the batch axis is not one instruction. Pairing refuses it.
  - Pairing offers `bypass` for a `tensor.Transpose` of four axes whose `perm` keeps axis 0 first and is not the identity.
  - The `perm` of the Op is the `perm` of the transpose without axis 0, minus 1 on each element.
  - Pairing offers `bypass` with `nonlinear=RELU` for `nn.ReLU`.

### 2.2 Processing-engine Ops (`ir/tir/pe`)

#### Pool

```python
class Pool(Op):
    """Effect form; dst = max or average pooling of src over kernel_size windows.

    Attributes:
        kernel_size: attribute; (height, width) of the window.
        stride: attribute; (height, width) step of the window.
        padding: attribute; (height, width) padding on both sides.
        ceil_mode: attribute; round the window count up.
        count_include_pad: attribute; count the padding in the average divisor.
        kind: attribute; MAX or AVG.
        dst: input; W; feature map.
        src: input; R; feature map.
    """

    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int] = (0, 0)
    ceil_mode: bool = False
    count_include_pad: bool = True
    kind: PoolKind
    dst: Tensor
    src: Tensor
```

- constraints:
  - `stride` MUST be in `STRIDES`. Each element is 1 or 2.
  - Each `padding` element MUST be at least 0 and less than the `kernel_size` element of its axis.
  - `dst` is `[1, C, H', W']`. `H'` and `W'` are the `Pool2D` window counts ([hir §1.3](../../../../../../docs/spec/hir.md#13-op)), with `ceil_mode` as stated there.
  - MAX takes the largest value of each window. It treats padding as negative infinity and does not round.
  - AVG is `dst = f16(sum * f16(1 / n))`. `sum` is the window sum rounded to fp16, and padding adds 0.
  - `n` is `kernel_size[0] * kernel_size[1]`. With `count_include_pad=False`, `n` is the count of non-padding elements in the window.
  - With `count_include_pad=True`, `n` is the window size also for a `ceil_mode` window past the padded input.
  - `count_include_pad` does not change MAX.
  - Pairing refuses an AVG `Pool2D` with `ceil_mode` and `count_include_pad`. `Pool2D` clips the divisor of that window ([hir §1.3](../../../../../../docs/spec/hir.md#13-op)).

#### Unary

```python
class Unary(Op):
    """Effect form; pre_scale and bias precede either square or scale, then abs and clamp.

    Attributes:
        pre_scale: attribute; the first factor.
        scale: attribute; the factor after the bias; must be 1 when square is true.
        bias: attribute; the addend after pre_scale.
        abs: attribute; apply the absolute value.
        square: attribute; apply the square.
        clamp: attribute; (low, high) bounds, or None.
        dst: input; W; feature map.
        src: input; R; feature map.
    """

    pre_scale: float = 1.0
    scale: float = 1.0
    bias: float = 0.0
    abs: bool = False
    square: bool = False
    clamp: tuple[float, float] | None = None
    dst: Tensor
    src: Tensor
```

- constraints:
  - The Op applies these steps in order:

    ```text
    v = f16(src * pre_scale)
    v = f16(v + bias)
    if square:
        v = f16(v * v)
    else:
        v = f16(v * scale)
        v = abs(v)                 when abs
    v = f16(clamp(v, low, high))    when clamp = (low, high)
    dst = v
    ```

  - `pre_scale` and `bias` always apply. `square` replaces the scale step.
    Codegen and the evaluator MUST reject `square=True` with `scale != 1`:
    hardware square mode ignores scale and cannot implement a non-default factor.
  - `f16` retains the evaluator's existing per-step round-to-nearest with ties away from zero.
    `abs` does not round. Some probes distinguish only operation order: their away/even and
    per-step/final candidates coincide. Those results do not establish hardware rounding
    for every admitted option combination.
  - `abs` with `square` and `clamp` MUST be rejected as not decoded yet, including by the evaluator.
  - Codegen and the evaluator MUST reject option combinations outside the following table
    as not decoded yet.
    An affine attribute not listed in a row MUST have its default value. Within a listed
    set, each attribute may independently have a non-default value.

    | Enabled options | Admitted non-default affine attributes |
    |---|---|
    | none | `pre_scale`, `bias`, `scale` |
    | `clamp` | `pre_scale`, `bias`, `scale` |
    | `abs` | `pre_scale`, `bias`, `scale` |
    | `abs`, `clamp` | `pre_scale`, `bias`, `scale` |
    | `square` | `pre_scale`, `bias` |
    | `abs`, `square` | `pre_scale`, `bias` |
    | `square`, `clamp` | `pre_scale`, `bias` |

  - `clamp` MUST satisfy `low <= high`.
  - `dst` MUST have the shape of `src`.
  - Each operand MUST have a flat layout and static row and channel strides.
  - These strides MUST be multiples of 64 bytes in `gmem` and 16 bytes in `smem`.
  - Pairing sets `abs=True` for ABS, `square=True` for SQUARE and `scale=-1.0` for NEG.
  - Every other `UnaryKind` has no `T.ane.unary` candidate.

#### Binary

```python
class Binary(Op):
    """Effect form; dst = lhs <kind> rhs.

    Attributes:
        kind: attribute; the operation.
        dst: input; W; feature map.
        lhs: input; R; feature map.
        rhs: input; R; feature map.
    """

    kind: BinaryKind
    dst: Tensor
    lhs: Tensor
    rhs: Tensor
```

- constraints:
  - `kind` MUST be ADD, SUB, MUL, MIN or MAX.
  - The Op computes `dst = f16(lhs <kind> rhs)`.
  - `dst`, `lhs` and `rhs` MUST have one shape.
  - Broadcasting and division are not one instruction. Pairing refuses them.

#### Post

```python
class Post(Enum):
    """What the engine applies to a reduction result: identity, 1/x, sqrt(x) or 1/sqrt(x)."""

    NONE = "none"
    INVERSE = "inverse"
    SQRT = "sqrt"
    RSQRT = "rsqrt"
```

- constraints:
  - The DSL name is `T.ane.Post`.
  - NONE gives `x`. INVERSE gives `1 / x`. SQRT gives `sqrt(x)`. RSQRT gives `1 / sqrt(x)`.

#### Reduce

```python
class Reduce(Op):
    """Effect form; dst = post(scale * reduce(square(src))) over axes.

    Attributes:
        axes: attribute; the reduced axes of the feature map.
        keepdim: attribute; keep each reduced axis with extent 1.
        kind: attribute; SUM, MAX or MIN.
        square: attribute; square each element before the reduction.
        scale: attribute; the factor applied to the reduction.
        post: attribute; the step after the scale.
        dst: input; W; feature map, or `[1, C]`.
        src: input; R; feature map.
    """

    axes: tuple[int, ...]
    keepdim: bool
    kind: ReduceKind
    square: bool = False
    scale: float = 1.0
    post: Post = Post.NONE
    dst: Tensor
    src: Tensor
```

- constraints:
  - The Op applies these steps in order:

    ```text
    v = f16(src * src) when square, else src
    r = f16(reduce_kind(v, axes))
    s = f16(scale * r)
    dst = f16(post(s))
    ```

  - `axes` MUST be one of `(2, 3)`, `(3,)`, `(1, 2, 3)` and `(1, 3)`.
  - `keepdim` MUST be true, unless `axes` is `(2, 3)`.
  - With `keepdim`, each reduced axis of `dst` has extent 1. Without it, `dst` is `[1, C]`.
  - `kind` MUST be SUM, MAX or MIN.
  - The last-bit rounding of the `post` step is not verified against hardware. It can differ by one unit in the last place.
  - Rounding `r` before `scale` is not verified against hardware for a sum that fp16 does not hold exactly.
  - A reduction over only the height or only the channel axis is not one instruction. Pairing refuses it.
  - Pairing refuses the kinds MEAN and ABS_MAX.
  - Pairing leaves `square`, `scale` and `post` at their defaults.

## 3. HIR result storage

| `lhs` and `rhs` storage | `MatMul` result storage |
|---|---|
| `smem` in either operand | `smem` |
| `gmem` in either operand, and no `smem` | `gmem` |
| `umat` in both operands | `umat` |

- constraints:
  - The table applies under the ANE target to a HIR `MatMul` ([hir §1.3](../../../../../../docs/spec/hir.md#13-op)).
  - The core rule gives the shape, layout and dtype of the result.

## 4. Refusals

| Violated constraint | Ops | Word in the refusal |
|---|---|---|
| An operand is not f16, or the operands differ in dtype | all | `dtype` |
| An operand shape is outside its form, or the operand shapes disagree | all | `shape` |
| `dst` extents differ from the computed extents | `conv`, `bypass`, `pool`, `reduce` | `dst shape` |
| An operand is in a storage that its form excludes | all | `storage` |
| `stride` is outside `STRIDES`, or is not `(1, 1)` in the matrix form | `conv`, `pool` | `stride` |
| `padding` is negative or not less than the kernel, or is not `(0, 0)` in the matrix form | `conv`, `pool` | `padding` |
| `dilation` is not `(1, 1)` | `conv` | `dilation` |
| `groups` is below 1, does not divide the channels, or is not 1 in the matrix form | `conv` | `groups` |
| The input channels of `weight` differ from `C / groups` | `conv` | `groups` |
| A feature map with `W` above 1 has a W stride other than 1 | all | `strides` |
| A matrix operand has strides that [§1.2](#12-matrix-forms) excludes | `conv`, `bypass` | `layout` |
| `perm` is not a permutation of `(0, 1, 2)`, or is not `(2, 1, 0)` in the matrix form | `bypass` | `perm` |
| `axes` is not one of the four admitted tuples | `reduce` | `axes` |
| `keepdim` is false and `axes` is not `(2, 3)` | `reduce` | `keepdim` |
| `kind` is outside the admitted kinds | `binary`, `reduce` | `kind` |
| `clamp` has `low` above `high` | `unary` | `clamp` |
| The mesh scope of the call does not match `ENGINE` | all | `mesh` |
| The architecture does not list the capability name | all | the capability name |

- constraints:
  - `tf.schedule` raises `VerifyError` with the refusal. A pairing lists the refusal in the candidate report.
  - Pairing refuses a failed HIR condition, an operand dtype or an operand storage with "no applicable HIR declaration". That refusal names no word.
  - Pairing refuses every other row with the word of the row.
