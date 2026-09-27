# TileFoundry Spec — Schedule

This spec owns the authored HIR operation that selects a concrete TIR
instruction for a tensor tile. Scheduling is explicit: `tf.schedule` names the
instruction, the values it reads, and the repetition of one instruction issue.
It does not copy the instruction's operand, scope, capability, or access
declarations into a second schema.

## 1. Authored form

```python
result = tf.schedule(
    (operand0, operand1, ...),
    op=T.some_instruction(...),
    repeat=None,
    order=None,
    buffers=1,
)
```

`operands` is one variadic input. Its tuple syntax is flattened into the
`ScheduleOp` call arguments; it is not an IR `Tuple` value. `op` is one
registered TIR `Op` instance. `repeat` and `order`, when present, are tuples of
integers. `buffers` is a positive integer.

These five parameters are the complete authored contract. In particular, a
schedule has no separate operand mapping, instruction scope, access map, or
capability field.

## 2. Operand and result roles

The selected instruction's input `ParamDef`s determine the schedule boundary:

- Parameters whose `effect` contains `READ` consume schedule operands, in
  declaration order.
- Parameters whose `effect` contains `WRITE` produce schedule results, in
  declaration order.
- A `READ | WRITE` parameter is present on both sides. Its result has exactly
  the type of the value passed for that parameter.
- A write-only result takes fixed shape, dtype, layout, and storage facts from
  its parameter pattern and instruction attributes. Fields left open by the
  destination pattern come from the read tile. A layout not otherwise stated
  is compact.

Each single-issue operand MUST match the corresponding instruction parameter
pattern. A source window is matched in the arrangement in which it lies: its
address offset is not part of the layout, and an unstated layout uses the
parent tensor's compact row-major strides. When an instruction reads a plain
shared-memory arrangement through an all-`Broadcast` `ShardLayoutPattern`, the
single-issue view is bound to one required instruction frame for matching. This
does not assign issues to frames; issue assignment is a lowering decision.

An instruction without a registered access relation cannot be selected by a
schedule.

## 3. Repeat and order

The instruction access relation defines the ordered iteration dimensions. For
matrix multiplication these are `(m, n, k)`; an elementwise transfer uses the
tile axes in order.

`repeat[i]` is the number of single instruction extents that cover iteration
dimension `i`. If omitted, it is inferred as:

```text
whole scheduled extent[i] / single-issue extent[i]
```

Every division MUST be exact. An authored `repeat` MUST equal the inferred
tuple. `order` is a permutation of these dimension positions and defaults to
the identity permutation. `order` controls lowering loop nesting; it does not
change operand or result types.

## 4. Access relation

The schedule relation is the selected instruction's single-issue relation with
one outer tiling band. Each instruction dimension is split into an outer repeat
coordinate and its inner single-issue coordinate; outer coordinates are placed
in `order`. A schedule whose repeat is all ones reaches the same coordinates as
one instruction issue.

The input/output convention for TIR instruction relations is defined in
[semantic-analysis §2](./semantic-analysis.md#2-access-relation-analysis).

## 5. Buffers and cost

`buffers` defaults to one. It multiplies the live result-tile occupancy of a
schedule whose selected instruction has a write-only result. A `READ | WRITE`
accumulator occupies one result tile regardless of `buffers`.

Schedule traffic is derived from the scheduled access relation. A transfer has
no floating-point work. An MMA reports one multiply and one add per contraction
point, `2 * M * N * K`, in the dtype of its multiplicative inputs.

Ideal compute time is the number of issues (`product(repeat)`) multiplied by
the selected instruction's ideal issue time from target facts. MMA issue time
is derived from tensor throughput; copy issue time is constrained by the
narrowest memory level crossed by the instruction.

The concrete mapping of issues to participants, and the lowering loop nest
selected by `order`, are lowering contracts rather than authored ScheduleOp
state.
