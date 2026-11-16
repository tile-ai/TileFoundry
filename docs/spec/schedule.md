# TileFoundry Spec — Schedule

This spec owns the authored HIR operation that selects a concrete TIR
instruction for a tensor tile. `tf.schedule` explicitly names an instruction,
the values it reads, and the repetition of one instruction issue. An
unscheduled HIR call may omit that spelling only when two conditions hold. Its
registry has exactly one candidate. Every required candidate attribute comes
from a same-named HIR attribute.

## 1. Authored form

```python
result = tf.schedule(
    (operand0, operand1, ...),
    op=T.some_instruction(...),
    repeat=None,
    order=None,
)
```

`operands` is one variadic input. Its tuple syntax is flattened into the
`ScheduleOp` call arguments. It is not an IR `Tuple` value. `op` is one
registered TIR `Op` instance. `repeat`, when present, is a tuple of positive
integers. `order`, when present, is a tuple of integers. Construction rejects
anything else.

These four parameters are the complete authored contract. The schedule does not
copy the operand, scope, capability, or access declarations of the instruction
into a second schema.

`tilefoundry schedule candidates` marks an accepted candidate as `default`
when the omission rule above selects that instruction. The rule uses the number
of registered candidates, not the number accepted at one site. Therefore a
`Reshard` with several registered candidates never becomes a default merely
because only one matches its current operands. Zero candidates remain an unknown
HIR call. Several candidates require an explicit `tf.schedule`. A required
instruction attribute with no same-named HIR value also requires one.

## 2. Operand and result roles

The input `ParamDef`s of the selected instruction determine the schedule
boundary:

- All input parameters consume explicit operands in schema order, including
  `READ`, `READ | WRITE`, and `WRITE` parameters.
- The result has exactly the type and bytes of the first `WRITE` operand.
  This applies equally to a read/write accumulator and a write-only destination.
- A write-only destination is declared with `tf.empty(Tensor[...], buffers=N)`.
  Its complete shape, dtype, layout, and storage come from the program. The
  instruction checks that type against its parameter pattern.
- Optional input parameters are omitted from both sides when not supplied.
  Supplying them preserves their declared effects. The schedule allocates no
  workspace on behalf of the author.
- Operand count MUST match the supplied instruction parameters. A mismatch
  names the instruction, the expected parameter names, and the actual count.

Automatic selection and explicit `tf.schedule` use the same instruction
lowering and emission path. An instruction declaring an `atom` uses the
participant and fragment geometry of that atom. Other instructions issue whole
when their write access maps collapse an axis Split across participants.
Otherwise, single-issue verification uses local fragments. Candidate matching
projects logical Split axes when that projection is possible. If factored splits
have no local logical box, it compares whole types instead.

Each single-issue operand MUST match the corresponding instruction parameter
pattern. `hir.schedule.match_instruction(op_type, attributes, operand_types, mesh, *, ctx,
repeat=None, order=None)` returns a tuple of frozen `hir.schedule.InstructionMatch(op, path)`
records. Each op uses the existing constructor. Its path contains every binding
from the closed matcher in binding order. A `schedule.RefusalError` carries the
rejected path and the failed-node reason. Type inference only validates the
matches. Inspection renders their paths. The operand types include every input
parameter in schema order, including the destination. An atom class in
`attributes` leaves its parameters open. An atom instance requests that exact
instruction. The execution mesh matches first. Then the shared input scope check
verifies each operand holder and storage. ScheduleOp skips that check in the
outer type-inference visitor. Every explicit schedule reaches it through this
matcher.

Open matching has these steps:

1. It enumerates the non-integer atom attributes. It admits their attribute
   patterns, execution mesh and operand fields, including dtype and storage.
   It does not require shape or layout matches.
2. It enumerates the integer parameters that their patterns admit.
3. If a parameter names a symbol in declared operand shapes, its values MUST
   divide every corresponding concrete operand axis extent. A parameter absent
   from those shapes retains the full range from one to the largest operand
   extent.
4. It tries integer tuples largest first.
5. Each concrete instruction then passes the shared closed check, including
   tile projection and layout matching.

A rejected extent does not stop other extents from matching. If all non-integer
bindings fail field admission, the refusal names that field. It does not name an
unrelated integer extent. Otherwise, if no extent is accepted, the first failed
closed check supplies the refusal. `ctx` is required. It reaches the verifier
unchanged, including its target capability checks. Explicit schedule inference
uses this same matcher with the authored instruction attributes, context, repeat,
and order.

Operand, tiling, pattern, and verifier rejections raise `schedule.RefusalError`.
Errors from implementation code propagate to the caller. This includes
`TypeError` and ISL errors.

Matching reads a source window in the layout in which it lies. The address
offset is not part of the layout. An unstated layout uses the compact row-major
strides of the parent tensor. When an instruction reads a plain shared-memory
layout through an all-`Broadcast` `pattern.ShardLayout`, matching binds the
single-issue view to one required execution mesh.

An instruction without a registered access relation cannot be selected by a
schedule.

`schedule facts` reports the target-neutral instruction set. It includes
`T.arange`, `T.binary`, `T.cast`, `T.clamp`, `T.unary`, `T.where`, `T.copy`, `T.relu`,
and `T.reduce`. These instructions state no target capability. Target-specific
instructions are added when the selected target admits their capabilities.

## 3. Repeat and order

The access relation of the instruction defines the ordered iteration
dimensions. For matrix multiplication these are `(m, n, k)`. An elementwise
transfer uses the tile axes in order.

`repeat[i]` is the number of single instruction extents that cover iteration
dimension `i`. Its derivation depends on whether the selected instruction
patterns fix a single-issue shape.

If any selected operand pattern declares a shape, that shape is the fixed
single-issue contract. Repeat is inferred as:

```text
whole scheduled extent[i] / single-issue extent[i]
```

Every division MUST be exact. An authored `repeat` MUST equal the inferred
tuple. Tiled MMA instructions use this hardware-fixed path.

If no selected operand pattern declares a shape, the authored `repeat` is the
source of the single-issue shape. It defaults to all ones. Each operand
coordinate projected from iteration dimension `i` has single-issue extent
`whole extent / repeat[i]`. Every division MUST again be exact. Tiling on this
open-shape path is not implemented. Every repeat count MUST be one. A larger count
is rejected as `transfer tiling is not yet supported`.

`order` is a permutation of the iteration dimension positions. It defaults to
the identity permutation. It does not change operand or result types. For an
atom instruction, lowering issues the serial tiles in `order`, from outermost
to innermost, with static integer operand starts. A per-operand CuTe
`zipped_divide` yields a tile and a rest. A rest mode Split outside the
execution mesh of the instruction binds that iteration axis for every operand.
Other rest modes are serial. Lowering emits no
`For(o_<axis>)` loops. It does not group issues by swizzle row. Existing view
inference determines the swizzle phase at each static start. Transfer
instructions issue once. Non-identity `order` is supported.

ScheduleOp operands follow the same Visit and memo path as other operations.
CuTe `slice_and_offset` fixes the bound modes once per unit, before issuing.
Shared storage uses MeshCoord. Private storage uses coordinate zero and keeps
the authored strides. The retained ShardLayout uses the execution mesh.
Each serial issue slices this local value at its logical rest coordinate.
When that Slice has the required type, the issue uses it directly. Otherwise, a
TensorView over PtrOf(Slice) replaces only the execution mesh and attrs. The
shape, strides, swizzle and offset of the Slice remain unchanged. Storage
forwarding does not introduce views. Explicit HIR view operations remain
represented, including identity operations.

For each operand layout, non-unit repeat counts name leading CuTe tile modes.
The remaining inner modes are the instruction fragment. It is matched against
the operand declaration. The mesh layout of a sharded operand follows the same
rule. Its leading modes are issue groups. Its inner modes are the execution
mesh.

## 4. Access relation

Schedule access relations follow the registered construction and input/output
convention defined by
[semantic-analysis §2](./semantic-analysis.md#2-access-relation-analysis).

## 5. Buffers and cost

`tf.empty` has a positive integer `buffers` attribute. The default is one.
It declares N physical slots. The slots rotate across the iterations of the
innermost surrounding loop. Their declaration uses the existing staging owner
outside that loop. `ScheduleOp` has no `buffers` attribute. Storage and staging
belong to the explicit destination operand.

Schedule traffic comes from the scheduled access relation. A transfer has no
floating-point work. An MMA reports one multiply and one add per contraction
point, `2 * M * N * K`, in the dtype of its multiplicative inputs.

Ideal MMA time is its `2 * M * N * K` work divided by the dense throughput of
the target for the multiplicative dtype. The `resource` declaration of an
instruction is not a second service count. Adding it to the FLOP time prices the
same tensor work twice.

Movement is priced independently at every crossed memory level for which the
target states a bandwidth. The longest such time is the memory time. These
levels may overlap, so their times are not summed. A level with no stated rate
contributes no bound. It is not treated as zero bandwidth. It is not rejected.

Finalization MUST NOT reject a program because analysis reports capacity
findings. It MUST retain the function-level analysis metadata on the resulting
PrimFunction. It MUST print the same report header as `analyze` before the
entire TIR program. Address assignment and cache findings remain analysis
diagnostics. The user decides whether to adopt the result. Successful
finalization MUST leave stdout and stderr empty. JSON output MUST contain only
the `source` field.

## 6. Reference value semantics

Reference evaluation dispatches on the selected TIR instruction through
`evaluator.registry`. The handler of `ScheduleOp` looks up the class of the
instruction in it. That handler describes the SSA value that `tf.schedule`
produces. It does not execute the effect-form TIR instruction. It does not write
an explicit destination.

The copy, asynchronous-copy, tensor-map-copy, and matrix-load instructions
produce the logical data of the source tensor with the schedule result type.
A tiled MMA produces `acc + torch.matmul(lhs.to(acc.dtype), rhs.to(acc.dtype))`.
The operands widen to the accumulator dtype first, as the instruction does.
Therefore no product or partial sum is rounded to the operand dtype.
`repeat`, `order`, and the `buffers` of the destination describe execution and
storage. They are not a second logical computation over the whole scheduled
operands.

Reference evaluation MUST fail for a selected instruction that has no
schedule-evaluation handler. It MUST NOT fall back to identity based on the
memory effects of the instruction.
