# TileFoundry Spec — Schedule

This spec owns the authored HIR operation that selects a concrete TIR
instruction for a tensor tile. `tf.schedule` explicitly names an instruction,
the values it reads, and the repetition of one instruction issue. An
unscheduled HIR call may omit that spelling only when its registry has exactly
one candidate and every required candidate attribute comes from a same-named
HIR attribute. The schedule does not copy the instruction's operand, scope,
capability, or access declarations into a second schema.

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
`ScheduleOp` call arguments; it is not an IR `Tuple` value. `op` is one
registered TIR `Op` instance. `repeat` and `order`, when present, are tuples of
integers.

These four parameters are the complete authored contract. In particular, a
schedule has no separate operand mapping, instruction scope, access map, or
capability field.

`tilefoundry schedule candidates` marks an accepted candidate as `default`
when that instruction would be selected by the omission rule above. The rule
uses the number of registered candidates, not the number accepted at one site:
a `Reshard` with several registered candidates therefore never becomes a
default merely because only one matches its current operands. Zero candidates
remain an unknown HIR call; several candidates, or a required instruction
attribute with no same-named HIR value, require an explicit `tf.schedule`.

## 2. Operand and result roles

The selected instruction's input `ParamDef`s determine the schedule boundary:

- All input parameters consume explicit operands in schema order, including
  `READ`, `READ | WRITE`, and `WRITE` parameters.
- The result has exactly the type and bytes of the first `WRITE` operand.
  This applies equally to a read/write accumulator and a write-only destination.
- A write-only destination is declared with `tf.empty(Tensor[...], buffers=N)`.
  Its complete shape, dtype, layout, and storage come from the program; the
  instruction checks that type against its parameter pattern.
- Optional input parameters are omitted from both sides when not supplied.
  Supplying them preserves their declared effects. No workspace is allocated
  on the author's behalf.
- Operand count MUST match the supplied instruction parameters. A mismatch
  names the instruction, expected parameter names, and actual count.

Automatic selection and explicit `tf.schedule` use the same instruction
lowering and emission path. An instruction declaring an `atom` uses that atom's
participant and fragment geometry. Other instructions issue whole when their
write access maps collapse an axis Split across participants. Otherwise,
single-issue verification uses local fragments. Candidate matching projects
logical Split axes when that projection is possible; if factored splits have
no local logical box, it compares whole types instead.

Each single-issue operand MUST match the corresponding instruction parameter
pattern. `match_instruction(op_type, attributes, operand_types, mesh, *, frame,
repeat=None, order=None)` returns a tuple of frozen `InstructionMatch(op, path)`
records. Each op uses the existing constructor; its path contains every binding
from the closed matcher in binding order. A `ScheduleRefusal` carries the
rejected path and failed-node reason. Type inference only validates the matches;
inspection renders their paths. The
operand types include every input parameter, including the destination, in
schema order. An atom class in `attributes` leaves its parameters open; an
atom instance requests that exact instruction. The execution mesh matches first,
then the shared input scope check verifies each operand holder and storage.
ScheduleOp skips that check in the outer type-inference visitor; every explicit
schedule reaches it through this matcher. Open matching then reads dtype
and storage form, then enumerates tile extents admitted by the parameter
patterns, divisibility, and tile projection.
For each projectable extent it separately matches layout major, solves the
bindings, constructs an instance, and performs the closed instruction checks.
A rejected extent does not stop other extents from matching. If none match,
the refusal comes from the largest projectable extent.
`frame` is required and reaches the verifier unchanged, including its target
capability checks. Explicit schedule inference uses this same matcher with
the authored instruction attributes, frame, repeat, and order.

Operand, tiling, pattern, and verifier rejections raise `ScheduleRefusal`.
Errors from implementation code, including `TypeError` and ISL errors,
propagate to the caller.

A source window is matched in the arrangement in which it lies: its
address offset is not part of the layout, and an unstated layout uses the
parent tensor's compact row-major strides. When an instruction reads a plain
shared-memory arrangement through an all-`Broadcast` `ShardLayoutPattern`, the
single-issue view is bound to one required instruction frame for matching. This
does not assign issues to frames; issue assignment is a lowering decision.

An instruction without a registered access relation cannot be selected by a
schedule.

The target-neutral instruction set reported by `schedule facts` includes
`T.binary`, `T.cast`, `T.clamp`, `T.unary`, `T.copy`, `T.relu`, and `T.reduce`.
Their capability is `all targets`; target-specific instructions are added when
the selected target admits their capabilities.

## 3. Repeat and order

The instruction access relation defines the ordered iteration dimensions. For
matrix multiplication these are `(m, n, k)`; an elementwise transfer uses the
tile axes in order.

`repeat[i]` is the number of single instruction extents that cover iteration
dimension `i`. Its derivation depends on whether the selected instruction
patterns fix a single-issue shape.

If any selected operand pattern declares a shape, that shape is the fixed
single-issue contract. Repeat is inferred as:

```text
whole scheduled extent[i] / single-issue extent[i]
```

Every division MUST be exact. An authored `repeat` MUST equal the inferred
tuple. This is the hardware-fixed path used by tiled MMA instructions.

If no selected operand pattern declares a shape, authored `repeat` is the
source of the single-issue shape and defaults to all ones. Each operand
coordinate projected from iteration dimension `i` has single-issue extent
`whole extent / repeat[i]`; every division MUST again be exact. Tiling on this
open-shape path is not implemented yet, so every repeat count MUST currently be
one and a larger count is rejected as `transfer tiling is not yet supported`.

`order` is a permutation of the iteration dimension positions and defaults to
the identity permutation. `order` controls lowering loop nesting; it does not
change operand or result types. For an atom instruction, lowering emits one
`For(o_<axis>)` per iteration dimension inside each physical issue group, in
`order` from outermost to innermost. It emits the loop even when its trip count
is one. A swizzled row may issue adjacent atoms as straight-line statements in
one loop iteration, but it does not remove that axis's loop. Transfer
instructions emit no such atom loops. Supporting non-identity `order` here is
an intentional extension beyond the AtomSched reference, which rejects it.

Straight-line row issue is a property of the matched operand layout
alternative. An alternative that packs adjacent issues MUST declare their
tensor axis and count; an alternative without that property contributes one
issue. Lowering MUST consume this declaration and MUST NOT infer the count from
layout strides.

For each operand layout, non-unit repeat counts name leading CuTe tile modes;
the remaining inner modes are the instruction fragment matched against its
operand declaration. A sharded operand's mesh layout follows the same rule:
leading modes are issue groups and the inner modes are the participant frame.

## 4. Access relation

Schedule access relations follow the registered construction and input/output
convention defined by
[semantic-analysis §2](./semantic-analysis.md#2-access-relation-analysis).

## 5. Buffers and cost

`tf.empty` has a positive integer `buffers` attribute, defaulting to one.
It declares N physical slots that rotate across iterations of the innermost
surrounding loop. Their declaration uses the existing staging owner outside
that loop. `ScheduleOp` has no `buffers` attribute. Storage and staging belong
to the explicit destination operand.

Schedule traffic is derived from the scheduled access relation. A transfer has
no floating-point work. An MMA reports one multiply and one add per contraction
point, `2 * M * N * K`, in the dtype of its multiplicative inputs.

Ideal MMA time is its `2 * M * N * K` work divided by the target's dense
throughput for the multiplicative dtype. An instruction's `resource`
declaration is not a second service count: adding it to the FLOP time would
price the same tensor work twice.

Movement is priced independently at every crossed memory level for which the
target states a bandwidth, and the longest such time is the memory time. These
levels may overlap, so their times are not summed. A level with no stated rate
contributes no bound; it is neither treated as zero bandwidth nor rejected.

The concrete mapping of issues to participants is a lowering contract rather
than authored ScheduleOp state.

Finalization MUST NOT reject a program because analysis reports capacity
findings. It MUST retain the function-level analysis metadata on the resulting
PrimFunction and print the same report header as `analyze` before the entire
TIR program. Placement and cache findings remain analysis diagnostics; whether
to adopt the result is the user's decision. Successful finalization MUST leave
stdout and stderr empty, and JSON output MUST contain only the `source` field.

## 6. Reference value semantics

Reference evaluation dispatches on the selected TIR instruction through the
schedule-evaluation registry. This registry describes the SSA value produced by
`tf.schedule`; it does not execute the effect-form TIR instruction or write an
explicit destination.

The copy, asynchronous-copy, tensor-map-copy, and matrix-load instructions
produce the source tensor's logical data with the schedule result type. A tiled
MMA produces `acc + torch.matmul(lhs.to(acc.dtype), rhs.to(acc.dtype))`: the
operands widen to the accumulator dtype first, as the instruction does, so no
product or partial sum is rounded to the operand dtype.
`repeat`, `order`, and the destination's `buffers` describe execution and storage rather than a
second logical computation over the whole scheduled operands.

A selected instruction with no schedule-evaluation handler MUST fail reference
evaluation. It MUST NOT fall back to identity based on its memory effects.
