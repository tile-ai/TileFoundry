---
type: BUG
component: ir-hir-tensor-gather
target_repo: tilefoundry
---

# [BUG][ir-hir-tensor-gather] Gather shard-layout derivation silently passes an inconsistent layout through

## Description

### Symptom / Motivation

`ir/hir/tensor/gather.py::_sliced_shard_layout` returns the **input**
`ShardLayout` unchanged (`return sl`) in every case it cannot derive a correct
output layout for: a `Split`-axis gather, a multi-index gather, and a
`strides=None` (sugar) layout. The output `TensorType.layout` then carries
`size(layout.layout.shape) != size(shape)`, silently violating the
`size(G) == size(T.shape)` invariant (shard.md §7.1.1). Measured on a
vocab-TP embedding lookup: `size(shape)=4096` vs `size(G)=529,530,880`.

### Root Cause Analysis

`_sliced_shard_layout` (`src/tilefoundry/ir/hir/tensor/gather.py:63-113`) has
six `return sl` sites, none of which derive a layout consistent with the
gather's output shape:

- `gather.py:73` — `sl.layout` is a `ComposedLayout` (or `sl` isn't a
  `ShardLayout` at all, which is the correct no-op case for an unsharded
  input, but is not distinguished from the composed-layout case here).
- `gather.py:78` — any index shape other than scalar `()` or `(1,)` passes
  through unconditionally, including the masked-gather shapes (`(1, 1)`,
  `(1, 6)`) that should derive `Partial(sum)`.
- `gather.py:82` — `cute_strides is None` (the `strides=None` contiguous
  sugar form) passes through instead of deriving prefix-product strides.
- `gather.py:88` — the gathered axis maps to no cute position.
- `gather.py:91` — a `Split` lies on the gathered axis; this is exactly the
  masked-gather case and should derive `Partial(sum)`, not pass through.
- `gather.py:96` — a scalar index collapses every cute position.

### Related Files

- src/tilefoundry/ir/hir/tensor/gather.py
- docs/spec/hir.md

## Goal

`_sliced_shard_layout` derives a size-consistent output `ShardLayout` (or
raises via `ctx.error`) for every reachable case; no case silently passes the
input layout through onto a differently-shaped output.

## Constraints

- `batch_dims=0` is guaranteed whenever `_sliced_shard_layout` runs against a
  `ShardLayout` operand — `batch_dims>0` over a sharded operand already
  raises earlier in `Gather`'s typeinfer (`gather.py:126-134`); the masked-gather
  rule does not need to re-check `batch_dims`.
- The masked-gather `Partial(sum)` rule applies for **any** index size (no
  numel restriction) — zero-filled masked gathers sum to the true gather
  regardless of how many rows are gathered.
- Downstream protection of a `Partial` output against a non-sum-homomorphic
  consumer is out of scope here; propagation auto-inserts a reshard
  (separate work item), so `Gather` itself only needs to produce the correct
  `Partial(sum)` layout.
- No change to `Gather`'s shape/dtype/eval rules, `batch_dims>0` handling, or
  any other HIR op.

## Milestones

### Milestone M0: Fail-closed shard-layout derivation in `_sliced_shard_layout`

#### Depends
- None

#### Related Files
- src/tilefoundry/ir/hir/tensor/gather.py

#### Plan
- [ ] step 0.1 Replace the composed-layout / malformed-layout `return sl`
  sites (`gather.py:73,84,88`) with `ctx.error` calls that name the axis and
  the reason (composed layout, unmapped axis); keep the unsharded-input
  no-op (`sl` is not a `ShardLayout`) as a plain pass-through.
- [ ] step 0.2 Derive `strides=None` as `prefix_product(cute_shape)`
  (shard.md §3 default) instead of passing through when strides are absent.
- [ ] step 0.3 Reorder the Split check ahead of the scalar/`(1,)` index
  restriction: when the gathered axis carries exactly one `Split` and no
  other `Split` exists anywhere in `attrs`, derive an output `ShardLayout`
  whose `attrs` replace that `Split` with `Partial(sum)` on the same mesh
  axis and whose `layout.shape` substitutes the index shape at the gathered
  cute position(s) (fresh prefix-product strides) — for any index shape.
- [ ] step 0.4 For every other case (multiple `Split`s anywhere, or a
  `Split` elsewhere combined with a non-scalar/`(1,)` index, or a
  non-`Split` axis with a non-scalar/`(1,)` index), call `ctx.error` naming
  the offending `Split` axis (or axis + index shape when no `Split` is
  involved) instead of returning `sl`.
- [ ] step 0.5 Keep the existing scalar/`(1,)`-index pure-slice derivation
  for the remaining case (no `Split` on the gathered axis, scalar or `(1,)`
  index), including the position-remap of any `Split` elsewhere in `attrs`.

#### Acceptance Criteria
- [ ] AC-0-1: `repro_g07.py::test_g07_sugar_slice_derives` green — scalar
  index on a non-`Split` axis of a `strides=None` `Broadcast` layout derives
  a sliced layout with `size(G) == size(shape)` and contiguous strides.
- [ ] AC-0-2: `repro_g07.py::test_g07_vocab_tp_gather_derives_partial` green
  — single-row lookup on a `Split(vocab)` weight derives `Partial(sum)`.
- [ ] AC-0-3: `repro_g07.py::test_g07_multi_row_gather_derives_partial`
  green — the masked-gather rule holds for a multi-row (`idx=(1,6)`) index,
  not just numel-1.
- [ ] AC-0-4: `repro_g07.py::test_g07_control_unsharded_gather_ok` stays
  green — an unsharded gather is unaffected.
- [ ] AC-0-5: `pytest tests/` (existing gather coverage) stays green — no
  regression on the pre-existing scalar/`(1,)`-index slice derivation or
  `batch_dims>0` fail-closed check.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M1: Spec parity for Gather's shard-layout rules

#### Depends
- M0

#### Related Files
- docs/spec/hir.md

#### Plan
- [ ] step 1.1 Update the `Gather` entry in `docs/spec/hir.md` to document
  all three shard-layout rules (contiguous-sugar slice derivation,
  masked-gather `Partial(sum)`, fail-closed for the remaining unanalyzable
  cases), replacing the current constraint bullet that describes only the
  pre-existing scalar/`(1,)`-index slice behavior.

#### Acceptance Criteria
- [ ] AC-1-1: `docs/spec/hir.md` Gather entry's constraints describe all
  three rules and no longer describe the old pass-through behavior.
- [ ] AC-1-2: `scripts/spec_rules_lint.py docs/spec/hir.md` passes.
<!-- policy_ac:start -->
- [ ] Spec section MUST NOT reference plans, milestones, task IDs, commit hashes, PR numbers, agent / human names, or thread / message IDs. <!-- policy_ac: spec_discipline-0 -->
- [ ] Spec section MUST NOT carry a `Non-Goals` / `Future / TODO` / `Out of scope` section. <!-- policy_ac: spec_discipline-1 -->
- [ ] Spec section MUST NOT carry a `Tests` / `Testing` / `Test plan` section or a list of test names. <!-- policy_ac: spec_discipline-2 -->
- [ ] Spec section MUST be in English. <!-- policy_ac: spec_discipline-3 -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

## Execution Preflight

<!-- policy_preflight:start -->

### Policy Rules & Knowledge

- Comment hygiene — Code comments only describe local logic; no plan / milestone / version / review / discussion narration. (see `docs/develop.md § Code comments`) <!-- policy_rules: comment_hygiene -->
- Scope discipline — One commit touches only what the current task requires; unrelated edits / submodule bumps / autoformat go in separate commits or are called out explicitly. (see `docs/develop.md § Scope`) <!-- policy_rules: scope_discipline -->
- Spec discipline — Spec sections follow the spec-writing contract: principle-first, RFC 2119 style, no cross-layer leakage. (see `docs/SPEC-RULES.md § Principle`, `docs/SPEC-RULES.md § Constraints`) <!-- policy_rules: spec_discipline -->
<!-- policy_preflight:end -->
