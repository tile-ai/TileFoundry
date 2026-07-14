---
type: FEAT
component: reshape
target_repo: tilefoundry
---

# [FEAT][reshape] Divisibility-based Split refactor through Reshape

## Description

### Symptom / Motivation

Gap register G-8 (`research/deepseek-v4-flash-dataflow/11-tilefoundry-gaps-decode-module.md`
§G-8, TileOpsGov). `Reshape` rejects every new shape over a `Split`-carrying
`ShardLayout` — `VerifyError: "Reshape cannot express the sharded layout: new
shape does not align with the input cute factorization"` — even when the
mesh extent divides the regrouped factor cleanly. This refutes doc-07 §3 step
13's assumption that the head→o-group regroup "composes cleanly"
(`[1,1,64,512] Split(heads)@dev4 → [1,1,8,4096]`, 16 heads/GPU = 2 whole
o-groups) and blocks any TP line whose weights pass through the in-graph
dequant reshape (`[8192,1024] Split(0)@dev4 → (64,128,8,128)`).
`gap-repros/repro_g08.py` (TileOpsGov) reproduces both as FAIL, with a
straddling case (`[6,4] Split(0)@dev2 → [3,8]`) as a green regression guard.

### Root Cause Analysis

- `_carry_sharded_reshape` (`src/tilefoundry/ir/hir/tensor/reshape.py:27-104`)
  walks `new_shape` and greedily consumes whole cute positions from the input
  `ShardLayout.layout.shape` to fill each new axis. A cute position that would
  overshoot the current new axis (`prod * cs > d`, `reshape.py:79`) always
  returns `None` (fail closed) — the algorithm has no notion of *splitting* a
  single cute position across a new-axis boundary. This is correct for a pure
  view (merge of whole positions, unit-axis insertion/removal) but rejects
  every case where a `Split`-bound axis itself must be subdivided to express
  the new shape, which is exactly the doc-07 o-group regroup and the dequant
  refactor.
- Verified by direct calculation (row-major flatten, `IntTuple` corder
  strides per `docs/spec/shard.md` §1/§3): for a `Split`-bound cute position
  of extent `N` on a mesh axis of extent `P`, subdividing `N` into an outer
  factor `M` (destined for the earlier new axis) and inner residual `m =
  N/M` preserves the per-device block-partition (`local_shape = N/P`,
  `docs/spec/shard.md` §7.1.1) **iff** `P` divides `M`; `Split` then
  relocates to the new cute position holding `M` (local extent `M/P`), and
  `m` carries forward as a plain (non-`Split`) cute position. This is the
  factorization algebra `docs/spec/shard.md` §7.1.1 already sketches for
  parse-time sugar (`N @ m.a → (mesh_extent @ m.a, N // mesh_extent)`),
  generalized to an arbitrary dividing `M` (not just `M == mesh_extent`).

### Related Files
- `src/tilefoundry/ir/hir/tensor/reshape.py`
- `tests/ops/test_reshape.py`
- `docs/spec/hir.md`
- `research/deepseek-v4-flash-dataflow/gap-repros/repro_g08.py` (TileOpsGov, read-only acceptance input)

## Goal

Extend `Reshape`'s `@register_typeinfer` rule to derive a refactored
`ShardLayout` when every `Split`-bound cute position that must subdivide
across a new-axis boundary does so at a mesh-extent-dividing point, and keep
failing closed when it does not.

## Constraints

- Priority SUBSET (user decision O9.1, `plans/2026-07-15-wi09-reshape-split-refactor.md`
  TileOpsGov): implement adjacent-axis merge/split where each `Split`-bound
  cute position's outer sub-factor is divisible by its bound mesh extent.
  Arbitrary rank-N regroup (an old `Split` axis whose device-owned block
  spans a boundary deeper than one split point, or multiple `Split` axes
  interacting across the same regroup) is the TARGET contract — documented
  in the spec, not implemented; it MUST fail closed (`_carry_sharded_reshape`
  already returns `None` → `VerifyError` for anything the subset does not
  recognize, so "not implemented" and "fails closed" are the same code path).
- The fix lives in `Reshape`'s `@register_typeinfer` rule (user decision
  O9.2), not a separate pass — this rule is shared by parse-time and any
  later elaboration-time re-typeinfer (WI-7 plan-B instantiation), so a
  pass would run at the wrong points or duplicate the logic.
- `repro_g08.py` is read-only acceptance input — the fix MUST make all 3
  tests pass without editing that file.
- **Discovered conflict (flagged, not silently resolved):** two existing
  `tests/ops/test_reshape.py` cases — `straddle_fails_closed` (`(16, 8)
  Split(0) mesh(4,) → (4, 32)`) and `split_dim_straddles_fails_closed`
  (`(4096,) Split(0) mesh(4,) → (32, 128)`) — currently assert `VerifyError`.
  Independent numeric verification (exact row-major index replay per device,
  not just a size/divisibility check) shows both satisfy the same
  mesh-divides-the-outer-factor condition the acceptance shapes require
  (`4 | 4` and `4 | 32` respectively) and are therefore genuinely derivable
  under this fix — their current `ExpectedError` was a snapshot of the
  pre-fix algorithm's blanket "no position may split" restriction, not an
  intentional negative contract. Milestone M1 migrates them to derive
  cases (mirrors the "migrate any pre-existing test that asserted the old
  behavior" precedent in `docs/plans/wi07-typeinfer-visitor-rebuild.md`
  Milestone M3). This is called out here per "no silent design changes" —
  the alternative (leaving a narrower rule that keeps these two failing)
  is not reachable without ALSO rejecting the dequant acceptance shape,
  since both existing cases and the dequant shape are the identical
  single-old-axis-splits-into-two-new-axes pattern.
- Never modify the main worktree (`/home/qihang.zheng/zqh/TileFoundry`) or
  TileOpsGov; no pushes, no PRs.

## Milestones

### Milestone M0: Divisibility-based position splitting in `_carry_sharded_reshape`

#### Depends
- None

#### Related Files
- `src/tilefoundry/ir/hir/tensor/reshape.py`

#### Plan
- [x] step 0.1 In the new-axis walk (`reshape.py:66-86`), when the next cute
      position `cs` would overshoot the current new axis (`prod * cs > d`),
      compute `needed = d // prod`; if `d % prod != 0` or `cs % needed != 0`,
      keep the existing fail-closed `return None` (genuine straddle,
      unchanged behavior for non-dividing cases).
- [x] step 0.2 Otherwise split `cs` into `(needed, residual = cs // needed)`:
      append `needed` as the new position completing the current axis
      (stride `cute_strides[ci] * residual` when strides are concrete);
      push `residual` back as a synthetic pending position (stride
      `cute_strides[ci]`, no `old_pos` link) for the next axis in the walk,
      instead of advancing `ci`.
- [x] step 0.3 When the split position carries a `Split` attr (`ci in
      old_to_new`-eligible), require the bound mesh extent divides `needed`;
      fail closed (`return None`) if it does not. On success, remap the
      `Split` to the new position holding `needed` (the residual position
      is plain, no `Split`).
- [x] step 0.4 Preserve existing behavior for: whole-position merges (no
      split needed), unit-axis insertion/removal, `Partial`/`Broadcast`
      carry-through — no regression in the non-splitting paths.

#### Acceptance Criteria
- [x] AC-0-1: `<worktree>/.venv/bin/python -m pytest
      /home/qihang.zheng/zqh/TileOpsGov/research/deepseek-v4-flash-dataflow/gap-repros/repro_g08.py
      -v` — 3 passed, repro file unmodified.
- [x] AC-0-2: `pytest tests/ops/test_reshape.py -v` — every case from
      `CASES` that predates this change and is NOT one of the two flagged
      in Constraints stays green with unchanged expectations.
<!-- policy_ac:start -->
- [x] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M1: Regression migration for the two now-derivable existing cases

#### Depends
- M0

#### Related Files
- `tests/ops/test_reshape.py`

#### Plan
- [x] step 1.1 Rewrite `straddle_fails_closed` to `split_divides_carries`:
      `(16, 8) Split(0) mesh(4,) → (4, 32)` now expects the derived
      `ShardLayout` (`Split(0)` on the size-4 leading axis, local extent 1;
      residual 4 merges with the whole size-8 position into the trailing
      axis). Docstring restated around the mesh-divides condition instead
      of "straddles".
- [x] step 1.2 Rewrite `split_dim_straddles_fails_closed` to
      `flat_split_divides_carries`: `(4096,) Split(0) mesh(4,) → (32, 128)`
      expects `Split(0)` on the size-32 leading axis (local extent 8),
      residual 128 plain. Docstring updated accordingly.
- [x] step 1.3 Add a genuinely-straddling negative case under the
      (now-vacated) name `straddle_fails_closed`, mirroring the repro's
      `[6,4] Split(0) mesh(2,) → [3,8]` (mesh extent 2 does not divide the
      outer sub-factor 3) — the file keeps a fail-closed regression guard
      of its own, not just via the external repro.

#### Acceptance Criteria
- [x] AC-1-1: `pytest tests/ops/test_reshape.py -v` fully green with the two
      cases now asserting derived output, plus a new straddle-negative case.
- [x] AC-1-2: Commit body names both migrated tests and the one-line reason
      (pre-fix blanket restriction vs. the mesh-divides-the-outer-factor
      contract), per `docs/develop.md` Scope discipline.
<!-- policy_ac:start -->
- [x] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M2: Spec edit — `Reshape` gets its own `hir.md` entry

#### Depends
- M0

#### Related Files
- `docs/spec/hir.md`

#### Plan
- [ ] step 2.1 Split `Reshape` out of the grouped "Reshape / Transpose /
      Slice / Concat / Stack / ShapeOf / Rank" consensus one-liner
      (`hir.md` §1.3 `ir/hir/tensor/`) into its own entry, following the
      Unified Entry Format (source-form `class Reshape(Op): ...`,
      `- constraints:` list) — matching how `Cast`/`Gather`/`Reduce` already
      have their own entries in the same subsection for the same reason
      (non-consensus, custom constraints).
- [ ] step 2.2 State the implemented subset as a constraint: a genuine
      `ShardLayout` carries through `Reshape` when every cute position lies
      entirely within one new axis, OR (this WI) a `Split`-bound position
      subdivides at a boundary where the bound mesh extent divides the outer
      sub-factor — `Split` relocates there, the inner residual becomes a
      plain cute position.
- [ ] step 2.3 State the unimplemented target contract as a constraint,
      following the `Gather` `batch_dims` precedent (`hir.md` §1.3, "not yet
      supported and MUST fail closed"): arbitrary rank-N regroup — any
      further nesting of splits, or a device-owned block that does not align
      with a single split point — is not implemented and MUST fail closed
      (this is the existing `_carry_sharded_reshape` → `None` →
      `VerifyError` path; no separate check needed for this to be true, only
      the spec sentence recording it as the boundary, not an oversight).

#### Acceptance Criteria
- [ ] AC-2-1: `docs/spec/hir.md`'s new `Reshape` entry has no plan /
      milestone / task-ID / commit-hash reference (SPEC-RULES.md
      Constraints); no `Non-Goals` / `Future` / `Tests` section.
- [ ] AC-2-2: `python scripts/spec_rules_lint.py docs/spec/hir.md` (or
      the repo's equivalent entry point) passes.
<!-- policy_ac:start -->
- [ ] Spec section MUST NOT reference plans, milestones, task IDs, commit hashes, PR numbers, agent / human names, or thread / message IDs. <!-- policy_ac: spec_discipline-0 -->
- [ ] Spec section MUST NOT carry a `Non-Goals` / `Future / TODO` / `Out of scope` section. <!-- policy_ac: spec_discipline-1 -->
- [ ] Spec section MUST NOT carry a `Tests` / `Testing` / `Test plan` section or a list of test names. <!-- policy_ac: spec_discipline-2 -->
- [ ] Spec section MUST be in English. <!-- policy_ac: spec_discipline-3 -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M3: Full-suite validation

#### Depends
- M1
- M2

#### Related Files
- `tests/`

#### Plan
- [ ] step 3.1 Run the full suite (`pytest tests/`); triage every failure
      against Milestone M0/M1's scope — any unrelated failure is a STOP
      item, not a silent fix.
- [ ] step 3.2 Re-run `repro_g08.py` after the full suite to confirm no
      cross-test interference.

#### Acceptance Criteria
- [ ] AC-3-1: `pytest tests/` fully green in the worktree venv.
- [ ] AC-3-2: `<worktree>/.venv/bin/python -m pytest repro_g08.py -v`
      (run from `gap-repros/`) — 3 passed.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

## Execution Preflight

<!-- This block is auto-filled by `scripts/finalize_plan_context.py`.
     It surfaces the policy entries from `docs/policies/project-policy.json`
     whose `when.path_glob` matches the plan-level `### Related Files`
     above, so the implementer and reviewer can see the cross-cutting
     rules / knowledge for this plan in one place. Leave the marker
     pair below and run the finalizer; do not hand-edit the body. -->
<!-- policy_preflight:start -->

### Policy Rules & Knowledge

- Comment hygiene — Code comments only describe local logic; no plan / milestone / version / review / discussion narration. (see `docs/develop.md § Code comments`) <!-- policy_rules: comment_hygiene -->
- Scope discipline — One commit touches only what the current task requires; unrelated edits / submodule bumps / autoformat go in separate commits or are called out explicitly. (see `docs/develop.md § Scope`) <!-- policy_rules: scope_discipline -->
- Test discipline — Tests exercise intended behaviour; no excessive defensive / catch-all tests that lock implementation detail. (see `docs/develop.md § Tests`) <!-- policy_rules: test_discipline -->
- Spec discipline — Spec sections follow the spec-writing contract: principle-first, RFC 2119 style, no cross-layer leakage. (see `docs/SPEC-RULES.md § Principle`, `docs/SPEC-RULES.md § Constraints`) <!-- policy_rules: spec_discipline -->
<!-- policy_preflight:end -->
