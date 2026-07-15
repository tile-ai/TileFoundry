---
type: FEAT
component: relation_build
target_repo: tilefoundry
---

# [FEAT][relation_build] DimExpr support: affine inverse + opaque floordiv parameter

## Description

### Symptom / Motivation

Gap register G-2/G-3 (`research/deepseek-v4-flash-dataflow/11-tilefoundry-gaps-decode-module.md`
§G-2, §G-3, TileOpsGov). Relation-derived ops (`MatMul`, `Binary`) push an
extent through a `DimExpr → isl string (build) → isl extent → DimExpr
(recover)` round-trip. Today the two sides are not mutual inverses:

- build side accepts only `+`/`-` and `*`-with-a-constant; a `DimFloorDiv`
  extent (`P // 4`) raises before it ever reaches isl.
- recovery side only accepts a pure constant or a single coefficient-1,
  offset-0 bare `DimVar`; even the purely-affine `128 + P` extent (isl
  normalizes it to `127 + P`, then `+1`) cannot be inverted back to a
  `ShapeDim`.

`gap-repros/repro_g02_g03.py` (TileOpsGov) reproduces both as 4 FAIL with 1
green guard (bare `DimVar` matmul). Impact: DSV4 plan decision 3's single
`DimVar P` with derived cache dims (`P // 4`, `128 + P // 4`) is blocked at
every relation-derived op and is worked around today with independent bare
`DimVar`s; hand-written shape rules (`insert_slice`/`gather`/`topk`/`reduce`/
`softmax`) already accept the same derived dims — only the relation path
fails (doc-11 note N-2, TileOpsGov).

### Root Cause Analysis

- `_dim_to_isl` (`src/tilefoundry/visitor_registry/relation_build.py:64-66`)
  falls through to a blanket `NotImplementedError` for any dim op it does not
  special-case; `DimFloorDiv` is not one of the special-cased ops (only
  `DimAdd`/`DimSub`/`DimMul`), so it is rejected outright rather than bound to
  an isl parameter.
- `_extent_of_domain_dim` (`relation_build.py:108-115`) recovers a `ShapeDim`
  from an isl affine's normalized `const + Σ coeff_i·param_i` form only when
  there are zero nonzero params (pure constant) or exactly one nonzero param
  with `coeff == 1` and `const == 0` (a bare `DimVar`). Any other affine
  combination — a nonzero constant offset, a non-unit coefficient, or more
  than one param — falls through to the closing `raise ValueError` at line
  115, even though the isl side has already normalized it losslessly.

### Related Files
- `src/tilefoundry/visitor_registry/relation_build.py`
- `tests/analysis/test_relation_build.py`
- `docs/spec/visitor-registry.md`
- `research/deepseek-v4-flash-dataflow/gap-repros/repro_g02_g03.py` (TileOpsGov, read-only acceptance input)

## Goal

Make `relation_build`'s build/recover round-trip invert any affine
combination of `DimVar`s and accept `DimFloorDiv` extents via an opaque isl
parameter, so relation-derived ops accept the same derived shape dims
hand-written shape rules already do.

## Constraints

- Scheme B (opaque parameter) per user decision O12.2
  (`plans/2026-07-15-wi12-relation-dimexpr.md`, TileOpsGov): symbol×symbol
  `DimMul`, `DimMod`, `DimMax`, `DimMin` keep raising `NotImplementedError` —
  out of scope, not exercised by DSV4.
- All implementation changes live in
  `src/tilefoundry/visitor_registry/relation_build.py` (helpers may be added
  there); no new analysis pass, no other op files — dispatched in parallel
  with wi07/08/09/11 with zero file overlap.
- `repro_g02_g03.py` (TileOpsGov) is read-only acceptance input — the fix
  MUST make all 5 cases pass without editing that file.
- The same canonicalized `DimFloorDiv` expression MUST bind to the same isl
  parameter name everywhere it recurs within a single `build_domain` call
  (keyed table), so `build_domain`'s existing conflicting-bounds check stays
  meaningful for the opaque parameter too.
- **Discovered conflict (flagged, not silently resolved):**
  `tests/analysis/test_relation_build.py::test_build_domain_non_affine_extent_raises`
  currently asserts `build_domain` raises `NotImplementedError` for a bare
  `DimVar // const` extent. That is exactly the behavior this plan replaces
  (Milestone M1) — the case was a snapshot of the pre-fix blanket rejection,
  not an intentional negative contract distinct from the opaque-parameter
  path. Milestone M1 migrates it to assert the new success behavior and adds
  a narrower negative case (symbolic divisor) so the file keeps its own
  fail-closed regression guard.
- Work stays in this worktree/branch; never modify the main worktree or
  TileOpsGov; no pushes, no PRs.

## Milestones

### Milestone M0: Recovery-side affine inverse

#### Depends
- None

#### Related Files
- `src/tilefoundry/visitor_registry/relation_build.py`
- `tests/analysis/test_relation_build.py`

#### Plan
- [ ] step 0.1 Replace `_extent_of_domain_dim`'s narrow accept-check
      (`relation_build.py:108-115`) with a general reconstruction: for each
      nonzero `(param_name, coeff)` pair, look up the param's original
      `DimExpr` (bare `DimVar`s are already collected into `dimvars` from
      `input_types`; `coeff != 1` wraps the base via
      `simplify_dim(DimMul, (coeff, base))`), then fold the constant term (if
      nonzero) and every param term left-to-right via
      `simplify_dim(DimAdd, ...)`. A single coeff-1, zero-const param stays a
      bare passthrough (no wrapping) so the existing bare-`DimVar` recovery
      is unchanged bit-for-bit.
- [ ] step 0.2 Keep the existing fail-closed checks unchanged: piecewise
      `dim_max`, a `DIV` term in the affine, or a param name with no known
      `DimExpr` all still raise `ValueError`.
- [ ] step 0.3 Add positive unit tests to `test_relation_build.py` for
      `shape_from_relation` recovering a nonzero-constant sum (`M + 1`) and a
      non-unit coefficient (`2 * M`) shape, alongside the existing bare-param
      case.

#### Acceptance Criteria
- [ ] AC-0-1: `repro_g02_g03.py::test_g03_matmul_over_affine_add_dim` passes.
- [ ] AC-0-2: `repro_g02_g03.py::test_guard_bare_dimvar_matmul_ok` stays
      green (bit-for-bit same recovered `DimVar`, not a re-wrapped
      equivalent).
- [ ] AC-0-3: new `test_relation_build.py` affine-sum / scaled-coefficient
      recovery cases pass.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M1: Opaque isl parameter for `DimFloorDiv`

#### Depends
- M0

#### Related Files
- `src/tilefoundry/visitor_registry/relation_build.py`
- `tests/analysis/test_relation_build.py`

#### Plan
- [ ] step 1.1 In `_dim_to_isl`, add a `DimFloorDiv` case: require the
      divisor be a positive constant (else `NotImplementedError`, symbolic
      divisor stays unsupported); canonicalize the whole expression via
      `simplify_dim`; look it up in a module-level keyed registry mapping
      canonical `DimExpr` → isl parameter name, minting and recording a fresh
      name on miss (same expression anywhere in the process ⇒ same name).
- [ ] step 1.2 Add an `_affine_bounds` helper mirroring `_dim_to_isl`'s
      affine dispatch (`DimVar`/`Constant`/`DimAdd`/`DimSub`/`DimMul`-by-const)
      that computes the dividend's half-open `[lo, hi)` value bounds; derive
      the opaque parameter's own bound as
      `[lo // c, (hi - 1) // c + 1)` and register it into the per-call
      `params` dict exactly like a `DimVar` bound.
- [ ] step 1.3 Register the reverse mapping (isl parameter name → original
      canonical `DimExpr`) in the same module-level table; `shape_from_relation`
      /`_extent_of_domain_dim` consult it as a fallback alongside the
      `input_types`-derived `dimvars`, so an opaque parameter resolves back to
      its original expression through the M0 bare-param / affine-rebuild path
      with no new recovery-side logic.
- [ ] step 1.4 Migrate
      `test_build_domain_non_affine_extent_raises` (flagged in Constraints)
      to assert the new success behavior (isl parameter bound derived from
      the dividend); add a narrower negative case asserting a symbolic
      divisor (`DimFloorDiv(M, N)`, both `DimVar`) still raises
      `NotImplementedError`.
- [ ] step 1.5 Add positive unit tests: opaque-parameter bound derivation
      from a `DimVar` dividend, and same-expression-same-name dedup (the same
      canonical `DimFloorDiv` used twice in one `build_domain` call yields one
      isl parameter, not two).

#### Acceptance Criteria
- [ ] AC-1-1: `repro_g02_g03.py::test_g02_matmul_over_floordiv_dim` and
      `test_g02_binary_over_floordiv_dim` pass.
- [ ] AC-1-2: `repro_g02_g03.py::test_g02_g03_matmul_over_composite_dim`
      (`128 + P // 4`, exercises M0 + M1 together) passes.
- [ ] AC-1-3: updated + new `test_relation_build.py` cases pass; the
      symbolic-divisor negative case still raises `NotImplementedError`.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M2: Spec accuracy — domain parameter description

#### Depends
- M1

#### Related Files
- `docs/spec/visitor-registry.md`

#### Plan
- [ ] step 2.1 §4.1's `domain` description states a dynamic extent becomes
      "an isl parameter (one per `DimVar`)" — update it to also cover the
      opaque parameter a non-affine-in-isl extent (`DimFloorDiv`) binds to,
      so the sentence stays accurate to what `build_domain` now does.

#### Acceptance Criteria
- [ ] AC-2-1: `docs/spec/visitor-registry.md` has no plan/milestone/task-ID
      reference, no `Non-Goals`/`Tests` section, English only.
- [ ] AC-2-2: `python scripts/spec_rules_lint.py docs/spec/visitor-registry.md`
      passes.
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
- [ ] step 3.1 Run `repro_g02_g03.py` — 5/5 green (4 flips + the bare-`DimVar`
      guard).
- [ ] step 3.2 Run the full suite (`pytest tests/ -q`); triage any failure
      against M0/M1's scope — an unrelated failure is a STOP item.
- [ ] step 3.3 Run `scripts/spec_rules_lint.py` / `scripts/spec_entropy_lint.py`
      on every touched file.

#### Acceptance Criteria
- [ ] AC-3-1: `.venv/bin/python -m pytest tests/ -q` fully green.
- [ ] AC-3-2: `.venv/bin/python -m pytest repro_g02_g03.py -v` — 5 passed.
- [ ] AC-3-3: both lint scripts clean on touched files.
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
