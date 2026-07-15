---
type: REFACTOR
component: typeinfer
target_repo: tilefoundry
---

# [REFACTOR][typeinfer] Elaboration-based call model + typeinfer as pure per-node visitor

## Description

### Symptom / Motivation

Gap register G-10 (`gap-repros/repro_g10.py`, TileOpsGov). A wildcard
(`layout=None`) `hir.Function` parameter is documented to accept an argument
of any layout and propagate it through the body
(`docs/spec/hir.md` §1.1 "Call typing"), but when the callee's body ends in a
core `Tuple` (`return y, y`), the `Split` sharding carried by the caller's
argument is silently dropped: the returned `TupleType`'s fields come back
`layout=None` instead of carrying `Split`. `repro_g10.py::test_g10_b2_*`
reproduces this; `b1`/`b3`/`b4` are green regression guards (`b4` proves the
explicit-annotation layout *contract* half already works — only the
*wildcard-propagation* half is broken).

### Root Cause Analysis

- `TypeInferContext._compute`
  (`src/tilefoundry/visitor_registry/contexts.py:58-75`) is an `isinstance`
  chain that special-cases only `Constant` / `Var` / `Call`. Any other `Expr`
  subclass — `Tuple`, `GridRegionExpr` — falls through to the last branch
  (`contexts.py:72-74`): `declared = getattr(expr, "type", None); return
  declared` — i.e. it returns whatever `.type` was stamped on the node at
  **parse time**, never recursing into the node's children. For the repro's
  `inner_multi` (`return y, y`, a core `Tuple`), this is the parse-time
  (unsharded) `TupleType` — stale under a differently-typed call.
- `_typeinfer_hir_function_call`
  (`src/tilefoundry/ir/hir/function.py:128-151`) re-derives a callee's body
  via an "overlay": a scratch `TypeInferContext` seeded with
  `sub.cache[param] = arg_ty` (`function.py:146-150`), then
  `sub.type_of(callee.body)` (`function.py:151`). This correctly re-derives a
  `Call` node (registry-dispatched, e.g. `Binary`) but, per the bug above,
  returns the stale stamped type when `callee.body` is a `Tuple`. It also
  never reconstructs IR — every call site shares the *same* body `Expr`
  objects, which conflicts with "types stamped on nodes are the single
  source of truth" once two call sites need two different concrete types for
  the same shared node.

### Related Files
- `src/tilefoundry/visitor_registry/contexts.py`
- `src/tilefoundry/visitor_registry/visitors.py`
- `src/tilefoundry/ir/hir/function.py`
- `src/tilefoundry/parser/base.py`
- `docs/spec/hir.md`
- `docs/spec/analysis.md`
- `docs/spec/visitor-mutator.md`
- `docs/spec/visitor-registry.md`
- `tests/ir/test_function_call_typeinfer.py`
- `tests/parser/hir/test_nested_func_call.py`

## Goal

Make HIR function-call typing an elaboration (per-call-site IR construction,
one exhaustive per-`Expr`-kind `TypeInferVisitor`) so a wildcard parameter's
argument layout propagates through the whole callee body, including a
`Tuple` / `GridRegionExpr` return.

## Constraints

- Never modify the main worktree (`/home/qihang.zheng/zqh/TileFoundry`) or
  TileOpsGov; no pushes, no PRs.
- `gap-repros/repro_g10.py` is read-only input — the fix must make all 4
  tests green without editing that file (confirmed feasible: the fix is
  fully internal to the registered `Function` typeinfer handler, which every
  driver in the repro already goes through).
- DimVar shape dispatch (`Function.variants` / `.specialize` envelope
  matching) is unchanged — out of scope (owned by WI-8/9/10 for any shard
  *propagation rule* changes; this WI only touches the call/typeinfer
  mechanism).
- A `Call`'s derived type always comes from re-deriving the callee body,
  never from trusting a possibly-stale `Function.return_type` field —
  locked today by
  `tests/ir/test_function_call_typeinfer.py::test_explicit_sharded_formal_accepts_matching_actual`
  (the fixture's declared `return_type` is deliberately "wrong" to prove
  this).
- Instance dedup (skip rebuilding IR when every bound parameter type already
  equals the callee's current parameter type) is an allowed optimization,
  not a semantic — and MUST preserve object identity where already locked:
  `tests/parser/hir/test_nested_func_call.py::test_parser_emits_call_with_hir_function_target`
  asserts `Call.target is` the original sibling `Function` for a
  same-annotation call.
- `TypeInferVisitor` becomes exhaustive over every `Expr` subclass that
  reaches `ctx.type_of` across BOTH hir and tir (`Var`, `Constant`, `Call`,
  `Tuple`, `GridRegionExpr`, and tir's `ShapeOf` — confirmed via
  `ir/tir/verify.py::_verify_symbol_call` forwarding `ShapeOf` args through
  `ctx.type_of`); an `Expr` kind with no rule raises, it does not fall back
  to a stamped field.
- No "`assemble_step`" module exists in this codebase (grepped, zero hits) —
  the plan's "programmatic builders" constraint has no concrete file to
  migrate here; noted, not blocking.

## Milestones

### Milestone M0: Pure per-kind `TypeInferVisitor`

#### Depends
- None

#### Related Files
- `src/tilefoundry/visitor_registry/contexts.py`
- `src/tilefoundry/visitor_registry/visitors.py`

#### Plan
- [x] step 0.1 Delete `TypeInferContext._compute`'s `isinstance` chain
      (`contexts.py:58-75`); `TypeInferContext.type_of` delegates to
      `TypeInferVisitor(self).visit(expr)` and becomes a walk-local
      cache + `error()` helper only.
- [x] step 0.2 `TypeInferVisitor` (`visitors.py`) grows exhaustive
      `visit_Var` / `visit_Constant` / `visit_Call` (registry dispatch,
      moved from `_compute`) / `visit_Tuple` (structural: `TupleType` over
      `ctx.type_of(e)` for each element) / `visit_GridRegionExpr`
      (carry/body: no-carry → body type, single/multi carry → the
      `carried_args` phi Vars' own declared types, matching what the parser
      already stamps) / `visit_ShapeOf` (tir: returns the node's own
      declared `.type` — a `ShapeOf` is always constructed with an explicit
      type, never derived from children).
- [x] step 0.3 Override `generic_visit` to raise via `ctx.error` for any
      `Expr` subclass with no explicit rule (no silent fallback).

#### Acceptance Criteria
- [x] AC-0-1: `TypeInferContext` has no `_compute` method; every dispatch
      rule lives on `TypeInferVisitor` as a `visit_<Kind>` method.
- [x] AC-0-2: `pytest tests/ops tests/ir_types tests/analysis -q` stays
      green (Call-kind typeinfer regression guard).
- [x] AC-0-3: `pytest tests/passes tests/e2e tests/codegen tests/runtime -q`
      stays green (tir `ShapeOf` / verify regression guard).
<!-- policy_ac:start -->
- [x] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M1: Elaboration construction entry + call-site wiring

#### Depends
- M0

#### Related Files
- `src/tilefoundry/ir/hir/function.py`
- `src/tilefoundry/parser/base.py`

#### Plan
- [x] step 1.1 Add `elaborate(callee, arg_types, ctx=None) -> Function` in
      `function.py`: arity check; per-param bind via the existing
      `_check_arg_against_param` semantics generalized to return the *bound*
      type (wildcard → argument's full type incl. `ShardLayout`; explicit
      layout → contract, mismatch raises); a dispatch prototype
      (`variants != ()` or `body is None`) short-circuits to `callee`
      unchanged (no body to elaborate); when every bound type already
      equals the callee's current param type, return `callee` unchanged
      (dedup); otherwise substitute fresh param `Var`s into a rebuilt body
      via a memoized, identity-preserving `_Elaborator(ExprMutator)` (keyed
      by `id()` so SSA-as-DAG sharing survives the rebuild) that re-stamps
      every changed node's `.type` through `TypeInferContext.type_of`, and
      return a new `Function.build(...)` instance.
- [x] step 1.2 Rewrite `_typeinfer_hir_function_call`
      (`function.py:128-151`) to: compute `arg_types` from `ctx.type_of` on
      each `call.args` entry, call `elaborate(callee, arg_types, ctx)`, and
      return the FRESHLY re-derived type of the resulting instance's body
      (a new `TypeInferContext` walk — never trust `instance.return_type`
      directly, per the Constraints section) — or `instance.return_type`
      when `instance.body is None` (dispatch prototype). Delete the
      `sub = TypeInferContext(...)` overlay-seeding code.
- [x] step 1.3 `parser/base.py::_build_function_call` calls `elaborate`
      before constructing the `Call` so `Call.target` is the actual
      elaborated instance (not just `Call.type`) — required so the viewer's
      inline-expansion (`inspection/viewer/builder.py`, unchanged) and any
      body/printer read of `call.target` see correctly-propagated types.

#### Acceptance Criteria
- [x] AC-1-1: `<worktree>/.venv/bin/python -m pytest
      /home/qihang.zheng/zqh/TileOpsGov/research/deepseek-v4-flash-dataflow/gap-repros/repro_g10.py
      -v` — 4 passed, repro file unmodified.
- [x] AC-1-2: `pytest tests/ir/test_function_call_typeinfer.py
      tests/parser/hir/test_nested_func_call.py -v` green with no test-body
      changes (only doc/comment updates allowed if wording goes stale).
- [x] AC-1-3: `pytest tests/ir tests/parser tests/dsl tests/inspection -q`
      green.
<!-- policy_ac:start -->
- [x] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M2: Spec edits

#### Depends
- M1

#### Related Files
- `docs/spec/hir.md`
- `docs/spec/analysis.md`
- `docs/spec/visitor-mutator.md`
- `docs/spec/visitor-registry.md`

#### Plan
- [x] step 2.1 Rewrite `hir.md` §1.1 "Call typing" to the elaboration model:
      per-call-site construction, wildcard-vs-contract annotation semantics,
      dedup as an optimization not a semantic, dispatch prototypes
      unaffected.
- [x] step 2.2 `analysis.md` §1 gains a short note that relation-derived
      type behavior composes under elaboration (a callee's body is
      re-derived per call site; per-op relation rules are unaffected).
- [x] step 2.3 `visitor-registry.md` §4 updates the `TypeInferVisitor` /
      `TypeInferContext` contract text to match the exhaustive-visitor
      shape (no more "registry consulted by `_compute`" framing).
- [x] step 2.4 `visitor-mutator.md` touch-up only if the elaboration
      mutator's memoized-rebuild pattern needs a documented note under
      `ExprMutator`; otherwise no change (call out N/A in the commit if so).

#### Acceptance Criteria
- [x] AC-2-1: `python scripts/spec_rules_lint.py` (or equivalent existing
      spec lint entry point) passes on the touched spec files.
- [x] AC-2-2: No spec file mentions this plan, milestones, or task IDs
      (SPEC-RULES.md Constraints).
<!-- policy_ac:start -->
- [x] Spec section MUST NOT reference plans, milestones, task IDs, commit hashes, PR numbers, agent / human names, or thread / message IDs. <!-- policy_ac: spec_discipline-0 -->
- [x] Spec section MUST NOT carry a `Non-Goals` / `Future / TODO` / `Out of scope` section. <!-- policy_ac: spec_discipline-1 -->
- [x] Spec section MUST NOT carry a `Tests` / `Testing` / `Test plan` section or a list of test names. <!-- policy_ac: spec_discipline-2 -->
- [x] Spec section MUST be in English. <!-- policy_ac: spec_discipline-3 -->
- [x] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M3: Full-suite validation and regression migration

#### Depends
- M1
- M2

#### Related Files
- `tests/`

#### Plan
- [x] step 3.1 Run the full suite (`pytest tests/`); triage every failure.
- [x] step 3.2 Migrate any pre-existing test that asserted the old
      overlay/layout-mismatch-permissive behavior to the elaboration /
      instantiation pattern; record which tests and why in the milestone
      commit body (develop.md Scope discipline). Result: zero — the full
      suite passed unmodified (no pre-existing test exercised the
      overlay-permissive scenario the fix corrects).
- [x] step 3.3 Re-run the full suite to green.

#### Acceptance Criteria
- [x] AC-3-1: `pytest tests/` fully green in the worktree venv.
- [x] AC-3-2: `<worktree>/.venv/bin/python -m pytest repro_g10.py -v` (run
      from `gap-repros/`) — 4 passed.
- [x] AC-3-3: The commit body enumerates every migrated test file and the
      one-line reason (old assumption vs. new elaboration contract). Result:
      none migrated — `pytest tests/` was 827 passed / 0 failed on the first
      run after M1, with no test-file edits.
<!-- policy_ac:start -->
- [x] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M4: Rework — 3 acceptance-review defects (D1/D2/D3)

#### Depends
- M1
- M3

#### Related Files
- `src/tilefoundry/ir/hir/function.py`
- `src/tilefoundry/parser/base.py`
- `tests/ir/test_function_call_typeinfer.py`
- `tests/parser/hir/test_nested_func_call.py`

#### Plan
- [ ] step 4.1 D1: `_Elaborator` (`function.py`) gains `visit_GridRegionExpr`,
      re-stamping `carried_args` from the rewritten `init_args` (hir.md §1.2:
      "the first-iteration value of each `carried_args` phi is its
      `init_args` entry" — the same rule the parser applies) before
      rewriting `body` / `yield_values`, so a wildcard-propagated layout
      survives a carrying loop. `TypeInferVisitor.visit_GridRegionExpr`
      (`visitors.py`) needs no change once `carried_args` is correctly
      re-stamped.
- [ ] step 4.2 D2: `_Elaborator` gains `visit_Call`, which re-elaborates a
      nested Call's `target` (when it is a hir `Function`) against the
      call's rewritten arg types, so `Call.target` — not just `Call.type`
      — reflects the fresh per-call-site instance through a multi-level
      wildcard chain.
- [ ] step 4.3 D3: thread the originating `Call` through `elaborate()` /
      `_bind_param_type()` (`function.py`) and the parser's pre-Call
      surrogate (`parser/base.py::_build_function_call`) so an
      arity/bind `VerifyError` reports `at <call.loc>` instead of the
      callee's (always-`None`) own `.loc`.
- [ ] step 4.4 Add regression cases to `tests/ir/test_function_call_typeinfer.py`
      (D1, D3) and `tests/parser/hir/test_nested_func_call.py` (D2)
      covering the three defects' observable behavior.

#### Acceptance Criteria
- [ ] AC-4-1: `gap-repros/repro_g10b.py::test_d1_carry_loop_propagates_split` passes.
- [ ] AC-4-2: `gap-repros/repro_g10b.py::test_d2_nested_call_target_reelaborated` passes.
- [ ] AC-4-3: `gap-repros/repro_g10b.py::test_d3_arg_mismatch_error_keeps_call_loc` passes.
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
