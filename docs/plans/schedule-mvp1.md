---
type: FEAT
component: schedule MVP
target_repo: tilefoundry
---

# [FEAT][schedule] CTA AutoDist MVP

## Description

### Symptom / Motivation

The existing schedule prototype solves a toy entry function with a
target-specific ring-style scheduler. The MVP now needs one parser, metadata
attached to immutable HIR expressions, whole-call-graph distribution, injected
target services, concrete HIR materialization, and an Agent-readable result.

### Root Cause Analysis

The current prototype merges parser-side schedule input, logical graph facts,
target candidates, CTA timing, and materialization. It also treats one entry
body as the solve boundary and has no generic Agent Constraint metadata or
Module-level report.

### Related Files

- `docs/plans/schedule-mvp1.md`
- `src/tilefoundry/ir/core/expr.py`
- `src/tilefoundry/ir/hir/function.py`
- `src/tilefoundry/parser/__init__.py`
- `src/tilefoundry/parser/hir_parser.py`
- `src/tilefoundry/inspection/python_printer.py`
- `src/tilefoundry/lower.py`
- `src/tilefoundry/schedule/__init__.py`
- `src/tilefoundry/schedule/constraints/__init__.py`
- `src/tilefoundry/schedule/constraints/base.py`
- `src/tilefoundry/schedule/constraints/layout.py`
- `src/tilefoundry/schedule/graph.py`
- `src/tilefoundry/schedule/candidates.py`
- `src/tilefoundry/schedule/solver.py`
- `src/tilefoundry/schedule/materialize.py`
- `src/tilefoundry/schedule/report.py`
- `src/tilefoundry/providers/registry.py`
- `src/tilefoundry/providers/services.py`
- `src/tilefoundry/providers/cuda/provider.py`
- `src/tilefoundry/providers/cuda/profiles.py`
- `src/tilefoundry/providers/cuda/cost_model.py`
- `tests/models/deepseek_v4/__init__.py`
- `tests/models/deepseek_v4/moe.py`
- `tests/models/deepseek_v4/test_moe.py`
- `tests/schedule/test_autodist.py`
- `tests/schedule/test_constraints.py`
- `tests/schedule/test_graph.py`
- `tests/schedule/test_candidates.py`
- `tests/schedule/test_provider.py`
- `tests/schedule/test_solver.py`
- `tests/schedule/test_materialize.py`
- `tests/schedule/test_report.py`

## Goal

Implement a deterministic one-dimensional CTA AutoDist path from a candidate
Python file to a concrete, constraint-free HIR Module and structured report,
covering the complete call graph reachable from `Module.entry`.

## Constraints

- The Agent owns mathematical structure, function duplication, and `where(...)` constraints; AutoDist owns only legal distribution choices and reshards.
- There is one HIR parser and one canonical Python printer for Functions and Modules.
- Metadata is generic IR metadata and does not affect type inference, equality, hashing, or logical fingerprints.
- `StorageConstraint`, `ConstraintList`, `parse_schedule_func`, and the old CTA-internal ring/start-tick solver are removed from the new MVP path.
- Core AutoDist code depends on injected provider services and never imports CUDA provider code or branches on target types.
- CTA topology is one-dimensional and contiguous in v1; exact divisible splits only, with no tail or masked work.
- The H200 profile has 132 SMs, but CTA counts are selected by legality and cost and are not fixed to research placeholders.
- The workflow materializes ordinary HIR only. TIR lowering, CUDA generation, storage placement, buffers, barriers, rings, and CTA-internal pipelines remain out of scope.
- The existing real DSV4 model organization under `tests/models/deepseek_v4` is preserved and extended only as needed.

## Milestones

### Milestone M0: Contract and baseline

#### Depends
- None

#### Related Files
- `docs/plans/schedule-mvp1.md`

#### Plan
- [ ] step 0.1 Replace this plan with the finalized CTA AutoDist contract and run `scripts/finalize_plan_context.py`.
- [ ] step 0.2 Record the current branch, environment import, baseline tests, and old schedule boundary before implementation.

#### Acceptance Criteria
- [ ] AC-0-1: The plan states the parser, metadata, whole-call-graph, provider IoC, candidate, solver, materialization, report, DSV4, and scope-exclusion contracts.
- [ ] AC-0-2: The finalized plan contains M1 through M8 with focused verification expectations.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M1: Metadata, constraints, and one parser

#### Depends
- M0

#### Related Files
- `src/tilefoundry/ir/core/expr.py`
- `src/tilefoundry/schedule/constraints/__init__.py`
- `src/tilefoundry/schedule/constraints/base.py`
- `src/tilefoundry/schedule/constraints/layout.py`
- `src/tilefoundry/parser/__init__.py`
- `src/tilefoundry/parser/hir_parser.py`
- `tests/schedule/test_constraints.py`

#### Plan
- [ ] step 1.1 Add immutable `IRMetadata`, `AgentConstraintsMetadata`, `AgentConstraint`, `LayoutConstraint`, `LayoutDimConstraint`, and `PartialConstraint` records.
- [ ] step 1.2 Extend Expr construction/rebuild and parser finalization so inline and standalone `where` forms attach metadata to the exact SSA Expr identity.
- [ ] step 1.3 Remove schedule-specific parser entry points and support `parse_func`, `parse_func_source`, and `parse_module_source` only.

#### Acceptance Criteria
- [ ] AC-1-1: Every metadata class ends in `Metadata`; Expr metadata defaults empty and is excluded from comparison, hashing, and repr.
- [ ] AC-1-2: Wildcard, `D`, `H @ cta`, and Partial sugar create the specified constraints; storage constraints are absent.
- [ ] AC-1-3: Inline and standalone annotations print canonically, parameter constraints print first in the function body, and round-trip through the one parser.
- [ ] AC-1-4: Metadata is attached by memoized DAG rebuild, so shared SSA uses point to the same rebuilt expression and type inference ignores metadata.
- [ ] AC-1-5: `parse_schedule_func`, `parse_schedule_script`, `ConstraintList`, and `ParsedScheduleInput` are unavailable from the public schedule path.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M2: Module parsing and logical graph

#### Depends
- M1

#### Related Files
- `src/tilefoundry/parser/hir_parser.py`
- `src/tilefoundry/parser/__init__.py`
- `src/tilefoundry/schedule/graph.py`
- `src/tilefoundry/schedule/fingerprint.py`
- `tests/schedule/test_graph.py`

#### Plan
- [ ] step 2.1 Parse full Module source and provide canonical Function/Module script output and loading.
- [ ] step 2.2 Expand the Module entry and every reachable HIR func into one ProgramScheduleGraph with function regions, call instances, values, ops, edges, and constraints.
- [ ] step 2.3 Add stable logical identities and a metadata-independent fingerprint; reject recursive calls and unresolved entry Partial values.

#### Acceptance Criteria
- [ ] AC-2-1: `parse_module_source` and `as_script(Function | Module)` support full Module round-trip with no second schedule AST.
- [ ] AC-2-2: `GraphValueRef` contains `(call_path, function_id, local_value_id)` and every call instance has separate execution identity.
- [ ] AC-2-3: Same func definitions share one scheme domain across call instances; caller result constraints bind call edges; conflicts require Agent-side duplication.
- [ ] AC-2-4: Recursive calls fail clearly, internal Partial returns are accepted, entry Partial returns are rejected, and logical fingerprints ignore metadata, names, and locations.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M3: Provider IoC and H200 profile

#### Depends
- M2

#### Related Files
- `src/tilefoundry/ir/target/__init__.py`
- `src/tilefoundry/providers/registry.py`
- `src/tilefoundry/providers/services.py`
- `src/tilefoundry/providers/cuda/provider.py`
- `src/tilefoundry/providers/cuda/profiles.py`
- `tests/schedule/test_provider.py`

#### Plan
- [ ] step 3.1 Add `CudaTarget(arch="sm_90", device="h200_sxm")` fields and global service collection/registry records.
- [ ] step 3.2 Register CUDA architecture, device, CTA schedule, and cost services through `CudaProvider`.
- [ ] step 3.3 Inject resolved services into common AutoDist and reject missing concrete devices for scheduling.

#### Acceptance Criteria
- [ ] AC-3-1: Normal CUDA compilation works without a device, while AutoDist without a concrete device fails clearly.
- [ ] AC-3-2: The H200 profile reports 132 SMs, 4.8e12 HBM bytes/s, dense Tensor Core peaks at half sparsity-advertised peaks, and integer-nanosecond units.
- [ ] AC-3-3: Common AutoDist contains no CUDA-provider import or target-type branch.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M4: Common distribution candidates

#### Depends
- M3

#### Related Files
- `src/tilefoundry/schedule/candidates.py`
- `src/tilefoundry/schedule/distribution.py`
- `src/tilefoundry/schedule/graph.py`
- `tests/schedule/test_candidates.py`

#### Plan
- [ ] step 4.1 Define `DistributionState` and `OpCandidate` with layout, CTA count, Partial state, input/output states, and abstract work.
- [ ] step 4.2 Implement common access/distribution rules for elementwise, cast, reshape, transpose, Tuple, TupleGetItem, matmul, reduce, gather, topk, and func boundaries.
- [ ] step 4.3 Filter to exact divisible splits, reject unsupported operations, and propagate matmul/reduce Partial states without silently defaulting to Broadcast.

#### Acceptance Criteria
- [ ] AC-4-1: Legal candidates are target-neutral and finite; only exact divisible splits are emitted.
- [ ] AC-4-2: Matmul M/N Split and K Partial, reduce non-reduction Split and reduction Partial, gather/topk rules, and tuple/call boundary propagation are tested.
- [ ] AC-4-3: Unsupported rules and illegal divisibility produce diagnostics rather than guessed candidates.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M5: Formula cost and common CP-SAT solve

#### Depends
- M4

#### Related Files
- `src/tilefoundry/schedule/solver.py`
- `src/tilefoundry/schedule/cost.py`
- `tests/schedule/test_solver.py`

#### Plan
- [ ] step 5.1 Implement formula-only H200 cost using dense peaks, HBM bandwidth, CTA share, real FP4/FP8 storage widths, compute dtype, and zero fixed latency.
- [ ] step 5.2 Build a common OR-Tools CP-SAT model for one-dimensional contiguous CTA intervals, call-instance timing, dependency/reshard costs, and NoOverlap2D.
- [ ] step 5.3 Apply hard Agent Constraints, shared func scheme domains, status reporting, infeasibility reports, and entry makespan minimization.

#### Acceptance Criteria
- [ ] AC-5-1: Reported op costs include total/per-CTA FLOPs, bytes, compute time, memory time, and selected duration in integer nanoseconds.
- [ ] AC-5-2: The solver returns OPTIMAL or FEASIBLE_NOT_PROVEN, never softens hard constraints, and rejects higher-rank topology.
- [ ] AC-5-3: All call instances of one func share distribution scheme variables while retaining instance-specific intervals.
- [ ] AC-5-4: Identical layout/submesh edges are zero-cost, Broadcast parent-to-child reads avoid communication, and incompatible Split/Partial/submesh edges use explicit cheapest reshards.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M6: Concrete HIR materialization

#### Depends
- M5

#### Related Files
- `src/tilefoundry/schedule/materialize.py`
- `src/tilefoundry/ir/hir/sharding/reshard.py`
- `src/tilefoundry/ir/types/shard/mesh.py`
- `tests/schedule/test_materialize.py`

#### Plan
- [ ] step 6.1 Materialize complete CTA Meshes, contiguous Mesh slices, concrete ShardLayouts, and Partial states on parameters, returns, and intermediate values.
- [ ] step 6.2 Insert only selected explicit reshard calls and rebuild Function callable types through type inference.
- [ ] step 6.3 Remove all Agent metadata, strip distribution-only information, and verify logical fingerprint preservation.

#### Acceptance Criteria
- [ ] AC-6-1: The solution is an ordinary HIR Module with no solver-private records and zero AgentConstraintsMetadata.
- [ ] AC-6-2: Type inference and verification pass; internal Partial is preserved and entry Partial is rejected.
- [ ] AC-6-3: Stripped solution logical fingerprint equals the candidate fingerprint and no recomputation, replication, or structural fusion is inserted.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M7: SolveResult and report

#### Depends
- M6

#### Related Files
- `src/tilefoundry/schedule/report.py`
- `src/tilefoundry/schedule/__init__.py`
- `tests/schedule/test_report.py`

#### Plan
- [ ] step 7.1 Define `SolveResult` and structured `ScheduleReport` fields for status, fingerprint, target, level, makespan, regions, constraints, choices, costs, reshards, critical path, diagnostics, and fusion opportunities.
- [ ] step 7.2 Add deterministic Markdown rendering derived only from the structured report and write it beside `solution.py`.
- [ ] step 7.3 Report cross-op/cross-func zero-reshard identical-layout/submesh fusion opportunities without applying them.

#### Acceptance Criteria
- [ ] AC-7-1: Structured report is authoritative and Markdown rendering is deterministic and agrees with its fields.
- [ ] AC-7-2: Fusion suggestions require identical layout, identical submesh, and zero reshard.
- [ ] AC-7-3: Infeasible and unsupported cases return diagnostics in the report or raise the specified schedule exception with that report.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M8: Real DSV4 acceptance and simplification

#### Depends
- M7

#### Related Files
- `tests/models/deepseek_v4/__init__.py`
- `tests/models/deepseek_v4/moe.py`
- `tests/models/deepseek_v4/test_moe.py`
- `tests/schedule/test_autodist.py`
- `tests/schedule/test_constraints.py`
- `tests/schedule/test_graph.py`
- `tests/schedule/test_candidates.py`
- `tests/schedule/test_provider.py`
- `tests/schedule/test_solver.py`
- `tests/schedule/test_materialize.py`
- `tests/schedule/test_report.py`

#### Plan
- [ ] step 8.1 Keep the real DSV4 dimensions and ordinary MoE dataflow, then run the full constrained Module through parser, graph, candidates, provider, solver, materializer, and report.
- [ ] step 8.2 Assert shared/routed overlap on disjoint submeshes, total CTA use <= 132, parallel cost below forced serial, call-site sharing, reshards, constraints removal, and full Module round-trip.
- [ ] step 8.3 Remove obsolete toy schedule files, compatibility paths, target-specific core branches, and abstractions not required by this contract.

#### Acceptance Criteria
- [ ] AC-8-1: The real DSV4 model has `DIM=4096`, `N_ROUTED=256`, `N_ACT=6`, and `MOE_INTER=2048`, with top-k routing, routed FP4 gather/dequant, shared FP8 dequant, and pre-MoE RMSNorm.
- [ ] AC-8-2: No Tiny/Real variants or forbidden fused semantic shortcut Ops exist in the DSV4 model.
- [ ] AC-8-3: The constrained full-graph solve satisfies hard constraints, overlaps shared/routed on disjoint submeshes, uses at most 132 CTAs, and beats forced serial cost.
- [ ] AC-8-4: The solution has no Agent metadata, prints/parses/typeinfers/verifies, and the structured report and Markdown agree.
- [ ] AC-8-5: `pytest tests/schedule -q` and affected parser/IR/model tests pass; broader tests are run when touched imports justify them.
<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

## Scope Exclusions

- No TIR lowering, CUDA code generation, GPU execution, or CTA-internal storage/buffer/barrier/ring/instruction scheduling.
- No 2D or multidimensional CTA solving, tail CTAs, uneven local shapes, masked remainder work, or dynamic CTA extents.
- No recomputation, replication, structural fusion, AutoDist hint invention, or generic objective language.
- No new fused MoE/FP8/expert HIR Ops and no import of TileOpsGov at test runtime.
- Do not modify unrelated IR behavior or discard existing branch history; obsolete schedule code is removed only through normal new commits.

## Execution Preflight

<!-- policy_preflight:start -->

### Policy Rules & Knowledge

- Comment hygiene — Code comments only describe local logic; no plan / milestone / version / review / discussion narration. (see `docs/develop.md § Code comments`) <!-- policy_rules: comment_hygiene -->
- Scope discipline — One commit touches only what the current task requires; unrelated edits / submodule bumps / autoformat go in separate commits or are called out explicitly. (see `docs/develop.md § Scope`) <!-- policy_rules: scope_discipline -->
- Test discipline — Tests exercise intended behaviour; no excessive defensive / catch-all tests that lock implementation detail. (see `docs/develop.md § Tests`) <!-- policy_rules: test_discipline -->
- DSL/HIR authoring — Conventions for @func DSL fixtures: no docstring in @func body, tf.<op> attribute path, variadic op call surface, SSA reachability. (see `docs/develop.md § DSL / HIR authoring`) <!-- policy_knowledge: dsl_hir_authoring -->
<!-- policy_preflight:end -->
