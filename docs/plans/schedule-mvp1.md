---
type: FEAT
component: schedule MVP
target_repo: tilefoundry
---

# [FEAT][schedule] Main-based schedule MVP

## Description

### Symptom / Motivation

The main branch has HIR parsing, sharding, and evaluation but no inspectable
schedule pipeline. The MVP needs one small end-to-end path from authored HIR to
a deterministic CTA schedule and value-semantic materialized HIR.

### Root Cause Analysis

N/A for a new package. The implementation must establish explicit immutable
layers instead of coupling graph facts, target choices, costs, solver state, or
materialized output.

### Related Files

- `src/tilefoundry/__init__.py`
- `src/tilefoundry/parser/__init__.py`
- `src/tilefoundry/parser/hir_parser.py`
- `src/tilefoundry/schedule/__init__.py`
- `src/tilefoundry/schedule/input.py`
- `src/tilefoundry/schedule/constraints.py`
- `src/tilefoundry/schedule/graph.py`
- `src/tilefoundry/schedule/space.py`
- `src/tilefoundry/schedule/cost.py`
- `src/tilefoundry/schedule/solver.py`
- `src/tilefoundry/schedule/solution.py`
- `src/tilefoundry/schedule/registry.py`
- `src/tilefoundry/schedule/pipeline.py`
- `src/tilefoundry/schedule/builders/__init__.py`
- `src/tilefoundry/schedule/builders/function_calls.py`
- `src/tilefoundry/schedule/cuda/__init__.py`
- `src/tilefoundry/schedule/cuda/backend.py`
- `src/tilefoundry/schedule/cuda/space.py`
- `src/tilefoundry/schedule/cuda/cost.py`
- `src/tilefoundry/schedule/cuda/solver.py`
- `src/tilefoundry/schedule/cuda/materialize.py`
- `src/tilefoundry/ir/types/shard/mesh.py`
- `src/tilefoundry/ir/types/shard/scope_match.py`
- `src/tilefoundry/ir/hir/math/binary.py`
- `src/tilefoundry/ir/hir/sharding/reshard.py`
- `src/tilefoundry/ir/hir/sharding/local.py`
- `src/tilefoundry/ir/hir/tensor/reshape.py`
- `src/tilefoundry/ir/types/shard/shard_layout.py`
- `src/tilefoundry/ir/types/utils.py`
- `src/tilefoundry/visitor_registry/shard_propagate.py`
- `tests/schedule/__init__.py`
- `tests/schedule/fixtures/__init__.py`
- `tests/schedule/fixtures/dsv4_moe_mvp.py`
- `tests/schedule/test_mvp.py`

## Goal

Implement a deterministic main-based CTA schedule MVP whose immutable layers
are inspectable as `ScheduleInput -> ScheduleGraph -> ScheduleSpace ->
CostTable -> SolveProblem -> ScheduleSolution -> hir.Function`, with parser,
solver, materialization, and CPU-evaluator acceptance coverage.

## Constraints

- Keep one shared HIR parser implementation with explicit `parse_func` and `parse_schedule_func` entry points.
- Definition-local `where(storage=...)` is the only schedule annotation; constraints remain immutable side data and target HIR object identity.
- Graph records contain logical facts only and graph-local opaque IDs; no CUDA constants, timing, solver objects, or materialized mesh slices.
- Core schedule modules must not import `CudaTarget`, OR-Tools, or `src/tilefoundry/schedule/cuda`; target implementations depend on core interfaces.
- Schedule space is finite and target-neutral; CUDA CTA implementations own placement, roofline costs, exact enumeration, and materialization.
- Solving never mutates graph, space, or costs. Fingerprints cover all solve semantics, including finite choices and costs.
- MVP scheduling is static one-dimensional CTA, finite contiguous slices, and spatial-or-time exclusion. Unsupported levels and dynamic or multidimensional placement fail clearly.
- Materialization creates a new HIR function, uses ordinary composed function calls and explicit existing `Reshard` bridges, and preserves evaluator value semantics.
- The DSV4-shaped fixture is self-contained, CPU-sized, uses four ordinary function calls, one real storage constraint, and the registered real CUDA CTA backend.
- Do not modify the main checkout or the abandoned reference worktree, merge or cherry-pick the schedule-mvp0 branch, or add GPU code generation.

## Milestones

### Milestone M0: Input and logical graph

#### Depends
- None

#### Related Files
- `src/tilefoundry/parser/__init__.py`
- `src/tilefoundry/parser/hir_parser.py`
- `src/tilefoundry/schedule/input.py`
- `src/tilefoundry/schedule/constraints.py`
- `src/tilefoundry/schedule/graph.py`
- `src/tilefoundry/schedule/builders/function_calls.py`
- `src/tilefoundry/__init__.py`
- `tests/schedule/fixtures/dsv4_moe_mvp.py`
- `tests/schedule/test_mvp.py`

#### Plan
- [ ] step 0.1 Add immutable `ScheduleInput`, `ConstraintList`, `StorageConstraint`, source location, and author provenance records.
- [ ] step 0.2 Extend the existing HIR visitor with schedule-mode `where(storage=...)` assignment handling while keeping ordinary parsing fail-closed and shared.
- [ ] step 0.3 Build the admitted entry-body call graph with opaque graph-local node/value/use IDs, identity-bound constraints, and the four-node diamond topology.

#### Acceptance Criteria
- [ ] AC-0-1: `parse_func(...)` returns a normal `hir.Function`; `parse_schedule_func(...)` returns a real frozen `ScheduleInput` with source-ordered immutable constraints.
- [ ] AC-0-2: The exact `x_r: where(storage="rmem") = producer(...)` form creates the same HIR expression as an unannotated assignment plus one `AUTHOR` `StorageConstraint`, without an identity or schedule HIR node.
- [ ] AC-0-3: Empty/unknown/positional/invalid/non-tensor/duplicate `where` forms and `require(...)` fail with clear errors; ordinary `parse_func` directs annotated callers to `parse_schedule_func`.
- [ ] AC-0-4: The graph inspects only the entry body, treats top-level calls to `hir.Function` as opaque nodes, preserves data dependencies, and binds every authored constraint to exactly one admitted value with exactly one consumer.
- [ ] AC-0-5: The DSV4-shaped fixture produces exactly four logical nodes named/equivalent to `route_func`, `routed_func`, `shared_func`, and `combine_func` with the expected diamond dependencies.

<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M1: Finite CTA space, costs, and exact solve

#### Depends
- M0

#### Related Files
- `src/tilefoundry/schedule/space.py`
- `src/tilefoundry/schedule/cost.py`
- `src/tilefoundry/schedule/solver.py`
- `src/tilefoundry/schedule/solution.py`
- `src/tilefoundry/schedule/registry.py`
- `src/tilefoundry/schedule/pipeline.py`
- `src/tilefoundry/schedule/cuda/backend.py`
- `src/tilefoundry/schedule/cuda/space.py`
- `src/tilefoundry/schedule/cuda/cost.py`
- `src/tilefoundry/schedule/cuda/solver.py`
- `tests/schedule/test_mvp.py`

#### Plan
- [ ] step 1.1 Define immutable target-neutral representation, placement, node option, edge option, resource, backend, cost, problem, and solution interfaces.
- [ ] step 1.2 Register and resolve the concrete `(CudaTarget, "cta")` backend without importing CUDA or OR-Tools from core modules.
- [ ] step 1.3 Build finite legal CTA slices and DIRECT/RESHARD edge choices, calculate finite roofline estimates, fingerprint the full solve semantics, and enumerate deterministic schedules with dependency and spatial-or-time exclusion.

#### Acceptance Criteria
- [ ] AC-1-1: `ScheduleSpace` contains finite legal structural choices only; graph, space, and cost records are immutable and contain no target solver objects or selected assignments.
- [ ] AC-1-2: `register_schedule_backend(CudaTarget, level="cta", backend=...)` and `resolve_schedule_backend(CudaTarget(...), level="cta")` resolve the real CUDA CTA backend; unsupported levels and dynamic/multidimensional placement fail clearly.
- [ ] AC-1-3: Every cost table entry used by the solve is finite, uses `max(bytes / effective_bandwidth, flops / effective_peak)`, and does not choose candidates or mutate logical records.
- [ ] AC-1-4: `SolveProblem` carries graph, space, costs, constraints, and a fingerprint over every solve-relevant semantic input; `ScheduleSolution` carries explicit opaque-ID node/edge choices, placements, start/end times, makespan, and the matching fingerprint.
- [ ] AC-1-5: The deterministic exact solver finds the true minimum for the four-node fixture, lets `shared` and `routed` overlap in time on disjoint CTA slices, and beats the forced-serial sum of selected node and edge durations.

<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

### Milestone M2: Materialization and behavioral acceptance

#### Depends
- M1

#### Related Files
- `src/tilefoundry/schedule/cuda/materialize.py`
- `src/tilefoundry/ir/types/shard/mesh.py`
- `src/tilefoundry/ir/types/shard/scope_match.py`
- `src/tilefoundry/ir/hir/math/binary.py`
- `src/tilefoundry/ir/hir/sharding/reshard.py`
- `src/tilefoundry/ir/hir/sharding/local.py`
- `src/tilefoundry/ir/hir/tensor/reshape.py`
- `src/tilefoundry/ir/types/shard/shard_layout.py`
- `src/tilefoundry/ir/types/utils.py`
- `src/tilefoundry/visitor_registry/shard_propagate.py`
- `src/tilefoundry/schedule/pipeline.py`
- `tests/schedule/fixtures/dsv4_moe_mvp.py`
- `tests/schedule/test_mvp.py`

#### Plan
- [ ] step 2.1 Materialize solved placements as sliced `Mesh` values in `ShardLayout.mesh`, retaining ordinary composed calls and inserting explicit `Reshard` only at representation/mesh boundaries.
- [ ] step 2.2 Fix only the minimum ComposedLayout shape access and cross-slice type-inference behavior required by the scheduled fixture while preserving unsliced behavior.
- [ ] step 2.3 Add the self-contained CPU fixture and end-to-end assertions for graph shape, finite costs, constraint consumption, overlap, disjoint slices, fingerprint, verification/type inference, explicit bridges, cross-slice failure, and evaluator equivalence.

#### Acceptance Criteria
- [ ] AC-2-1: Materialization returns a new `hir.Function`, never mutates the input function, and creates no schedule-specific HIR operations or fused MoE operations.
- [ ] AC-2-2: Sliced mesh `axes`/`shape` access works through `ComposedLayout`; existing unsliced mesh behavior remains unchanged.
- [ ] AC-2-3: A value from one CTA slice cannot be directly combined with a value from another slice; type inference fails unless an explicit existing `Reshard` bridges the boundary, while full-parent `Broadcast` remains readable in child slices.
- [ ] AC-2-4: The materialized function passes HIR verification/type inference and CPU evaluation, and its output numerically equals evaluation of the original function on the fixture inputs.
- [ ] AC-2-5: The primary acceptance command `.venv/bin/pytest -q tests/schedule/test_mvp.py` passes, and focused parser/IR/IR-type tests plus Ruff pass on touched Python files.

<!-- policy_ac:start -->
- [ ] No touched C++/CUDA files in this milestone — clang-format gate N/A <!-- policy_ac: clang_format-na -->
<!-- policy_ac:end -->

## Execution Preflight

<!-- policy_preflight:start -->

### Policy Rules & Knowledge

- Comment hygiene — Code comments only describe local logic; no plan / milestone / version / review / discussion narration. (see `docs/develop.md § Code comments`) <!-- policy_rules: comment_hygiene -->
- Scope discipline — One commit touches only what the current task requires; unrelated edits / submodule bumps / autoformat go in separate commits or are called out explicitly. (see `docs/develop.md § Scope`) <!-- policy_rules: scope_discipline -->
- Test discipline — Tests exercise intended behaviour; no excessive defensive / catch-all tests that lock implementation detail. (see `docs/develop.md § Tests`) <!-- policy_rules: test_discipline -->
<!-- policy_preflight:end -->

## Scope Exclusions

- Do not implement parser syntax such as `with cta[0:32]`.
- Do not implement TIR lowering, MeshScope generation, CUDA code generation, compilation, or GPU execution.
- Do not implement device-level or multi-GPU scheduling, warp/thread/instruction scheduling, rings, async pipelines, tensor-core recipes, or ISL/polyhedral lifting.
- Do not implement a generic configurable objective language, fused MoE/FP8/expert HIR operations, broad deletion/refactoring of existing unusual ops, or a full port of the old schedule package.
- The generality in this MVP is only the clean boundary between logical graph, target-created finite space, target cost model, target solver, and target materializer.
