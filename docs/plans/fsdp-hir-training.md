---
type: FEAT
component: evaluator
target_repo: tilefoundry
---

# [FEAT][evaluator] Model minimal FSDP training in HIR

## Description

Execute static GPU mesh participants and explicit redistributions in the HIR reference evaluator, then model two steps of fully sharded training with separately authored forward and backward functions. Compare release/regather and retain-through-backward parameter residency using existing structured liveness and allocation. This is a reference semantics contribution, not an FSDP runtime or a network performance prediction.

The implementation base is upstream commit `3496c57ca5768cae3ff2410feb174e8ed2cb306e` (#279).

Related work: [discussion #181](https://github.com/tile-ai/TileFoundry/discussions/181), [discussion #183](https://github.com/tile-ai/TileFoundry/discussions/183), the evaluator item in [#255](https://github.com/tile-ai/TileFoundry/issues/255) (originally [#138](https://github.com/tile-ai/TileFoundry/issues/138)), and [#278](https://github.com/tile-ai/TileFoundry/issues/278). Build on the Partial alias correction in [#260](https://github.com/tile-ai/TileFoundry/pull/260), local slicing from [#162](https://github.com/tile-ai/TileFoundry/pull/162) and [#165](https://github.com/tile-ai/TileFoundry/pull/165), and structured lifetimes from [#170](https://github.com/tile-ai/TileFoundry/pull/170). These references identify dependencies and partial coverage, not claims to close their whole scope.

### Current state

These observations describe the implementation base above.

- `src/tilefoundry/evaluator/interpreter.py:93` binds a MeshRegion in one environment without executing its participants.
- `src/tilefoundry/ir/hir/sharding/reshard.py:175` passes the same data through Reshard; `sharding/mesh_coord.py:76` and `sharding/local.py:47` refuse participant-dependent execution.
- `src/tilefoundry/runtime/tensor.py:31` uses `local_layout_and_offset` to project a checkpoint tensor through the canonical mesh/layout algebra.
- `src/tilefoundry/analysis/liveness.py:93` builds structured definition/use intervals; `src/tilefoundry/analysis/allocation.py:424` remaps returned aliases to caller arguments.
- `tests/nvidia/cuda/fixtures/placed/tp_all_to_all.py:1` documents ownership-blind aliasing, despite the storage-instance requirement in `docs/spec/analysis.md`.

### Decisions

- D1 How to deliver -- Use one PR with separable milestone commits, because the author requested an implementation timeline and authorized this training-focused work. Corrective changes stay separate from the example and documentation; this plan records the milestones in the repository format.
- D2 How to represent local contributions -- Keep Mesh, ShardLayout, Split, Broadcast and Partial as the public distribution model and add an internal participant-value carrier, because distinct local contributions cannot be represented by a single dense tensor. Reuse canonical local projection, including layout-domain Split axes.
- D3 How to execute collectives -- Execute participants together at each HIR operation on the caller's existing torch device, because collectives need actual local inputs without a distributed runtime or a user-facing distributed flag. Fresh function and loop memo scopes remain the execution boundaries.
- D4 Which transitions to support -- Initially support static even shards, no-op, Broadcast-to-Split, Split-to-Broadcast, and Partial(sum)-to-Split/Broadcast, because these suffice for the minimal training cycle. Reject unresolved Partial at observation, Dynamic, incompatible meshes, and unsupported redistribution explicitly. Plain layouts do not prove participant invariance; observing a plain/Broadcast value requires agreement.
- D5 How to express residency -- Use retained HIR function calls as sequential stages, because existing call boundaries and allocation already model temporary versus escaped results. Release stages return saved activations; retain stages additionally return the gathered weight. Backward explicitly regathers or consumes that returned weight. No FSDP-specific operation or allocator is introduced.
- D6 Which workload to model -- Use X[8,8], W1[8,8], ReLU, W2[8,4], target[8,4], FP32, equal batches, dim-0 parameter shards and momentum SGD, because this exercises both parameter groups and backward reuse without unrelated training features. Compute dX and both parameter gradients. Reduce local-mean gradients with sum and divide by participant count exactly once. Pass updated weights and velocity as ordinary inputs/outputs for two steps.
- D7 How to validate -- Test participant counts 1, 2 and 4 against independent dense torch autograd and torch.optim.SGD plus a 2x2 subgroup test, because numerical agreement and structural bytes/event counts can prove this reference design without calibrated bandwidth. Defer topology cost models, codegen, automatic backward, inference, overlap and full HSDP.

- D8 Whether another lifetime mechanism is needed -- Keep the existing retained-call allocation machinery, because the stage probe and an escaping-view regression preserve distinct materializations and complete backing allocations. M4 therefore fixes GPU ownership alias proofs, while M6 reads existing windows and accepted aliases for the scoped residency report.
- D9 How to evaluate the scalar loss -- Preserve Reduce's existing complete-reduction contract, because its inferred result is Broadcast while MatMul contraction produces Partial(sum). A separate M5 corrective commit implements complete reference reductions and tests sum/mean over split and unsplit axes; the example does not invent a new backward or reduction type rule.
- D10 Which byte quantities to publish -- Report logical collective input/output sizes, because an unscheduled HIR operation does not determine a network algorithm, routing, or achieved bandwidth. These sizes are distinct from the allocator's full-weight peak and from network traffic.
- D11 Which execution boundary to extend -- Enumerate single-level GPU meshes with concrete tensor shapes, because FSDP device ownership differs from the existing global reference semantics of CTA/warp/thread programs. Preserve invariant values, treat mesh names as labels, and use local reshape extents. Full-suite comparison exposed this compatibility boundary; hierarchical GPU-to-CTA execution and symbolic GPU shapes remain follow-up work.

## Milestones

### Milestone M0: Record the contract

#### Depends
- None

#### Target State Design

##### Delivered
```diff
# docs/plans/fsdp-hir-training.md
+Record scope, decisions, milestones and validation before implementation.
```

##### Accepted by
Existing plan finalizer and repository hooks; no new test.
- [x] The supported training subset and deferred runtime features are explicit.

<!-- policy_ac:start -->
- [x] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
<!-- policy_ac:end -->

#### Related Files
- docs/plans/fsdp-hir-training.md

### Milestone M1: Share local tensor projection

#### Depends
- M0

#### Target State Design

##### Delivered
```diff
# src/tilefoundry/runtime/tensor.py
+Share local view construction using local_layout_and_offset.
+Project canonical logical inputs independently of device-local physical strides.
```

##### Accepted by
Extend existing local-layout/runtime tests only where canonical logical inputs differ from physical checkpoint views.
- [x] Tagged rows and columns select the correct data, including permuted mesh axes.
- [x] Existing physical-view and offset behavior remains valid.

<!-- policy_ac:start -->
- [x] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [x] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- src/tilefoundry/runtime/tensor.py
- tests/core/ir/types/test_local_layout.py
- docs/spec/runtime.md

### Milestone M2: Execute static mesh participants

#### Depends
- M1

#### Target State Design

##### Delivered
```diff
# src/tilefoundry/evaluator/interpreter.py
-Evaluate MeshRegion once without coordinates.
+Evaluate operations over participant values, preserving call/loop scopes.
# src/tilefoundry/evaluator/mesh.py
+Carry actual local values with their existing global HIR type.
```

##### Accepted by
Extend evaluator cases with rank-dependent arithmetic, repeated calls and loop carries; these catch incorrectly shared rank-zero values.
- [x] MeshCoord and Local use the selected logical coordinate.
- [x] Output reconstruction checks replicas rather than silently choosing rank zero.
- [x] Unsupported participation is diagnosed.

<!-- policy_ac:start -->
- [x] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [x] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- src/tilefoundry/evaluator
- src/tilefoundry/ir/hir/sharding/local.py
- src/tilefoundry/ir/hir/sharding/mesh_coord.py
- tests/core/evaluator
- docs/spec/evaluator.md

### Milestone M3: Resolve explicit redistributions

#### Depends
- M2

#### Target State Design

##### Delivered
```diff
# src/tilefoundry/ir/hir/sharding/reshard.py
-Return the operand's data for every distribution transition.
+Resolve supported transitions from actual participant-local values.
```

##### Accepted by
Tagged gather/scatter, unequal sum contributions, and 2x2 subgroup tests; failure cases exercise unresolved Partial and unsupported transitions.
- [x] Contributions [1,2,3,4] and [10,20,30,40] reduce-scatter to [11,22] and [33,44].
- [x] Reduction never mixes distinct subgroups or counts Broadcast copies twice.
- [x] No-op preserves local data; observations do not insert program communication.

<!-- policy_ac:start -->
- [x] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [x] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- src/tilefoundry/evaluator
- src/tilefoundry/ir/hir/sharding/reshard.py
- tests/core/evaluator
- docs/spec/evaluator.md

### Milestone M4: Preserve materialization lifetimes

#### Depends
- M3

#### Target State Design

##### Delivered
```diff
# src/tilefoundry/ir/hir/sharding/reshard.py
+Reject aliases that change independent device ownership.
# tests/core/analysis
+Exercise gathered temporaries and escaped results across retained calls.
```

##### Accepted by
Use existing allocation/liveness interfaces to verify storage identities and lifetimes, extending them only if the stage probe demonstrates a missing contract.
- [x] Release and regather create separate materializations.
- [x] Returning a full weight or its alias keeps its storage alive until backward.
- [x] Identity redistributions still alias.

<!-- policy_ac:start -->
- [x] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [x] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- src/tilefoundry/ir/hir/sharding/reshard.py
- src/tilefoundry/analysis
- tests/core/analysis
- tests/core/ir/test_partial_reshard.py
- docs/spec/analysis.md

### Milestone M5: Model a sharded training step

#### Depends
- M4

#### Target State Design

##### Delivered
```diff
# examples/fsdp_training.py
+Author explicit forward/backward stages and functional momentum SGD.
+Provide release/regather and retain-through-backward policies.
```

##### Accepted by
One parameterized workflow compares outputs, loss, gradients, weights and velocity over two steps to independent dense autograd and SGD.
- [x] Both policies match at participant counts 1, 2 and 4.
- [x] The second step consumes returned optimizer state and parameters.
- [x] dX normalization is checked against the global-mean loss.

<!-- policy_ac:start -->
- [x] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
<!-- policy_ac:end -->

#### Related Files
- examples/fsdp_training.py
- tests/core/evaluator/test_fsdp_training.py
- docs/spec/evaluator.md

### Milestone M6: Validate residency and logical transfers

#### Depends
- M5

#### Target State Design

##### Delivered
```diff
# tests/core/analysis
+Assert scoped parameter residency and redistribution occurrences.
```

##### Accepted by
Analyze both policies using actual HIR call occurrences and shared storage identities, not counters supplied by the example.
- [x] For multiple participants, release has four all-gathers, retain two, and both have two gradient reduce-scatters per step.
- [x] Full-weight-only peaks are 256 versus 384 bytes; persistent parameter-plus-velocity state is 768/P bytes per participant.
- [x] Single-participant execution has zero interparticipant traffic.
- [x] Scoped components are never presented as total allocator peaks or measured network traffic.

<!-- policy_ac:start -->
- [x] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
<!-- policy_ac:end -->

#### Related Files
- tests/core/analysis
- docs/spec/analysis.md

### Milestone M7: Document and verify the workflow

#### Depends
- M6

#### Target State Design

##### Delivered
```diff
# docs/tutorial/fsdp-training.md
+Document runnable checks, residency interpretation and unsupported features.
```

##### Accepted by
Run the documented workflow, touched suites, full suite against the baseline, and repository hooks. Record actual results in the PR, including environment failures.
- [x] A reviewer can reproduce correctness and the residency distinction without GPU codegen.
- [x] PR references distinguish partial issue coverage and related upstream work.

<!-- policy_ac:start -->
- [x] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
<!-- policy_ac:end -->

#### Related Files
- docs/tutorial/fsdp-training.md
- docs/spec/evaluator.md
- docs/spec/analysis.md

## Final Gate

<!-- final_gate:start -->
<!-- final_gate:end -->
