# Minimal FSDP training in HIR

[`examples/fsdp_training/`](../../examples/fsdp_training/README.md) models two-block FP32 transformer training with explicit parameter sharding, separately authored forward and backward functions, and momentum SGD. Its GQA workload uses RMSNorm, RoPE, QK normalization, ReLU-squared MLPs, and causal cross-entropy. It executes through the HIR reference evaluator. The NVIDIA target declaration supplies storage ownership and allocation facts; the example does not compile CUDA or predict training time.

Install the repository and NVIDIA target in an editable development environment as described in [develop](../develop.md). Run:

```bash
python -m pytest tests/core/evaluator/test_fsdp_training.py tests/core/analysis/test_fsdp_training.py -q
python -m examples.fsdp_training.run --participants 1 2 4
```

The tests isolate ownership, reduction and residency using a small MLP fixture, with one two-participant transformer smoke case per policy. The example runner checks two transformer training steps at one, two and four participants against independent dense autograd and PyTorch SGD, then prints one JSON record per residency policy. All reference inputs remain on CPU; participants are logical mesh positions simulated on the input tensors' existing device. Memory analysis can take several minutes per configuration.

## Training contract

The example has batch 8, sequence length 8, width 16, four query heads, two KV heads, MLP width 64, and vocabulary 32. Equal batches of complete sequences and dim-0 parameter shards use the same one-dimensional FSDP mesh. Inputs deliberately differ between participants. Each block has six parameter tensors; embedding and head are untied. Stage boundaries group parameter lifetimes, while each parameter's redistribution remains explicit. Collective packing and physical launch grouping are not modeled.

Each participant forms gradients of its local mean cross-entropy. MatMul over the split token contraction produces `Partial(sum)`. An explicit `Reshard` reduce-scatters that value into the parameter layout, followed by exactly one division by the participant count. The resulting parameter gradients match the global mean loss. The scalar loss uses HIR `Reduce`, whose existing semantics complete a reduction across a Split axis; this is distinct from an unresolved MatMul Partial.

Momentum uses learning rate 0.01, momentum 0.9, zero initial velocity, no dampening, no weight decay, and no Nesterov update. Parameters and velocity are ordinary function inputs and outputs. The second step feeds back the first step's returned state; weights are not loaded again as checkpoint constants. The oracle uses dense autograd and PyTorch's optimizer rather than the example's derivative or sharding helpers. Transformer comparisons use `rtol=5e-4`, `atol=2e-5`, and the runner reports observed errors for every output category. The [example README](../../examples/fsdp_training/README.md) documents its small-vocabulary one-hot embedding representation and the differences from modded-nanogpt.

## Residency is ordinary HIR dataflow

The release policy gathers inside each forward stage and returns saved activations plus the original parameter shards. Backward stages gather again where weights are needed. The retain policy returns gathered weights as additional forward results and passes them to backward. Embedding backward does not read its weights, so its forward gather is sufficient under either policy. Existing retained-call liveness and allocation express the distinction; no FSDP-specific lifetime operation, autodiff annotation, or second allocator is required.

For two or four participants, each step has:

| Scoped quantity | Release/regather | Retain through backward |
|---|---:|---:|
| Parameter all-gather occurrences | 27 | 14 |
| Gradient reduce-scatter occurrences | 14 | 14 |
| Peak live gathered-weight storage per participant | 11,264 bytes | 24,576 bytes |
| Persistent parameters plus velocity | 53,248/P bytes | 53,248/P bytes |
| Sum of all-gather result sizes per participant | 51,200 bytes | 26,624 bytes |
| Sum of reduce-scatter input sizes per participant | 26,624 bytes | 26,624 bytes |

These counts come from reached call occurrences, including retained callees. The runner additionally prints the full-weight peak. It follows gather provenance through call/tuple results, selects the relevant windows from the existing allocation summary, and deduplicates storage aliases. The small escaping-view regression additionally checks that a 128-byte view keeps its entire 256-byte gathered backing allocation reserved while another stage executes.

The table states selected components, not total training memory. Activations, gradients, scalar loss, functional optimizer outputs, caller-owned inputs, and other temporaries also occupy storage. Inputs are persistent because the function has no permission to reclaim caller-owned storage. Logical collective input/output sizes are not bytes sent on a network link: they do not specify a ring, tree, NVLink/NVSwitch route, retransmission, or overlap. There is no hardware cost model or throughput claim. With one participant, interparticipant redistribution counts are zero and a separate gathered-weight peak is reported as `null`, since local aliases need no extra full-weight materialization.

## Supported reference semantics

`Mesh`, `ShardLayout`, `Split`, `Broadcast` and `Partial` remain the distribution model. Single-level GPU mesh regions and sharded operands select participation automatically, with concrete tensor shapes and even shards. Specialize symbolic GPU tensor shapes first. Existing CTA/warp/thread reference evaluation keeps its global logical view; hierarchical GPU-to-CTA execution is deferred. Local pieces use the canonical layout-domain slicing algebra shared with checkpoint projection. Mesh coordinates, fresh call/loop scopes, explicit gather/scatter, and sum reductions operate on actual participant values. A 2x2 permuted-mesh test verifies subgroup isolation without claiming full HSDP support.

Output reconstruction is observation, not communication inserted into the program. Split pieces reconstruct the declared logical output. Plain/Broadcast replicas must agree, and unresolved Partial values cannot be returned as completed tensors. Supported transitions are unchanged ownership, Broadcast-to-Split, Split-to-Broadcast, and Partial(sum)-to-Split/Broadcast. Dynamic placements, changing participant meshes, general Split-to-Split redistribution, and participant-local computation on unresolved Partial inputs are outside this initial evaluator subset and must be resolved or diagnosed.

This is a minimal FSDP2-style training model, not the complete PyTorch FSDP2 runtime. Uneven shards, full HSDP meshes, collective packing, mixed precision, offload, gradient accumulation, communication/computation overlap, prefetch, autograd integration, checkpoint policy, inference, and code generation remain follow-up work. The first next extension should build on these executable semantics and a concrete workload, rather than add a calibrated network model before scheduling exists.

## Upstream analysis boundaries

Lowering's shared emission decisions from [#280](https://github.com/tile-ai/TileFoundry/pull/280) and [#282](https://github.com/tile-ai/TileFoundry/pull/282) consume the existing alias proof: a GPU gather remains an instruction while an unchanged distribution remains a view. Future scheduled lowerings should use these shared decisions for copies and issuing meshes. Their treatment of Partial as Broadcast for instruction ownership does not resolve a numerical Partial contribution; the reference evaluator still requires explicit reduction.

The opt-in `sync` analysis from [#283](https://github.com/tile-ai/TileFoundry/pull/283) plans ordering for shared-memory accesses inside a device. Local compatibility probes of the small GMEM-only MLP fixture produce an empty sync plan at 1, 2 and 4 participants and retain the same memory report under both policies. This checks compatibility, not collective completion or generated-kernel correctness. Retained callees holding shared memory are currently refused by that analysis; adding staged kernels requires addressing that boundary first. Its `owned_bytes` helper describes placed shared-memory address ranges and does not replace the lifetime windows used for this example's residency peak.

The example continues to request memory analysis only. No completion declarations, async collective tokens, synchronization lowering, or hardware cost model are added here.

## Related design work

- [Discussion #181](https://github.com/tile-ai/TileFoundry/discussions/181): participant reference evaluation, Partial values and logical redistribution.
- [Discussion #183](https://github.com/tile-ai/TileFoundry/discussions/183): HIR-first distributed modeling and cross-device ownership.
- [#255](https://github.com/tile-ai/TileFoundry/issues/255), originally [#138](https://github.com/tile-ai/TileFoundry/issues/138): the evaluator work item; this example covers a training subset.
- [#278](https://github.com/tile-ai/TileFoundry/issues/278): broader Broadcast canonicalization remains separate.
- [#260](https://github.com/tile-ai/TileFoundry/pull/260), [#162](https://github.com/tile-ai/TileFoundry/pull/162), [#165](https://github.com/tile-ai/TileFoundry/pull/165), and [#170](https://github.com/tile-ai/TileFoundry/pull/170): Partial alias handling, mesh/layout slicing, checkpoint projection, and structured lifetimes used here.
