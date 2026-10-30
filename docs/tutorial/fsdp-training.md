# Minimal FSDP training in HIR

[`examples/fsdp_training.py`](../../examples/fsdp_training.py) models two-layer FP32 training with explicit parameter sharding, separately authored forward and backward functions, and momentum SGD. It executes through the HIR reference evaluator. The NVIDIA target declaration supplies storage ownership and allocation facts; the example does not compile CUDA or predict training time.

Install the repository and NVIDIA target in an editable development environment as described in [develop](../develop.md). Run:

```bash
python -m pytest tests/core/evaluator/test_fsdp_training.py tests/core/analysis/test_fsdp_training.py -q
python examples/fsdp_training.py --participants 2
```

The first command checks two training steps at one, two and four participants against independent dense `torch.autograd` and `torch.optim.SGD`. The second prints one JSON record per residency policy. All reference inputs can remain on CPU; participants are logical mesh positions simulated on the input tensors' existing device.

## Training contract

The network is `Y = relu(X @ W1) @ W2`, with `X[8,8]`, `W1[8,8]`, `W2[8,4]`, and targets `[8,4]`. Equal batch slices and dim-0 parameter shards use the same one-dimensional FSDP mesh. The reference data deliberately differs between ranks. There is one parameter per communication group, no bias, and no stochastic operation.

Each participant forms gradients of its local mean squared error. MatMul over the split batch contraction produces `Partial(sum)`. An explicit `Reshard` reduce-scatters that value into the parameter layout, followed by exactly one division by the participant count. The resulting parameter gradients match the global mean loss. Concatenated `dX` is still a local-mean derivative and is divided by the participant count for comparison to dense global-mean autograd. The scalar loss uses HIR `Reduce`, whose existing semantics complete a reduction across a Split axis; this is distinct from an unresolved MatMul Partial.

Momentum uses learning rate 0.01, momentum 0.9, zero initial velocity, no dampening, no weight decay, and no Nesterov update. Parameters and velocity are ordinary function inputs and outputs. The test's second step feeds back the first step's returned state; weights are not loaded again as checkpoint constants. The oracle uses dense autograd and PyTorch's optimizer rather than the example's derivative or sharding helpers. All comparisons use `rtol=1e-5`, `atol=1e-6`.

## Residency is ordinary HIR dataflow

The release policy gathers inside each forward stage and returns only activations. Backward stages gather again. The retain policy returns the gathered weight as an additional forward result and passes it to backward. Existing retained-call liveness and allocation express the distinction; no FSDP-specific lifetime operation, autodiff annotation, or second allocator is required.

For two or four participants, each step has:

| Scoped quantity | Release/regather | Retain through backward |
|---|---:|---:|
| Parameter all-gather occurrences | 4 | 2 |
| Gradient reduce-scatter occurrences | 2 | 2 |
| Full-weight peak per participant | 256 bytes | 384 bytes |
| Persistent parameters plus velocity | 768/P bytes | 768/P bytes |
| Sum of all-gather result sizes per participant | 768 bytes | 384 bytes |
| Sum of reduce-scatter input sizes per participant | 384 bytes | 384 bytes |

These counts come from reached call occurrences, including retained callees. The full-weight peak selects the relevant windows from the existing allocation summary and deduplicates storage aliases. The escaping-view regression additionally checks that a 128-byte view keeps its entire 256-byte gathered backing allocation reserved while another stage executes.

The table states selected components, not total training memory. Activations, gradients, scalar loss, functional optimizer outputs, caller-owned inputs, and other temporaries also occupy storage. Inputs are persistent because the function has no permission to reclaim caller-owned storage. Logical collective input/output sizes are not bytes sent on a network link: they do not specify a ring, tree, NVLink/NVSwitch route, retransmission, or overlap. There is no hardware cost model or throughput claim. With one participant, interparticipant redistribution counts are zero and a separate gathered-weight peak is reported as `null`, since local aliases need no extra full-weight materialization.

## Supported reference semantics

`Mesh`, `ShardLayout`, `Split`, `Broadcast` and `Partial` remain the distribution model. Single-level GPU mesh regions and sharded operands select participation automatically, with concrete tensor shapes and even shards. Specialize symbolic GPU tensor shapes first. Existing CTA/warp/thread reference evaluation keeps its global logical view; hierarchical GPU-to-CTA execution is deferred. Local pieces use the canonical layout-domain slicing algebra shared with checkpoint projection. Mesh coordinates, fresh call/loop scopes, explicit gather/scatter, and sum reductions operate on actual participant values. A 2x2 permuted-mesh test verifies subgroup isolation without claiming full HSDP support.

Output reconstruction is observation, not communication inserted into the program. Split pieces reconstruct the declared logical output. Plain/Broadcast replicas must agree, and unresolved Partial values cannot be returned as completed tensors. Supported transitions are unchanged ownership, Broadcast-to-Split, Split-to-Broadcast, and Partial(sum)-to-Split/Broadcast. Dynamic placements, changing participant meshes, general Split-to-Split redistribution, and participant-local computation on unresolved Partial inputs are outside this initial evaluator subset and must be resolved or diagnosed.

This is a minimal FSDP2-style training model, not the complete PyTorch FSDP2 runtime. Uneven shards, full HSDP meshes, grouping multiple parameters, mixed precision, offload, gradient accumulation, communication/computation overlap, prefetch, autograd integration, checkpoint policy, inference, and code generation remain follow-up work. The first next extension should build on these executable semantics and a concrete workload, rather than add a calibrated network model before scheduling exists.

## Related design work

- [Discussion #181](https://github.com/tile-ai/TileFoundry/discussions/181): participant reference evaluation, Partial values and logical redistribution.
- [Discussion #183](https://github.com/tile-ai/TileFoundry/discussions/183): HIR-first distributed modeling and cross-device ownership.
- [#255](https://github.com/tile-ai/TileFoundry/issues/255), originally [#138](https://github.com/tile-ai/TileFoundry/issues/138): the evaluator work item; this example covers a training subset.
- [#278](https://github.com/tile-ai/TileFoundry/issues/278): broader Broadcast canonicalization remains separate.
- [#260](https://github.com/tile-ai/TileFoundry/pull/260), [#162](https://github.com/tile-ai/TileFoundry/pull/162), [#165](https://github.com/tile-ai/TileFoundry/pull/165), and [#170](https://github.com/tile-ai/TileFoundry/pull/170): Partial alias handling, mesh/layout slicing, checkpoint projection, and structured lifetimes used here.
