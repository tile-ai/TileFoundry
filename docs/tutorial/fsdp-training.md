# Minimal FSDP and HSDP training in HIR

[`examples/fsdp_training/`](../../examples/fsdp_training/README.md) models two-block FP32 transformer training with explicit parameter sharding, separately authored forward and backward functions, and momentum SGD. Its GQA workload uses RMSNorm, RoPE, QK normalization, ReLU-squared MLPs, and causal cross-entropy. It executes through the HIR reference evaluator. The NVIDIA target declaration supplies storage ownership and allocation facts; the example does not compile CUDA or predict training time.

Install the repository and NVIDIA target in an editable development environment as described in [develop](../develop.md). Run:

```bash
python -m pytest tests/core/evaluator/test_fsdp_training.py tests/core/analysis/test_fsdp_training.py -q
python -m examples.fsdp_training.run --participants 1 2 4
python -m examples.fsdp_training.run --participants 4 --replicas 2
python -m examples.fsdp_training.run --participants 16 --replicas 4
python -m examples.fsdp_training.visualize --output-dir /tmp/fsdp-ownership
```

The tests isolate ownership, reduction and residency using a small MLP fixture, with an FSDP and an HSDP transformer smoke case per policy. The example runner checks two transformer training steps against independent dense autograd and PyTorch SGD, then prints one JSON record per residency policy: six FSDP records at one, two and four participants, plus two each for 2 × 2 and 4 × 4 HSDP. All reference inputs remain on CPU; participants are logical mesh positions simulated on the input tensors' existing device. Memory analysis can take several minutes per configuration. The final command exports SVG ownership diagrams from checked HIR without executing training.

## Training contract

The example has batch 8 (16 for 4 × 4 HSDP), sequence length 8, width 16, four query heads, two KV heads, MLP width 64, and vocabulary 32. FSDP uses one mesh axis for equal batches of complete sequences and dim-0 parameter shards; HSDP uses the two axes described below. The single-participant case keeps full parameters. Inputs deliberately differ between participants. Each block has six parameter tensors; embedding and head are untied. Stage boundaries group parameter lifetimes, while each parameter's redistribution remains explicit. Collective packing and physical launch grouping are not modeled.

Each participant forms gradients of its local mean cross-entropy. MatMul over the split token contraction produces `Partial(sum)`. An explicit `Reshard` reduce-scatters that value into the parameter layout, followed by exactly one division by the participant count. The resulting parameter gradients match the global mean loss. The scalar loss uses HIR `Reduce`, whose existing semantics complete a reduction across a Split axis; this is distinct from an unresolved MatMul Partial.

HSDP uses two logical axes, `(replicate, shard)`, over the same GPU topology level. Parameters and momentum carry `[Broadcast, Split(0)]`, while each axis splits the data batch. Parameter-gradient contractions produce `[Partial(sum), Partial(sum)]`. The first `Reshard` reduce-scatters along `shard` to `[Partial(sum), Split(0)]`; the second all-reduces along `replicate` to `[Broadcast, Split(0)]`. Only then does the program divide by the total participant count: four for 2 × 2 or sixteen for 4 × 4. The intermediate `Partial` state prevents a subgroup sum from being used as a completed global gradient.

Momentum uses learning rate 0.01, momentum 0.9, zero initial velocity, no dampening, no weight decay, and no Nesterov update. Parameters and velocity are ordinary function inputs and outputs. The second step feeds back the first step's returned state; weights are not loaded again as checkpoint constants. The oracle uses dense autograd and PyTorch's optimizer rather than the example's derivative or sharding helpers. Transformer comparisons use `rtol=5e-4`, `atol=2e-5`, and the runner reports observed errors for every output category. The [example README](../../examples/fsdp_training/README.md) documents its small-vocabulary one-hot embedding representation and the differences from modded-nanogpt.

## Residency is ordinary HIR dataflow

The release policy gathers inside each forward stage and returns saved activations plus the original parameter shards. Backward stages gather again where weights are needed. The retain policy returns gathered weights as additional forward results and passes them to backward. Embedding backward does not read its weights, so its forward gather is sufficient under either policy. Existing retained-call liveness and allocation express the distinction; no FSDP-specific lifetime operation, autodiff annotation, or second allocator is required.

For one-dimensional FSDP with two or four participants, each step has:

| Scoped quantity | Release/regather | Retain through backward |
|---|---:|---:|
| Parameter all-gather occurrences | 27 | 14 |
| Gradient reduce-scatter occurrences | 14 | 14 |
| Peak live gathered-weight storage per participant | 11,264 bytes | 24,576 bytes |
| Persistent parameters plus velocity | 53,248/P bytes | 53,248/P bytes |
| Sum of all-gather result sizes per participant | 51,200 bytes | 26,624 bytes |
| Sum of reduce-scatter input sizes per participant | 26,624 bytes | 26,624 bytes |

These counts come from reached call occurrences, including retained callees. For the gathered-weight peak, the runner follows gather provenance through call/tuple results, selects the relevant windows from the existing allocation summary, and deduplicates storage aliases. The small escaping-view regression additionally checks that a 128-byte view keeps its entire 256-byte gathered backing allocation reserved while another stage executes.

The 2 × 2 HSDP case retains those gather counts and peaks, adds fourteen gradient all-reduces, and stores 26 KiB of persistent parameters plus momentum per participant. All-reduce inputs total 13 KiB per participant because the preceding reduce-scatter has already halved each gradient. The report derives payloads from local logical shapes, keeping unchanged Split axes in account, and records the collective's axis, group size and disjoint group count. The [example README](../../examples/fsdp_training/README.md#hsdp-two-shard-groups-with-replicated-parameter-shards) gives both tables and their derivation.

For 4 × 4 HSDP, each collective has four disjoint groups of size four. Four-way sharding reduces all-reduce inputs to 6.5 KiB and persistent parameters plus momentum to 13 KiB per participant. Gather counts, gather-result and reduce-scatter-input payloads, and gathered-weight peaks stay the same.

The table states selected components, not total training memory. Activations, gradients, scalar loss, functional optimizer outputs, caller-owned inputs, and other temporaries also occupy storage. Inputs are persistent because the function has no permission to reclaim caller-owned storage. Logical collective input/output sizes are not bytes sent on a network link: they do not specify a ring, tree, NVLink/NVSwitch route, retransmission, or overlap. There is no hardware cost model or throughput claim. With one participant, interparticipant redistribution counts are zero and a separate gathered-weight peak is reported as `null`, since local aliases need no extra full-weight materialization.

## Supported reference semantics

`Mesh`, `ShardLayout`, `Split`, `Broadcast` and `Partial` remain the distribution model. Single-level GPU mesh regions and sharded operands select participation automatically, with concrete tensor shapes and even shards. Specialize symbolic GPU tensor shapes first. Existing CTA/warp/thread reference evaluation keeps its global logical view; hierarchical GPU-to-CTA execution is deferred. Local pieces use the storage-independent coordinate projection shared with checkpoint projection and physical layout construction. Mesh coordinates, fresh call/loop scopes, explicit gather/scatter, and sum reductions operate on actual participant values. A permuted 2 × 2 mesh regression checks each HSDP reduction stage and refuses the unresolved intermediate output.

Output reconstruction is observation, not communication inserted into the program. Split pieces reconstruct the declared logical output. Plain/Broadcast replicas must agree, and unresolved Partial values cannot be returned as completed tensors. Supported transitions are unchanged ownership, Broadcast-to-Split, Split-to-Broadcast, and Partial(sum)-to-Split/Broadcast. Dynamic placements, changing participant meshes, general Split-to-Split redistribution, and participant-local computation on unresolved Partial inputs are outside this initial evaluator subset and must be resolved or diagnosed.

This is a minimal FSDP2-style training model, not the complete PyTorch FSDP2 runtime. Uneven shards, physical multi-node binding, collective packing, mixed precision, offload, gradient accumulation, communication/computation overlap, prefetch, autograd integration, checkpoint policy, inference, and code generation remain follow-up work. The HSDP rows can later map to nodes, with different transport choices within and across rows; those choices do not change the logical reduction equations. No calibrated network model is required for these semantics.

## Logical composition and physical realization

`project_domain` constructs a coordinate transform without storage or target information. A partially fixed projection can be composed with another by applying its transform to the previous index map. The evaluator uses canonical global tensor indices; `local_layout` applies the same transform to the authored physical map, preserving its strides and composition. Storage-owner compaction remains in `ShardLayout.from_modes`.

Relation propagation similarly exposes `derive_output_distribution`: it computes the participating mesh and per-axis Split/Broadcast/Partial result before choosing physical strides. `derive_output_shard_layout` then preserves a compatible physical carrier or constructs a layout for the selected storage. This is a composable derivation for stated layouts and supported affine access relations, not automatic placement search. `Dynamic` inference and cost-guided selection remain outside it.

Identical index coverage does not make Partial and Broadcast interchangeable. A plain layout also does not prove participant values agree: output reconstruction checks that equality. These distinctions remain relevant to [#278](https://github.com/tile-ai/TileFoundry/issues/278); this refactor does not implement its broad canonicalization or invent a holder mesh from execution scope.

## Related design work

- [Discussion #181](https://github.com/tile-ai/TileFoundry/discussions/181): participant reference evaluation, Partial values and logical redistribution.
- [Discussion #183](https://github.com/tile-ai/TileFoundry/discussions/183): HIR-first distributed modeling and cross-device ownership.
- [#255](https://github.com/tile-ai/TileFoundry/issues/255), originally [#138](https://github.com/tile-ai/TileFoundry/issues/138): the evaluator work item; this example covers a training subset.
- [#278](https://github.com/tile-ai/TileFoundry/issues/278): broader Broadcast canonicalization remains separate.
- [#260](https://github.com/tile-ai/TileFoundry/pull/260), [#162](https://github.com/tile-ai/TileFoundry/pull/162), [#165](https://github.com/tile-ai/TileFoundry/pull/165), and [#170](https://github.com/tile-ai/TileFoundry/pull/170): Partial alias handling, mesh/layout slicing, checkpoint projection, and structured lifetimes used here.
