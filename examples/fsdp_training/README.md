# Two-block FSDP and HSDP training in HIR

This study asks how releasing gathered parameters after forward changes storage and logical communication while preserving the training update. The HIR evaluator simulates participants on CPU. Forward, backward, and momentum SGD are authored explicitly. PyTorch autograd supplies an independent numerical reference.

## Run the comparison

Use an editable TileFoundry installation with the NVIDIA target, following [the development guide](../../docs/develop.md). From the repository root:

```bash
python -m examples.fsdp_training.run --participants 1 2 4
python -m examples.fsdp_training.run --participants 4 --replicas 2
```

Each configuration performs two training steps on changing batches, checks logits, global mean loss, every parameter gradient, updated parameters, and momentum buffers, then analyzes memory. The first command prints six FSDP records; the second prints two HSDP records on a 2 × 2 logical mesh. Analysis can take several minutes per configuration. It needs neither a checkpoint nor a dataset download.

| File | Responsibility |
|---|---|
| `model.py` | Explicit forward/backward HIR and parameter residency policies |
| `reference.py` | Dense embedding, PyTorch SDPA/autograd, and optimizer oracle |
| `run.py` | Correctness checks and reports derived from existing HIR facts |
| `visualize.py` / `ownership.html` | Ownership trace exporter and its standalone interactive view |

## Explore tensor ownership

```bash
python -m examples.fsdp_training.visualize --output /tmp/hsdp-ownership.html
python -m examples.fsdp_training.visualize --participants 2 --replicas 1 --retain --output /tmp/fsdp-ownership.html
```

Open the generated HTML in a browser; it requires no server or external assets. The default is the four-participant, release-policy HSDP example. Choose any of the fourteen parameters to inspect its input, weight reshards, gradient reductions, and next optimizer state. The gradient is a distinct HIR value, not a mutation of the weight. Choose **All model tensor values** to inspect activations and other intermediates; the dependency links navigate to an operation's inputs and uses. Previous/next follows the selected inspection list, not a runtime schedule.

Each participant's band shows exact ownership ranges in the flattened global tensor. Displayed `Split` indices refer to logical tensor axes. `Partial` states are labeled separately because identical index coverage does not mean equal values. A `Reshard` reports the changed mesh axis and exact logical participant groups. Expand **Logical projection and physical storage** to compare local shape, logical offset, and storage offset in elements. The current head-weight input uses a dense address map: participant `(0, 1)` has logical and storage offsets of 256. A compact private address map could instead give storage offset zero without changing logical ownership; the explorer reports the map actually authored in the HIR. Local bytes describe one typed tensor value, not total memory or an allocation/lifetime estimate.

The exporter checks HIR types but does not execute training, prove replica equality, or run device communication. `trace_function()` follows reached function calls, mesh regions, and tuple leaves by SSA identity, keeping repeated calls separate; scalar literals are omitted from dependency navigation. `describe_tensor()` reuses `project_domain`, `logical_projection`, and the same logical selection as reference execution; applying that projection to the authored address map supplies physical offsets. Index tags describe coverage only and are never interpreted as numerical `Partial` contributions. Only `training_trace()` knows this model's parameter names and input/output signature. It adds convenient traces by following dependency edges and reading inferred types. The view supports the example's concrete, even-shard GPU meshes; loops and other region kinds are rejected explicitly. Plain layouts receive no invented holder mesh.

## Model and training contract

The workload borrows RMSNorm, QK normalization, rotary position encoding, and a ReLU-squared MLP from [modded-nanogpt's architectural direction](https://github.com/KellerJordan/modded-nanogpt/tree/4ea6b937337a4889b8cfe3f38a93d120048d8f71). GQA is an adaptation: this is not a reproduction of that repository's speedrun model or optimizer.

```text
tokens -> embedding -> block 1 -> block 2 -> RMSNorm -> vocabulary head -> cross-entropy
block  -> RMSNorm -> causal GQA -> residual -> RMSNorm -> ReLU² MLP -> residual
```

The fixed FP32 configuration has batch 8, sequence length 8, width 16, four query heads, two KV heads, head width 4, MLP width 64, and vocabulary 32. RMSNorm is affine-free, projections have no bias, embedding and head are untied, and there is no dropout. Random nonzero initialization makes every parameter's gradient nonzero on the first step. This avoids a zero output projection concealing an incorrect backward path.

HIR represents the tiny embedding lookup as one-hot matrix multiplication; the reference uses `torch.nn.functional.embedding`. Fixed rotation matrices express RoPE in HIR, while the reference rotates adjacent coordinate pairs. HIR broadcasts K/V over a separate query-group dimension; the reference repeats KV heads before SDPA. These are equivalent small-workload representations, not production embedding or attention kernels.

Every participant receives different complete sequences. Parameters and momentum are split along tensor dimension 0. Linear backward contracts the split token dimension to produce actual `Partial(sum)` contributions. FSDP resolves those with reduce-scatter; HSDP uses reduce-scatter and then all-reduce as described below. Exactly one division by the total participant count produces gradients of the global mean loss. HIR `Reduce` has complete reduction semantics, used for the scalar loss. The optimizer uses learning rate 0.01 and momentum 0.9, with no dampening, weight decay or Nesterov update. The second step consumes the first step's returned state; inputs are not mutated.

Checks use `rtol=5e-4`, `atol=2e-5` for FP32 reduction-order differences. The runner prints the observed maximum absolute errors separately for logits, loss, gradients, parameters and momentum.

## Residency experiment

Both variants use the same arithmetic and saved activations. The release variant gathers inside each forward stage, releases its full weights at the stage boundary, and re-gathers weights needed by backward. The retain variant returns full weights from forward and passes them to backward. Saved values are ordinary HIR tuples; there is no autodiff annotation, lifetime operation, or separate allocator.

Embedding backward does not read the embedding weights, so neither variant gathers them again. Each transformer block owns six parameter tensors. The head owns one. A stage boundary groups their lifetimes; it does not declare collective packing or an NCCL launch group.

For one-dimensional FSDP with multiple participants the per-step logical transfers are:

| Quantity | Release | Retain |
|---|---:|---:|
| Tensor all-gather occurrences | 27 | 14 |
| Tensor gradient reduce-scatter occurrences | 14 | 14 |
| Peak live gathered-weight storage per participant | 11,264 bytes (11 KiB) | 24,576 bytes (24 KiB) |
| All-gather result bytes per participant | 51,200 | 26,624 |
| Reduce-scatter input bytes per participant | 26,624 | 26,624 |
| Persistent parameters plus momentum per participant | 53,248 / P | 53,248 / P |

### How the table is generated

The command above generates all six records. For just the two-participant comparison, run `python -m examples.fsdp_training.run --participants 2`. Each record checks two numerical updates but reports the transfers and storage of one training-step graph, not the sum of two steps. `P` is the participant count and 1 KiB is 1,024 bytes.

In `run.py`, `study()` builds and checks each program, runs the independent training comparison, and calls `analyze(..., analysis="memory")`. `summarize_training()` then obtains the table from that analyzed HIR:

1. Walk reached function calls and inspect each mesh axis of `Reshard`: `Split -> Broadcast` is an all-gather, `Partial(sum) -> Split` is a reduce-scatter, and `Partial(sum) -> Broadcast` is an all-reduce. Each call occurrence is counted, so invoking a shared block definition twice counts both blocks. These graphs stage one changed axis per collective. Payload uses the participant-local shape from the shared logical projection, times dtype size: the all-gather result or the reduction input. Unchanged Split axes still reduce this local payload. The report records the changed axis, group size, and number of disjoint groups in `collective_groups`; occurrence counts are per participant, not multiplied by the number of groups.
2. Follow call and tuple results back to gathered values, then select their accepted allocation roots from `AllocationSummaryMetadata`. Use the lifetime windows in `RegionMemoryMetadata` to sum simultaneously live bytes, deduplicating aliases by root, and take the maximum over lifetime boundaries. Tensor shape alone does not identify a gathered weight.
3. Sum the allocation entries for the parameter and momentum inputs to obtain persistent state per participant. Fixed inputs, activations and returned update buffers are outside this selected quantity.

The following arithmetic independently explains the generated numbers. Every parameter is FP32, so each element occupies four bytes. Each block has six parameter tensors:

| Parameter | Shape | Full tensor bytes |
|---|---|---:|
| Embedding | `[32, 16]` | 2,048 |
| Q projection, per block | `[16, 16]` | 1,024 |
| K projection, per block | `[8, 16]` | 512 |
| V projection, per block | `[8, 16]` | 512 |
| Output projection, per block | `[16, 16]` | 1,024 |
| MLP up projection, per block | `[64, 16]` | 4,096 |
| MLP down projection, per block | `[16, 64]` | 4,096 |
| Vocabulary head | `[32, 16]` | 2,048 |

One block therefore holds 11,264 bytes (11 KiB). The fourteen tensors together hold `2 + 11 + 11 + 2 = 26 KiB` (26,624 bytes).

- **All-gathers:** forward gathers `1 + 6 + 6 + 1 = 14` tensors. Release re-gathers the two blocks and head during backward, adding `6 + 6 + 1 = 13`, for 27 total. Retain reuses those weights and stays at 14. Embedding backward needs the saved tokens and incoming gradient, not the embedding weights, so there is no embedding re-gather.
- **Reduce-scatters:** each of the fourteen parameter gradients has one `Partial(sum) -> Split` transition under either policy, giving 14.
- **Peak gathered-weight storage:** release keeps full weights within each stage, whose largest set is one 11 KiB block. Retain reaches a point where both blocks and the head are live: `11 + 11 + 2 = 24 KiB`. The embedding's gathered weights are already dead. These sums explain the peaks found by the allocation/lifetime analysis.
- **Collective payloads:** retain gathers 26 KiB of result tensors per participant. Release adds the 24 KiB needed by backward, giving 50 KiB (51,200 bytes). The fourteen reduce-scatter inputs total 26 KiB per participant under both policies. These quantities describe the full tensors entering or leaving logical collectives; they do not subtract a participant's existing shard or estimate network bytes.
- **Persistent state:** parameters occupy 26 KiB and momentum occupies another 26 KiB. Even sharding gives `52 KiB / P`: 52, 26 and 13 KiB per participant at `P = 1, 2, 4` respectively.

The multi-participant counts, full-tensor payloads and gathered-weight peaks stay the same at `P = 2` and `P = 4`; persistent shards get smaller. Single-participant execution reports zero interparticipant transfers and `null` for a separate gathered-weight peak, since full weights can alias their original storage.

At two participants, both policies matched the dense reference over two updates with maximum absolute errors of `5.1e-7` for logits, `1.2e-7` for gradients/momentum, and `3.0e-8` for updated parameters; scalar loss agreed at the reported FP32 precision. A separate local check against PyTorch FSDP2 on two sm90 GPUs also passed both policies, with maximum gradient error `3.9e-7`. That check used real sharded parameters, reduced gradients and sharded momentum, and does not establish distributed execution of the HIR or equality of physical communication schedules.

These are selected storage components, not total training memory: saved activations, gradients, functional update outputs, fixed inputs and other temporaries also occupy storage. Caller-owned inputs remain persistent. Logical collective payloads are not link traffic, achieved bandwidth, or elapsed time. The H200 target supplies storage ownership for this analysis; no H200 execution or speedup is claimed.

The result is a storage/communication trade-off. Retention removes re-gathers, while release permits smaller gathered-weight residency. Choosing between them needs a memory constraint or a future scheduled/runtime experiment.

## HSDP: two shard groups with replicated parameter shards

Run `python -m examples.fsdp_training.run --participants 4 --replicas 2`. The logical mesh axes are `(replicate=2, shard=2)`, with coordinates `(0,0), (0,1), (1,0), (1,1)`. Each row is a shard group; each column contains corresponding parameter shards across replicas. This follows the two-dimensional placement convention of [PyTorch FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html).

The distribution states are expressed using existing HIR attributes:

| Value or transition | Replicate axis | Shard axis |
|---|---|---|
| Tokens and activations | Split batch | Split batch |
| Parameters and momentum | Broadcast | Split parameter dimension 0 |
| Gathered parameters | Broadcast | Broadcast |
| Local parameter-gradient contributions | Partial(sum) | Partial(sum) |
| After reduce-scatter within each row | Partial(sum) | Split parameter dimension 0 |
| After all-reduce within each column | Broadcast | Split parameter dimension 0 |

The reduce-scatter must leave the replicate-axis contribution unresolved. Each column then all-reduces only its parameter shard. The program divides by four after both reductions, once. A small evaluator regression inspects the intermediate values from four distinct contributions and refuses to observe that still-Partial intermediate as a completed tensor. The transformer checks all gradients and both momentum updates against dense training.

The same reporter generates these differences from four-way FSDP:

| Per-step, per-participant quantity | FSDP, 4 shards | HSDP, 2 replicas × 2 shards |
|---|---:|---:|
| All-gathers, release / retain | 27 / 14 | 27 / 14 |
| Reduce-scatters | 14 | 14 |
| Gradient all-reduces | 0 | 14 |
| Reduce-scatter input | 26 KiB | 26 KiB |
| All-reduce input | 0 | 13 KiB |
| Persistent parameters plus momentum | 13 KiB | 26 KiB |
| Peak gathered weights, release / retain | 11 / 24 KiB | 11 / 24 KiB |

HSDP's all-gathers and reduce-scatters use two disjoint groups of size two on the `shard` axis. Its all-reduces use two groups of size two on the `replicate` axis. The fourteen full gradients total 26 KiB, so the all-reduce inputs after two-way reduce-scatter total `26 / 2 = 13 KiB` per participant. Parameters plus momentum occupy `52 / 2 = 26 KiB`; replication does not further divide that storage. Gather counts, result payloads and lifetime peaks follow the same stage arithmetic as FSDP above. No-op reshards do not count as transfers.

The mesh describes logical participant groups. A deployment could bind each row to one node and use intra-node gathers/reduce-scatters with inter-node all-reduces. The example does not assign nodes, links or routes, and it does not estimate their bandwidth. Logical projection and propagation determine indices and per-axis value state; physical realization chooses storage strides and owner-local allocation separately. A later communication lowering can consume these groups without changing the training equations.

A separate local check with real PyTorch HSDP on four sm90 GPUs passed two updates for both policies. It used a `(2, 2)` DeviceMesh and verified `Replicate(), Shard(0)` parameter and momentum placements, local shard contents, gradients, logits, and loss. The largest differences were `7.5e-7` for logits, `4.2e-7` for gradients/momentum, and `1.5e-8` for parameters. All four devices were on one host: this validates subgroup numerical behavior, not inter-node transport or HIR-generated distributed execution. The extra hardware comparison remains local; the repository keeps the semantic tests and runnable reference study.

## Scope

This models minimal FSDP2-style training semantics using existing mesh placements, including a 2 × 2 HSDP configuration. It supports a single GPU topology level, even shards, and concrete shapes. GQA supplies the workload; the parallelism demonstrated here is data parallel training with sharded parameters, gradients and optimizer state. Muon, mixed precision, accumulation, offload, prefetch/overlap, uneven shards, TP/CP, inference and distributed CUDA code generation remain follow-ups.

Small ownership, reduction and lifetime regressions live under `tests/core`; CI includes an FSDP and an HSDP transformer smoke case for each policy. The runner supplies the broader residency study. See [the HIR training tutorial](../../docs/tutorial/fsdp-training.md) for the evaluator contract and related upstream discussions.
