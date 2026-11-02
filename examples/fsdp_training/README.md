# Two-block FSDP training in HIR

This study asks how releasing gathered parameters after forward changes storage and logical communication while preserving the training update. The HIR evaluator simulates participants on CPU. Forward, backward, and momentum SGD are authored explicitly. PyTorch autograd supplies an independent numerical reference.

## Run the comparison

Use an editable TileFoundry installation with the NVIDIA target, following [the development guide](../../docs/develop.md). From the repository root:

```bash
python -m examples.fsdp_training.run --participants 1 2 4
```

Each configuration performs two training steps on changing batches, checks logits, global mean loss, every parameter gradient, updated parameters, and momentum buffers, then analyzes memory. The command prints one JSON record per participant count and residency policy. Analysis can take several minutes per configuration. It needs neither a checkpoint nor a dataset download.

| File | Responsibility |
|---|---|
| `model.py` | Explicit forward/backward HIR and parameter residency policies |
| `reference.py` | Dense embedding, PyTorch SDPA/autograd, and optimizer oracle |
| `run.py` | Correctness checks and reports derived from existing HIR facts |

## Model and training contract

The workload borrows RMSNorm, QK normalization, rotary position encoding, and a ReLU-squared MLP from [modded-nanogpt's architectural direction](https://github.com/KellerJordan/modded-nanogpt/tree/4ea6b937337a4889b8cfe3f38a93d120048d8f71). GQA is an adaptation: this is not a reproduction of that repository's speedrun model or optimizer.

```text
tokens -> embedding -> block 1 -> block 2 -> RMSNorm -> vocabulary head -> cross-entropy
block  -> RMSNorm -> causal GQA -> residual -> RMSNorm -> ReLU² MLP -> residual
```

The fixed FP32 configuration has batch 8, sequence length 8, width 16, four query heads, two KV heads, head width 4, MLP width 64, and vocabulary 32. RMSNorm is affine-free, projections have no bias, embedding and head are untied, and there is no dropout. Random nonzero initialization makes every parameter's gradient nonzero on the first step. This avoids a zero output projection concealing an incorrect backward path.

HIR represents the tiny embedding lookup as one-hot matrix multiplication; the reference uses `torch.nn.functional.embedding`. Fixed rotation matrices express RoPE in HIR, while the reference rotates adjacent coordinate pairs. HIR broadcasts K/V over a separate query-group dimension; the reference repeats KV heads before SDPA. These are equivalent small-workload representations, not production embedding or attention kernels.

Every participant receives different complete sequences. Parameters and momentum are split along tensor dimension 0. Linear backward contracts the split token dimension to produce actual `Partial(sum)` contributions. Explicit reduce-scatter followed by one division by the participant count produces gradients of the global mean loss. HIR `Reduce` has complete reduction semantics, used for the scalar loss. The optimizer uses learning rate 0.01 and momentum 0.9, with no dampening, weight decay or Nesterov update. The second step consumes the first step's returned state; inputs are not mutated.

Checks use `rtol=5e-4`, `atol=2e-5` for FP32 reduction-order differences. The runner prints the observed maximum absolute errors separately for logits, loss, gradients, parameters and momentum.

## Residency experiment

Both variants use the same arithmetic and saved activations. The release variant gathers inside each forward stage, releases its full weights at the stage boundary, and re-gathers weights needed by backward. The retain variant returns full weights from forward and passes them to backward. Saved values are ordinary HIR tuples; there is no autodiff annotation, lifetime operation, or separate allocator.

Embedding backward does not read the embedding weights, so neither variant gathers them again. Each transformer block owns six parameter tensors. The head owns one. A stage boundary groups their lifetimes; it does not declare collective packing or an NCCL launch group.

For multiple participants the per-step logical transfers are:

| Quantity | Release | Retain |
|---|---:|---:|
| Tensor all-gather occurrences | 27 | 14 |
| Tensor gradient reduce-scatter occurrences | 14 | 14 |
| Peak live gathered-weight storage per participant | 11,264 bytes (11 KiB) | 24,576 bytes (24 KiB) |
| All-gather result bytes per participant | 51,200 | 26,624 |
| Reduce-scatter input bytes per participant | 26,624 | 26,624 |
| Persistent parameters plus momentum per participant | 53,248 / P | 53,248 / P |

The runner also reports peak live gathered-weight storage. It identifies gathered values through call/tuple results and uses the allocator's accepted roots and lifetime windows to deduplicate aliases. It does not identify weights by tensor shape. Single-participant execution reports zero interparticipant transfers and `null` for a separate gathered-weight peak, since full weights can alias their original storage.

At two participants, both policies matched the dense reference over two updates with maximum absolute errors of `5.1e-7` for logits, `1.2e-7` for gradients/momentum, and `3.0e-8` for updated parameters; scalar loss agreed at the reported FP32 precision. A separate local check against PyTorch FSDP2 on two L20Z GPUs also passed both policies, with maximum gradient error `3.9e-7`. That check used real sharded parameters, reduced gradients and sharded momentum, and does not establish distributed execution of the HIR or equality of physical communication schedules.

These are selected storage components, not total training memory: saved activations, gradients, functional update outputs, fixed inputs and other temporaries also occupy storage. Caller-owned inputs remain persistent. Logical collective payloads are not link traffic, achieved bandwidth, or elapsed time. The H200 target supplies storage ownership for this analysis; no H200 execution or speedup is claimed.

The result is a storage/communication trade-off. Retention removes re-gathers, while release permits smaller gathered-weight residency. Choosing between them needs a memory constraint or a future scheduled/runtime experiment.

## Scope

This models minimal FSDP2-style training semantics using existing mesh placements. It supports a single GPU topology level, even shards, and concrete shapes. GQA supplies the workload; the parallelism demonstrated here is data parallel training with sharded parameters, gradients and optimizer state. Muon, mixed precision, accumulation, offload, prefetch/overlap, uneven shards, TP/CP/HSDP, inference and distributed CUDA code generation remain follow-ups.

Small ownership, reduction and lifetime regressions live under `tests/core`; CI includes one two-participant transformer smoke case for each policy. The runner supplies the broader 1/2/4-participant study. See [the HIR training tutorial](../../docs/tutorial/fsdp-training.md) for the evaluator contract and related upstream discussions.
