# Qwen3/dmuon communication and pipeline training

This example composes pipeline, context, expert and data parallelism around
Qwen3-30B-A3B and dmuon's public optimizer APIs. It also isolates an HSDP/TP
optimizer boundary for comparison with TileFoundry's evaluator. Weights are
randomly initialized; the workload checks training semantics, not convergence.

Model dimensions, routing and optimizer policy live here. TileFoundry core owns
generic communication values and ordering; the NVIDIA target owns NCCL lowering.
The training runner uses PyTorch autograd and `Schedule1F1B`. Generated NCCL
programs are exercised independently against the same primitive oracle.

## Setup

Install TileFoundry and its NVIDIA target following the
[developer guide](../../docs/develop.md). The training reference was validated
with PyTorch 2.14.1, Transformers 5.2.0 and dmuon at revision
`083ea977c045c5c15881845b3a3c465b6d2c4bb2`. No pretrained checkpoint is needed.
Generated communication additionally requires CUDA, CMake/Ninja and NCCL headers
and libraries; set `NCCL_HOME` when NCCL is outside the normal search path.
Plotly and Matplotlib are optional timeline exporters.

Run commands from the repository root and use a fresh output directory for each
run. `--transport native` uses NCCL's configured selection. `--transport socket`
forces the Socket path with P2P, SHM, NVLS and MNNVL disabled for protocol checks.
Transport details and software versions are recorded with each run's outputs.

## Communication contracts

The pinned dmuon FSDP2/DDP backends, Gram helpers, clipping and checkpoint paths
use seven tensor collective families. List and tensor variants share the same
value semantics. The surrounding training workload also requires P2P and
variable-count exchanges.

| Requirement | Representation |
| --- | --- |
| Broadcast, reduce, all-reduce | `Collective`, with SUM/AVG/MAX and communicator-local roots |
| Gather, scatter, all-gather, reduce-scatter | `Collective`, with static payloads and equal scatter partitions |
| P2P, all-to-all and uneven all-to-all | `Exchange`, with matched counts, including zero and self edges |
| Runtime counts | Count-row all-gather followed by capacity-bounded dynamic `Exchange` |
| Async issue and completion | Host TIR communicator/stream/event handles, grouped enqueue, waits and synchronization |
| Buffer reuse | Completion-dependent physical lifetime checks and explicit event export |
| Object metadata and process-group setup | Host-side PyTorch operations |

Participant ordering comes from existing `Mesh` objects. Context identities keep
independent communicators distinct even when their members coincide. `Reshard`
continues to express an ownership transition; explicit communication expresses
its realization. No training-specific communication operations are added.
The owning contracts are in [HIR](../../docs/spec/hir.md),
[host TIR](../../docs/spec/tir.md#24-host-communication),
[analysis](../../docs/spec/analysis.md#5-host-communication-ordering) and the
[CUDA runtime](../../targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md).

`check_primitives.py` compares the PyTorch/NCCL realization with independent HIR
evaluation. `check_native.py` compiles the same cases through the public compiler
entry points; data transfers execute in the generated library. Both cover 106
cases per rank: five dtypes, reduction variants, scalars, empty tensors, negative
axes, uneven exchanges, sliced subgroups and reordered participants. The native
probe adds 16 changing-count protocols, nested groups, cross-stream dependencies,
and invalid-count/alias rejection. Both passed with native and Socket transport.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/qwen3_dmuon/check_primitives.py --output .tmp/qwen-primitives
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/qwen3_dmuon/check_native.py --output .tmp/qwen-generated
```

GPU count readback synchronizes before variable-count NCCL issue. Exported events
obligate the caller to retain buffers and native resources until completion.
Static loops are checked across iterations; dynamic host control flow and unknown
kernel effects in the same host entry are rejected. Communication-only modules
compile without a dummy kernel or CUTLASS dependency.

## Pipeline training

Full-width mode retains hidden size 2048, 32 query heads, 4 KV heads, head dimension
128, 128 experts, top-8 routing, expert width 768 and vocabulary size 151,936.
Depth is configurable: two layers contain 1,868,573,184 parameters; all 48 layers
contain 30,532,122,624. Tiny mode keeps the same architectural features at reduced
width for quick checks.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/qwen3_dmuon/pipeline.py --width tiny --layers 4 --stages 2 \
  --microbatches 4 --tokens 64 --steps 2 --optimizer dmuon \
  --overlap on --output .tmp/qwen-pipeline
```

Use `--width full` to retain the original layer dimensions, and `--layers 48
--stages 4 --microbatches 8 --reference-offload` for full depth. Layers must divide
evenly among stages; microbatch count must be at least the stage count. Reference
offload moves the independent model between checks and retains only stage-owned
reference gradients. Timed training keeps its own model and optimizer state.

PP groups cross stages, while each stage owns its own data/optimizer groups.
The PP axis never enters dmuon's TP inference. The example checks each activation
and gradient transfer against an `Exchange` contract, including peers, context,
shape, dtype, count, microbatch and FIFO order. Router outputs cross stage
boundaries so the auxiliary loss covers all layers within each microbatch.

Attention projections and individual expert gate/up/down matrices use Muon;
embeddings, LM head, router and norms use AdamW. Explicit expert layouts preserve
matrix boundaries in fused storage. Each validation step compares the loss and
all local/owned gradients against an unpartitioned Transformers reference, checks
published weights across data peers, and updates the reference for the next step.
`--optimizer sgd` provides an independent partitioning baseline.

The pinned dmuon revision requires activation checkpointing for this 1F1B
integration; `--checkpoint auto` enables it. Uncheckpointed execution misses a
drained microbatch's gradient in the upstream backward lifecycle. The example
uses public optimizer APIs without patching that lifecycle.

## Packed sequences with CP and EP

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/qwen3_dmuon/pipeline.py --width full --layers 4 --stages 2 \
  --cp 2 --ep 2 --microbatches 4 --tokens 8192 --steps 2 \
  --optimizer dmuon --overlap on --reference-offload \
  --routing-check replay-near-ties --profile-steps 4 \
  --output .tmp/qwen-packed
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/qwen3_dmuon/check_routing.py
```

Each data lane packs multiple documents with unequal padding. Attention prevents
cross-document access, position IDs restart per document, and loss ignores
transitions and padding. Gradient scaling uses the global valid-target count
and the appropriate dense/expert replica groups.

CP gathers K/V with differentiable all-gather and SUM reduce-scatter backward.
EP exchanges counts, expert IDs and token rows, then reverses dispatch after
local expert computation. Contributions accumulate in ascending expert order;
empty destinations retain their autograd dependencies. Observed communication
shapes and count matrices are checked against the generic primitives.

The default `--routing-check strict` rejects changed top-k expert sets. At a near
tie, small FP32 differences can change a discrete choice. The explicit
`replay-near-ties` mode verifies probabilities and cutoff margins, then reuses
reference indices during validation while recomputing differentiable weights.
Timing/profiling always uses online routing. This mode checks gradients conditional
on common routing; it does not establish identical unconstrained top-k choices.

The full-width four-layer packed case completed 11 steps, with maximum owner
gradient error `6.58e-9`. Strict CP-only, EP-only and composed Socket cases also
passed. Full-depth PP/DP checks passed with maximum owner gradient error `1.40e-9`.
Primitive dtype coverage is separate from full-model mixed-precision training.

## Overlap and optimizer variants

`--overlap on` enables asynchronous optimizer updates and layer prefetch.
`--profile-steps N` adds N unprofiled timing steps and N captured steps after
validation and warmup; the reference is released before these windows. Compare
otherwise identical runs with overlap on/off. Timing uses the slowest rank.

```bash
python examples/qwen3_dmuon/timeline.py .tmp/qwen-packed
```

The exporter writes an interactive timeline, a PNG preview, Perfetto events and
an overlap summary. Metrics use unions of CUDA kernel intervals; NCCL intervals
include rendezvous/spinning, so their intersection with compute does not measure
hidden wire time. Merged traces currently require a shared host clock.
`--compare-final DIRECTORY` checks final weight fingerprints against a prior run
with the same number of data steps, allowing a fully checked replay of timed work.

HSDP uses `--replicas 2` inside each stage's data group. `--ns-backend gram`
selects dmuon's Gram implementation; `--muon-max-norm VALUE` checks global
Muon-only clipping against an independent norm calculation. AdamW parameters
are excluded. `--forward-unshard all_gather` selects attention publication;
explicit expert layouts retain owner broadcast.

The pinned HSDP path disables optimizer-tail prefetch because of its interaction
with model-state export, and rejects all-gather publication because exported
replica-row weights diverge. Packed CP/EP currently requires layer boundaries,
no HSDP replicas and no global Muon clipping across its optimizer views.

## Isolated HSDP/TP boundary

```bash
python examples/qwen3_dmuon/run.py --mode shapes
python examples/qwen3_dmuon/run.py --mode model --steps 2 --check-dmuon
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/qwen3_dmuon/run.py --mode nccl --steps 2 --check-dmuon \
  --output .tmp/qwen-optimizer
```

This consumer updates one query-projection matrix from real tiny-model gradients;
other weights remain fixed. Ranks are `4 * replica + 2 * shard + tp`. HSDP
reductions produce owner gradients, TP gather assembles the matrix for Muon, and
publication reverses the ownership flow. TP partitioning applies at the optimizer
boundary; the reference forward/backward remains replicated.

Checks separate reduction error from optimizer rounding: compare the gathered
gradient with the global mean, compare Newton-Schulz exactly with dmuon on the
same gradient, and compare published shards exactly with the owner's update.
This avoids treating FP16 polynomial sensitivity as a transport failure.

## Larger simulated grids

After numerical validation, measure communication costs and supply them to the
phase-level simulator:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/qwen3_dmuon/calibrate.py --transport native --output .tmp/native.json
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/qwen3_dmuon/calibrate.py --transport socket --output .tmp/socket.json
python examples/qwen3_dmuon/simulate.py \
  --validated-run .tmp/qwen-packed --native .tmp/native.json --remote .tmp/socket.json \
  --grid 4,4,4,1 --grid 4,8,8,1 --grid 8,8,8,2 --grid 4,32,8,1 \
  --layers 48 --tokens-per-cp 4096 --microbatches 16 --steps 16 \
  --output .tmp/qwen-virtual-grids
```

Grids specify `PP,CP,EP,outer_data`. The listed sweeps cover 64-1,024 virtual ranks
and 16K-128K packed positions. They validate mesh domains, counts, capacities,
dependency completion and resource serialization without evaluating model tensors.
PyTorch's existing 1F1B order supplies the pipeline schedule.

Communication interpolates measured samples and extrapolates group size;
compute scales measured kernel intervals by FLOP/ownership ratios.
`--remote-scale` and `--contention` expose assumptions for sensitivity studies.
The simulator has no within-phase compute/communication overlap or physical
memory-feasibility proof. Socket emulation and inferred timings are not measured
multi-host performance. Outputs retain calibration provenance and assumptions.

## Remaining integration

- Mixed compute/communication scheduling and a concrete core communication-cost
  realization. Current host compilation handles explicit communication protocols.
- Full training HIR and TP composition with the executing PP/CP/EP path.
- Scoped optimizer checkpoint/resume, HSDP sharded-publication consistency and
  sharded explicit expert layouts; uncheckpointed drain needs a dmuon fix.
- Registered-memory/NVSHMEM extensions and real-fabric calibration are separate
  follow-ups, each requiring its own completion and measurement checks.

## Sources

- [Qwen configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json).
- [Pinned dmuon source](https://github.com/X-Square-Robot/dmuon/tree/083ea977c045c5c15881845b3a3c465b6d2c4bb2).
- [PyTorch pipeline stages and schedules](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html).
- NCCL [stream semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)
  and [group semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html).
