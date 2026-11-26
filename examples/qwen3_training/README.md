# Qwen3 training: express five distributed configurations and export them to a simulator

Read these three limits first; every number this example prints depends on them:

1. **Communication times are assumed.** `calibration/assumed.json` states them
   (`"transport": "assumed"`: 10 µs plus bytes at 360 GB/s on the bus); nothing was
   measured on a network.
2. **One communication slot per rank.** A rank runs one compute node and one
   communication node at a time; two collectives of one rank never overlap.
3. **Expert routing is uniform.** Every rank sends every expert the same number of
   rows (a fixed capacity); the `all_to_all` bytes are the uniform-routing bytes.

## What this is

One training step of a Qwen3-MoE model (8 layers, 128 experts, top 8, 32 768 tokens
per sequence, 4 microbatches) on 8 GPUs, written in TileFoundry HIR for five
parallel configurations. TileFoundry derives the backward, places the
communication, and analyzes memory; this example reads that into a graph of nodes
(kernel calls, other Ops, one rank's part of each communication) and values, and
exports it to ASTRA-sim under three recompute choices and four schedules. Nothing
here executes the model. The goal is to tell which configuration is better, and
why, before anything runs.

```text
model.py      the model in HIR: one stage of layers, its kernels (fused child Modules), Muon
step.py       one training step: microbatch loop, both stages, backward, optimizer
configs.py    the five configurations and the model size
facts.py      step -> nodes, data edges, values, per-rank peak memory
policies.py   four schedules: order edges on top of the data edges
export.py     writes runs/<config>/<recompute>/<policy>/graph.json; --fit: the largest tokens that fit
report.py     joins the simulated times with graph.json: timeline.chrome.json, runs/report.md
render.py     timeline.chrome.json -> PNG
sim/          to_chakra.py (graph.json -> Chakra traces), run.sh, fit.py, simulator configs
calibration/  compute.json (measured kernels), assumed.json, simulator.json, measure.py
patches/      the one ASTRA-sim patch this needs
```

## Run it

From the repository root, in the project's environment:

```bash
python -m examples.qwen3_training.export --out runs            # 60 graph.json, about 5 minutes
SIM_ROOT=<install dir> examples/qwen3_training/sim/run.sh runs  # 60 simulations
python -m examples.qwen3_training.report runs                  # runs/report.md, timeline.chrome.json
```

Optional:

```bash
python -m examples.qwen3_training.export --fit --out runs      # runs/fit.json: largest tokens that fit
uv run --no-project --with matplotlib==3.9.2 python -m examples.qwen3_training.render \
    runs/cp_hsdp/none/program/timeline.chrome.json --out step.png --ranks 0,4
```

`export.py` takes `--config`, `--recompute` and `--policy` to choose a subset; a
choice left out means all of its values. `report.py` exits 1 when a run is missing,
deadlocked, or, under `program`, its simulated peak differs from the graph's peak.
Open a `timeline.chrome.json` in <https://ui.perfetto.dev>: per rank, a compute lane,
a communication lane, and a memory lane with the capacity and the peak.

## The configurations

The mesh is `(pp, data, ep, cp)` over 8 GPUs, `cp` varying fastest. `pp` is 2 in
every configuration.

| config | data | ep | cp | dense weights split over | expert weights split over |
|---|---|---|---|---|---|
| `cp` | 2 | 1 | 2 | data, cp | data, cp |
| `cp_ep` | 1 | 2 | 2 | ep, cp | cp |
| `ep` | 2 | 2 | 1 | data, ep | data |
| `cp_hsdp` | 2 | 1 | 2 | cp | cp |
| `ep_hsdp` | 2 | 2 | 1 | ep | — |

An axis that does not split a weight holds a whole copy of it (HSDP); its gradient is
all-reduced over that axis.

Recompute choices (`step.py`):

| choice | the backward computes again |
|---|---|
| `none` | only the inside of FlashAttention and the head (always: they keep no scores and no logits) |
| `regather` | also each layer's gathered weights: one more `all_gather` per weight in the backward |
| `layer` | also every Op of each layer outside its kernels (weight and K/V gathers, index selects) and the inside of each kernel, so each kernel's backward runs its forward again |

A recompute selector names Ops, not calls of child Modules: under `layer` a kernel's
output, which the next kernel reads, stays saved.

Schedules (`policies.py`), each a set of order edges; the data edges never change:

| policy | on each rank |
|---|---|
| `program` | every node in program order, one at a time: nothing overlaps (the peak is `facts`' peak) |
| `prefetch` | compute in one chain and communication in another, so they overlap; a weight's `all_gather` waits only for the communication before it |
| `1f1b` | the two chains, with microbatch passes in 1F1B order |
| `comm_first` | the two chains, and a compute node that feeds communication runs before others that are ready |

## Times

- **Compute.** A kernel with a ready-made implementation takes its time from
  `calibration/compute.json`, measured in BF16 on one H200 (`calibration/measure.py`,
  run in the venv of `calibration/requirements-fa.txt`). FlashAttention takes FA4's
  time as is. A kernel measured by its cuBLAS GEMMs never takes less than its fused
  roofline (`max(flops / peak, bytes / HBM)`), and under `layer` its backward also
  takes its forward's GEMMs. A node with no measurement takes the roofline. The HIR
  is f32: memory is f32, times are BF16.
- **Communication.** `calibration/assumed.json` gives the time of each kind, group
  size and byte count. ASTRA-sim takes a size, not a time, so each node carries
  `sim_bytes`: the size at which the simulator's network spends the assumed time,
  from the line `sim/fit.py` fits to the simulator (`calibration/simulator.json`).
  The real byte count stays in `bytes`.
- **Memory.** A value takes its bytes from the start of the node that produces it to
  the end of the last node that reads it; a view shares its root's storage. A
  kernel's own intermediate values are not counted (they stay on chip in a fused
  kernel).

## Install the simulator

ASTRA-sim with the analytical network backend, pinned:

| repo | commit |
|---|---|
| astra-sim | `4ce9ecfe746a3ee5fe47559a11d91644d641fa7f` |
| abseil-cpp | `20240722.0` |
| protobuf | `29.0` |

```bash
export SIM_ROOT=<install dir>
mkdir -p $SIM_ROOT/src $SIM_ROOT/deps/src && cd $SIM_ROOT/src
git clone --recursive https://github.com/astra-sim/astra-sim.git
cd astra-sim && git checkout 4ce9ecfe746a3ee5fe47559a11d91644d641fa7f && git submodule update --init --recursive

# abseil and protobuf from source (gives protoc; no root)
cd $SIM_ROOT/deps/src
wget https://github.com/abseil/abseil-cpp/releases/download/20240722.0/abseil-cpp-20240722.0.tar.gz
wget https://github.com/protocolbuffers/protobuf/releases/download/v29.0/protobuf-29.0.tar.gz
tar xf abseil-cpp-20240722.0.tar.gz && tar xf protobuf-29.0.tar.gz
D=$SIM_ROOT/deps
cmake -S $D/src/abseil-cpp-20240722.0 -B $D/build/absl -G Ninja -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DABSL_PROPAGATE_CXX_STD=ON \
  -DCMAKE_INSTALL_PREFIX=$D/install
cmake --build $D/build/absl --target install -j
cmake -S $D/src/protobuf-29.0 -B $D/build/protobuf -G Ninja -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package \
  -DCMAKE_PREFIX_PATH=$D/install -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$D/install
cmake --build $D/build/protobuf --target install -j

# ASTRA-sim, with the patch: without it the binary double-frees at exit with subgroup
# collectives, and the last "finished" lines can be lost
export PATH=$D/install/bin:$PATH CMAKE_PREFIX_PATH=$D/install PROTOBUF_FROM_SOURCE=True
export protobuf_DIR=$D/install/lib/cmake/protobuf absl_DIR=$D/install/lib/cmake/absl
cd $SIM_ROOT/src/astra-sim
git apply <repository>/examples/qwen3_training/patches/0001-collectiveplan-no-double-free.patch
./build/astra_analytical/build.sh

# the Python side: Chakra, for to_chakra.py and chakra_timeline_visualizer
cd $SIM_ROOT && uv venv --python 3.10 venv
VIRTUAL_ENV=$SIM_ROOT/venv uv pip install "protobuf==5.29.5" setuptools setuptools-grpc "grpcio-tools==1.70.0"
VIRTUAL_ENV=$SIM_ROOT/venv uv pip install --no-deps --no-build-isolation ./src/astra-sim/extern/graph_frontend/chakra
```

`run.sh` uses `$SIM_ROOT/src/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware`,
`$SIM_ROOT/venv/bin/python` and `$SIM_ROOT/venv/bin/chakra_timeline_visualizer`. The
network is one 8-port switch (`sim/net_switch8.yml`: 100 GB/s, 1 µs per hop); ring
collectives, direct all-to-all (`sim/system.json`). After changing either file, run
`SIM_ROOT=... $SIM_ROOT/venv/bin/python examples/qwen3_training/sim/fit.py` again.

## How this differs from PR #327

| item | PR #327 | here | why |
|---|---|---|---|
| EP routing | data-dependent all-to-all | fixed capacity, uniform | dynamic routing cannot be expressed yet |
| optimizer | DMuon; AdamW for the rest | our own Muon (momentum, 5 Newton–Schulz steps on one owner rank); momentum SGD for the rest | only the communication and memory structure matter here |
| router aux loss | yes | no | stages pass only the hidden state |
| attention | BF16 FA4, varlen by document | one fused f32 FlashAttention kernel, timed as FA4 BF16, causal over the whole sequence | document boundaries depend on data |
| activation checkpointing | per stage | three choices, `none` by default | an explicit dimension of the export |
| numbers | a real run | nothing runs | the goal is the comparison, not execution |

## Limits

- Communication times are assumed (above); a measured table in the same format
  replaces `assumed.json`.
- One communication slot per rank: ASTRA-sim issues a rank's collectives in order.
- Memory is per node: a value lives from its producer's start to its last reader's
  end; a kernel's intermediate values are not counted.
- Expert routing is uniform; there is no router aux loss.

## Add a configuration or a policy

- **Configuration.** Add a `Config` to `CONFIGS` in `configs.py` (8 GPUs, `pp` 2).
  If it splits the weights another way, `model.py` builds the types from
  `dense` and `expert`. Measure its kernels again (`calibration/measure.py`) or let
  them fall back to the roofline. Then `export.py --config <name>`.
- **Policy.** Add a function `Facts -> order edges` to `POLICIES` in `policies.py`.
  Keep each group's collectives in one order on all its ranks; `policies.check`
  refuses a cycle or a mismatched order before a file is written.
