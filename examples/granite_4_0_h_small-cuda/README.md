# granite-4.0-h-small on TileFoundry — CUDA C kernels

Verified at **v0.0.1**, 2026-08-02. Not verified since; nothing in CI re-runs it.

A hybrid Mamba-2 / attention MoE decoder taken end to end on TileFoundry: token
ids in, next token id out, every kernel on the path written here in CUDA C.

    145.5 tok/s   one H200, batch 1, bf16, greedy

---

## 1. Environment

Nothing here is installed by this directory; it is what the directory was written
and measured against.

| | |
|---|---|
| GPU | one NVIDIA H200 SXM (143 GB), driver 575.57.08 |
| CUDA | 12.8 (`nvcc`), `CUDA_HOME` must be set |
| Python | 3.12 |
| `tilefoundry` | installed from the wheel, **not** editable — nothing here reads the source tree |
| borrowed from the environment | `torch` 2.9.1+cu128, `transformers` 5.14.1 |
| extra packages | **none** |
| weights | the published `ibm-granite/granite-4.0-h-small` checkpoint, 60 GB on disk |

Do not install `nvidia-cuda-cccl`. The kernels build with `ninja` through the
shim in `kernels/__init__.py`, against the host CUDA toolkit.

## 2. How this was produced

One agent, one prompt, no human help after it started.

| | |
|---|---|
| agent | Claude Code, model `claude-opus-5`, `--effort xhigh`, `--permission-mode bypassPermissions` |
| given | the checkpoint directory, the backend to use (CUDA C), the four filenames to deliver |
| told | everything about TileFoundry must be asked of the `tilefoundry` command; the model itself is its own to research |
| `tilefoundry` | read-only in its venv, installed from the wheel built at `965693c` |
| human input | **one** message, the prompt itself. Nothing else for the whole run |
| duration | 1.98 h, 176 tool calls, no sub-agents |

This is the prompt, translated from the Chinese it was written in. Only the
checkpoint directory is redacted, to `<ckpt>`.

```markdown
# You are the Pilot

Object: granite-4.0-h-small
  weights + config: <ckpt>

Task: get real tokens out of it on TileFoundry, and make it fast.

  Give it a prompt, it emits the continuation, and you can report a
  tokens-per-second number. The whole path has to be your own implementation:
  from a token id in, to the next token id out.

Backend: CUDA C (fixed for this round; do not change it)

Scope: the text decode path only. If this checkpoint carries a vision tower, or a
       multi-token-prediction part, neither is in scope.

## Deliverables

When you are done the working directory must hold these four, found from the
outside under these names:

    model.py            the model description
    runtime_model.py    its high-performance implementation
    kernels/            kernel sources, organised however you like
    run.py              an entry point that can be typed from outside as:

        python run.py --prompt "The capital of France is" --max-new-tokens 32

It must print the continuation and a tokens-per-second number. Support two more
switches: `--seed`, and `--greedy` (sampling off, so runs can be diffed).

**How you implement it is up to you**; only the names and this command line are
specified -- because these four are going out as examples afterwards, so the
shape has to be uniform.

## Environment

Already set up for you; you do not build it:

- the working directory is the current directory, with `.venv` inside it. Code,
  notes, scratch files -- put whatever you like there, the directory is yours
- python: `.venv/bin/python`
- command: `.venv/bin/tilefoundry` (also on PATH, so plain `tilefoundry` works)
- GPU: already yours exclusively through `CUDA_VISIBLE_DEVICES`; just use it
- `CUDA_HOME` is set. Do **not** install `nvidia-cuda-cccl`
- to install a python package use `.venv/bin/pip install`; leave the system
  environment alone

The `tilefoundry` in `.venv` is **read-only** and you cannot change it -- that is
deliberate. If something is wrong with it, say so; do not work around it by
editing the tool.

## The tool

`tilefoundry` -- **everything about TileFoundry is to be asked of it**. Do not ask
a person, do not go looking elsewhere.

The model itself is yours to research: this checkpoint's structure, the Hugging
Face implementation, the mathematics -- none of that is TileFoundry's knowledge,
and looking it up was always your job.

## Parallelism

You may dispatch your own sub-agents to work at the same time. How many, to whom,
whether at all -- your call.

**But typing `tilefoundry`, and judging whether you are stuck, are yours** and are
not to be delegated -- only the one who hit the wall knows what they were trying
at the time. If a sub-agent hits a wall writing a kernel, have it report the
wording back to you.

## Done when

Tokens come out, a speed is reported, and the four deliverables are there.

You may also stop short -- any one of these:

- you judge that everything this command surface can answer has been asked, and
  going further would only be guessing
- three times in a row stuck on the same kind of thing
- the loop turns over and what is left is repetition
```

## 3. How to use it

    python run.py --ckpt <checkpoint dir> --prompt "The capital of France is" --max-new-tokens 32

`--ckpt` is required — where the weights live is a fact about the machine, and a
default that exists on only one machine is a guess.

| flag | |
|---|---|
| `--greedy` | argmax instead of sampling; use it when comparing token streams |
| `--seed N` | fixes the draw when sampling. Reproduces a continuation exactly |
| `--temperature` | default 1.0 |
| `--reference` | run the authored HIR in `model.py` through the evaluator instead of the twin. ~9 tok/s, on purpose: it is what the twin is measured against |
| `--no-graph` | launch each kernel per step instead of replaying the captured CUDA graph |

The first run compiles the kernels into `kernels/.build/` (once, about a minute).

    model.py            the model as authored HIR — the reference
    runtime_model.py    its runtime twin, one @runtime_func per authored function
    kernels/granite.cu  16 kernels plus the build shim
    run.py              the entry point
    hf_alias.py         maps the checkpoint's names onto the Module's
    config.json         the published config, copied in so this directory carries
                        its own dimensions

## 4. Where it stands

**Measured, on the environment above.** Independently re-run after the agent had
finished and left:

| | tok/s | ms/token |
|---|---|---|
| **this implementation** | **145.5** | 6.85 |
| the same without the CUDA graph (`--no-graph`) | 100.3 | 10.2 |
| the authored HIR through the evaluator (`--reference`) | 8.9 | 109 |
| Hugging Face `transformers` on the same GPU | 2.7 | 367 |
| memory roofline | ~250 | 4.0 |

Hugging Face here has `mamba_ssm` and `causal_conv1d` absent, so it takes its
`torch_forward` fallback: the 54x is against what is installed, not against a
tuned baseline. The model is 32 B parameters with ~8.7 B active per token (10 of
72 experts), so a step is 17.5 GB of weight reads and essentially nothing else —
6.85 ms against a 4.0 ms roofline is 58% of what the memory system can do.

Also checked, on this version: the same `--seed` reproduces the text byte for
byte across processes; `--no-graph` and the graphed path produce byte-identical
text; `--reference` and the twin agree on all of the first 24 greedy tokens;
throughput holds at a ~3.2 k-token prompt (143.6 tok/s, 2% off short context);
asking past the KV capacity is refused with the way out named.

**What the model is.** 40 layers, `config.layer_types` alternating 35 Mamba-2
mixers with 5 full-attention layers (5, 15, 25, 35). Every layer then runs the
same MoE block — 72 experts, top 10 — plus a dense shared MLP beside it. Three
things about it would silently break a fixture copied from another model:

* **No positional encoding at all.** `position_embedding_type` is `"nope"`, so
  attention sees `position_embeddings=None`. Position lives in the Mamba recurrence.
* **Four published scalar multipliers** that are not 1: `embedding_multiplier` 12,
  `residual_multiplier` 0.22, `attention_multiplier` 1/128, `logits_scaling` 16.
  Dropping any of them still produces fluent text.
* **The checkpoint uses the legacy GraniteMoE names.** `block_sparse_moe.
  input_linear` / `output_linear` / `router.layer.weight` are what is on disk;
  `transformers` renames them on load. `head_dim` is unpublished and derived
  (128), and `lm_head` is tied to the embedding table.

**How correct it is.** Every runtime leaf against its authored function: 13/13,
worst `rel_l2` 4.8e-3. Teacher-forced over 144 steps of four prompts against
Hugging Face: authored HIR 142/144 top-1, runtime twin 141/144 — both sides
disagree at the same two steps and every disagreement is a near-tie (`' at'` vs
`' ('` after `100°C`). Every kernel against a torch expression of the same
arithmetic: 25/25, most bit-exact.

Three places where this deliberately differs from Hugging Face at sub-ulp scale,
each documented at the line: RMSNorm scales in f32 and rounds once; the routed
experts' sum accumulates in f32 across the ten; the router's logits *are* rounded
to bf16 before selection, deliberately — which ten experts win is a discrete
decision, and taking a different ten is a different model.

**What made it fast** (101 → 146 tok/s). A decode step at batch 1 is a matvec
pipeline: every matmul has one column, so there is no reuse and no tiling to do,
only bandwidth.

* **`ROWS * UNROLL` loads in flight, not occupancy.** An 8192→4096 projection is
  128 blocks on 132 SMs, so the parallelism cannot come from the grid. Unrolling
  the reduction so a lane issues four 16-byte loads together is where it comes from.
* **The router was the worst kernel in the model.** Projecting 72 logits and
  selecting the top ten in one block meant one SM pulling 590 KB: 1.1 ms a token
  by itself. Split into a matvec and a one-warp selection with the logits in
  registers, it is 8.6 µs.
* **Attention had a scaling cliff.** One block per query head is 32 blocks on 132
  SMs: 94 µs at 2048 positions. Striping positions across eight more blocks per
  head and merging the partial softmaxes makes it 15.9 µs, and nearly flat below.
* **One CUDA graph per decode step.** A step is ~775 launches, and launching each
  through pybind costs more than several of the kernels take — worth 3.3 ms a
  token. The sampler is inside the graph and writes the next input token into the
  buffer the next replay reads, so a token costs no host round trip beyond a
  4-byte read to test for EOS.

Two measurement traps this hit, worth knowing before trusting a microbenchmark
here: the H200's 50 MB L2 hides the real cost of a 34 MB weight tested in a loop
(reported 4.5 TB/s that a real step never sees), and timing a pybind call in a
Python loop measures the call, whose floor is about six microseconds.

**Where the remaining time goes,** at production shapes with L2 defeated:

| kernel | per call | of HBM peak | per step |
|---|---|---|---|
| `experts_gate_up` (×40) | 36.4 µs | 72% | 1456 µs |
| `in_proj` 16768×4096 (×35) | 35.3 µs | 81% | 1236 µs |
| `experts_down` (×40) | 21.4 µs | 61% | 856 µs |
| `out_proj` 4096×8192 (×35) | 19.9 µs | 70% | 697 µs |
| `shared_mlp` in+out (×40) | 18.1 µs | 45% | 724 µs |
| `router` (×40) | 8.6 µs | — | 344 µs |
| `lm_head` 100352×4096 (×1) | 188 µs | **91%** | 188 µs |
| ~500 small kernels | 1.7–3.4 µs | — | ~1100 µs |

`lm_head` is what the machine can actually do. Everything smaller loses to wave
quantization, which is why the biggest remaining win is not a faster kernel but a
wider one — fusing across the boundaries `model.py` draws.

**Not done.** No tensor parallelism: the run above uses one GPU. ~775 launches a
step, roughly 1.1 ms of them in kernels doing almost no work.

**One thing about the tool, recorded as found.** `tilefoundry check` compares two
callables on one set of inputs, and a runtime twin whose whole point is advancing
state in place cannot be measured that way — whichever side runs second reads the
other's output, and the CLI has no way to say "give each side its own copy". The
Python `check` API with wrapped callables works and uses the same predicates.
Separately, `--fn allclose` on a discrete output is refused with the reason
spelled out, and that refusal caught a real bug: the router was picking two wrong
experts out of ten.
