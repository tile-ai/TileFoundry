# MiniCPM3-4B on TileFoundry — CuTeDSL kernels

Verified at **v0.0.1**, 2026-08-02. Not verified since; nothing in CI re-runs it.

An MLA decoder taken end to end on TileFoundry: token ids in, next token id out,
every kernel on the path written here in CuTeDSL.

    281.5 tok/s   one H200, batch 1, 32 new tokens

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
| extra package | `nvidia-cutlass-dsl` 4.5.2 |
| **required pin** | **`cuda-bindings==12.9.2`** — see below |
| weights | the published `openbmb/MiniCPM3-4B` checkpoint, 7.6 GB on disk |

**The pin is not optional.** `cuda-bindings` 13.3.1 carries the CUDA 13 runtime,
and on a driver older than that the first thing `cute.compile` does —
`cudaGetDeviceCount()` — returns `cudaErrorInsufficientDriver`. It is reported as
`Target SM ARCH unknown is not compatible`, which points at the architecture and
not at the driver. Install `cuda-bindings==12.9.2` into this example's venv; it
shadows a shared environment without changing it. `pip` will warn that
`cuda-python` wants `~=13.3.1`; that warning is expected.

## 2. How this was produced

One agent, one prompt, no human help after it started.

| | |
|---|---|
| agent | Claude Code, model `claude-opus-5`, `--effort xhigh`, `--permission-mode bypassPermissions` |
| given | the checkpoint directory, the backend to use (CuTeDSL), the four filenames to deliver |
| told | everything about TileFoundry must be asked of the `tilefoundry` command; the model itself is its own to research |
| `tilefoundry` | read-only in its venv, installed from the wheel built at `34129de` |
| human input | **one** message, the prompt itself. Nothing else for the whole run |
| duration | 1.98 h, 326 tool calls — 198 on the main line and 128 in one sub-agent |

This is the prompt, translated from the Chinese it was written in. Only the
checkpoint directory is redacted, to `<ckpt>`.

```markdown
# You are the Pilot

Object: MiniCPM3-4B
  weights + config: <ckpt>

Task: get real tokens out of it on TileFoundry, and make it fast.

  Give it a prompt, it emits the continuation, and you can report a
  tokens-per-second number. The whole path has to be your own implementation:
  from a token id in, to the next token id out.

Backend: CuTeDSL (fixed for this round; do not change it)

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

The environment fix above was found and applied by the agent itself, unaided.

## 3. How to use it

    python run.py --ckpt <checkpoint dir> --prompt "The capital of France is" --max-new-tokens 32

`--ckpt` is required — where the weights live is a fact about the machine, and a
default that exists on only one machine is a guess.

**The first run prepares the checkpoint** into `prepared/` beside `run.py` and
says so (about a minute, once). The published checkpoint is one
`pytorch_model.bin` under names the authored Module does not use; converting it is
`Module.prepare`'s job, and doing it from the entry point means this example needs
nothing but the checkpoint. Pass `--prepared <dir>` to put it elsewhere.

| flag | |
|---|---|
| `--greedy` | argmax instead of sampling |
| `--seed N` | seeds the sampler |
| `--temperature` / `--top-p` | default 0.8 / 0.8, which is what `generation_config.json` asks for |
| `--backend cute\|torch` | which kernels. `torch` is the reference the CuTeDSL kernels were written against |
| `--reference` | run the authored Module through the HIR evaluator instead of its twin — the same tokens, very slowly |
| `--quiet` | the two output lines only |

    model.py            the model as authored HIR — the reference every number is measured against
    runtime_model.py    its runtime twin: the @runtime_module tree, the buffers, the captured graph
    kernels/            the CuTeDSL kernels, plus the torch spelling of the interface
                        they were written against
    run.py              the entry point
    tools/prep_ckpt.py  the one-off conversion run.py invokes
    hf_alias.py         maps the checkpoint's names onto the Module's
    config.json         the published config, copied in

## 4. Where it stands

**Measured, on the environment above**, re-run after the agent had finished:

| new tokens | tok/s |
|---|---|
| 32 | **281.5** |
| 128 | 281.3 |
| 512 | 223.7 |

Throughput falls with the context the step has to attend, which is the only part
of the step that grows. The HIR evaluator running the same authored Module does
7.8 tok/s, so the twin is about 38x it.

Decode at batch 1 is entirely a bandwidth problem: 8.15 GB of weights read per
token, and this card streams about 4.2 TB/s, which the head reaches. Every design
decision follows from that:

* **One warp per output element**, weight stored `(out, in)` so the row is
  contiguous under it, 128 bits per lane. No tiling, no staging — nothing is read
  twice, so there is nothing to reuse.
* **Fuse whatever shares a read.** The norm lives inside the kernel that consumes
  it; `gate` and `up` are one launch whose warp holds both rows, so the 6400-wide
  intermediates never reach memory; `q_a` and `kv_a` are one concatenated weight;
  the `scale_depth` residual is an epilogue, not a kernel.
* **Split-K where the projections are too narrow to fill the card.** One warp per
  output means `down_proj`'s 2560 outputs are 81920 threads on a machine that
  holds 270336. `_PLAN` in `kernels/cute_ops.py` records what won, including where
  splitting lost.
* **One CUDA graph.** 374 launches is ~2 ms of CPU before the GPU starts. Captured
  on the third step, replayed after; host work per token then measures as nothing.

Two things exist only because of the graph: the context length is an `int32[1]`
tensor the attention kernel reads (a graph freezes scalar arguments, so a `ctx`
passed as a number would be whatever it was at capture), and `append_cache` grows
a window into one preallocated buffer instead of concatenating — 124 torch views
per token was 300 µs of Python.

Six launches a layer, 374 in a token:

| # | kernel | reads | µs | TB/s |
|---|---|---|---|---|
| 1 | `rms_norm` + `q_a`‖`kv_a` | 5.2 MB | 3.19 | 1.70 |
| 2 | `q_b` + `kv_b` | 8.1 MB | 5.33 | 1.60 |
| 3 | rope + assemble + attend (ctx 64) | 0.8 MB | 5.21 | 0.16 |
| 4 | `o_proj` + scaled residual | 12.5 MB | 5.51 | 2.38 |
| 5 | `rms_norm` + `gate`/`up` + swiglu | 62.5 MB | 19.07 | 3.44 |
| 6 | `down` + scaled residual | 31.2 MB | 10.14 | 3.23 |
| | `embed`, final `rms_norm`, `lm_head` | 359 MB | 92.4 | 4.23 |

**How correct it is.** Every leaf against the authored Module on the real
checkpoint: `embed`, `final_rms_norm`, `lm_head`, `input_rms_norm` and `mlp` are
**bit-exact** (`rel_l2` 0); `mla_attention` and `decoder_layer` pass at context
lengths 0, 1, 7, 64, 257, 1024.

Attention is the one thing that cannot be bit-exact, and the bound it is held to
was derived rather than fitted. The reference multiplies q by k *in bf16* and
reduces that; the kernel keeps f32 from the load onwards. Putting both against the
same computation in f64 at the checkpoint's own activation scale, **the kernel is
the closer of the two at every context length** (1.80e-3 vs 2.20e-3 at ctx 1024).
So the disagreement is the reference's rounding, not the kernel's.

End to end, 48 greedy tokens on four prompts through all three implementations:
three prompts give **all 48 tokens identical** across twin, authored Module and
Hugging Face. The fourth forks at step 1, where the top two logits are
*numerically equal* — 14.2500 and 14.2500 in HF, margin exactly 0. Greedy decoding
resolves that by reduction order and nothing else. For scale: Hugging Face
disagrees with **itself** on one of these prompts, its cached decode and its full
re-forward parting company at step 22.

**The defect this example found, and its fix.** `model.py` here started as the
shipped `tilefoundry models minicpm3_4b --source` file. That file's
`prepare_inputs_for_generation` handed the attention kernel `config.head_dim ** -0.5`
= 0.1768 where `MiniCPM3Attention.scaling` is `qk_head_dim ** -0.5` = 0.1021 — the
rotary slice alone instead of the assembled head. It survived because the model
still reads fluently at the wrong number: on the real checkpoint the logits sat at
cosine 0.99879 of Hugging Face's, against 0.99998 at the right one. The L2 test
missed it because it reads the scale off `layer.self_attn.scaling` rather than
going through that method. **Fixed upstream in `#52`**; the `model.py` here carries
the same fix.

**What would come next.** Not the kernels: five of the seven are at 3.2–4.2 TB/s
against a ~4.2 TB/s ceiling. The gap is the **374 kernel boundaries** — the
microbenchmarks sum to 3.10 ms and the graph replays in 4.37 ms at ctx 262, so
roughly 0.7 ms a token is drain-and-relaunch. Removing it means a persistent
kernel with grid-level synchronisation, not a faster kernel.

**One thing about the tool, recorded as found.** `check --inputs random` feeds
non-rotary random values to rotary tables, which produces ulp-level mismatches in
attention that look like a kernel bug; with real cos/sin the two sides are
bit-identical. On a leaf carrying rotary tables, `--inputs random` reports a false
FAIL.
