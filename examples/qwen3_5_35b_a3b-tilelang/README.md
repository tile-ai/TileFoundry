# Qwen3.5-35B-A3B on TileFoundry — tilelang kernels

Verified at **v0.0.1**, 2026-08-02. Not verified since; nothing in CI re-runs it.

A hybrid gated-delta-net / attention MoE decoder taken end to end on TileFoundry:
token ids in, next token id out, every kernel on the path written here in tilelang.

    312.8 tok/s   one H200, batch 1, 32 new tokens, greedy

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
| borrowed from the environment | `torch` 2.9.1+cu128, `tokenizers` |
| extra package | `tilelang` 0.1.12 |
| weights | the published `Qwen/Qwen3.5-35B-A3B` checkpoint, 67 GB on disk |

## 2. How this was produced

One agent, one prompt, no human help after it started.

| | |
|---|---|
| agent | Claude Code, model `claude-opus-5`, `--effort xhigh`, `--permission-mode bypassPermissions` |
| given | the checkpoint directory, the backend to use (tilelang), the four filenames to deliver |
| told | everything about TileFoundry must be asked of the `tilefoundry` command; the model itself is its own to research |
| `tilefoundry` | read-only in its venv, installed from the wheel built at `809271a` |
| human input | **one** message, the prompt itself. Nothing else for the whole run |
| duration | 2.54 h, 731 tool calls — 231 on the main line and 500 across its own sub-agents |

This is the prompt, translated from the Chinese it was written in. Only the
checkpoint directory is redacted, to `<ckpt>`.

```markdown
# You are the Pilot

Object: Qwen3.5-35B-A3B
  weights + config: <ckpt>

Task: get real tokens out of it on TileFoundry, and make it fast.

  Give it a prompt, it emits the continuation, and you can report a
  tokens-per-second number. The whole path has to be your own implementation:
  from a token id in, to the next token id out.

Backend: tilelang (fixed for this round; do not change it)

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
deliberate. If something is wrong with it, report it (see below); do not work
around it by editing the tool.

## The tool

`tilefoundry` -- **everything about TileFoundry is to be asked of it**. Do not ask
a person, do not go looking elsewhere.

The model itself is yours to research: this checkpoint's structure, the Hugging
Face implementation, the mathematics -- none of that is TileFoundry's knowledge,
and looking it up was always your job.

## Parallelism

You may dispatch your own sub-agents to work at the same time. How many, to whom,
whether at all -- your call.

**But typing `tilefoundry`, judging whether you are stuck, and writing the records
are yours** -- three things, not to be delegated, because only the one who hit the
wall knows what they were trying at the time. If a sub-agent hits a wall writing a
kernel, have it report the wording back to you and you write it down.

## Two records

### `WORKLOG.md` -- the running account

In time order. Each entry has to reconstruct **the judgement made at the time**,
not a todo written afterwards. What you did, why you chose it that way, what came
of it. When a number comes out, record the number.

### `FINDINGS.md` -- the account of being stuck, and the real output of this task

Every time you need something outside `tilefoundry` to move forward on the
TileFoundry side -- guessing, trying, looking elsewhere, going from memory --
stop immediately, write one entry, then continue. (Researching the model itself
does not count as being stuck; that was always your job.)

Seven fields per entry:

    ## Q<n>  <one-line title: what the command surface did not give you>
    - object/backend   Qwen3.5-35B-A3B / tilelang
    - stage            step one (translation) or step two (optimisation)
    - source           written for real, or adapted from something
    - category         command surface / env
    - what you hit     one line. Write "what the command surface did not give me",
                       not "I do not know how"
    - what you tried   raw, unpolished. What you typed and what came back, verbatim
    - how you got past it   or: did not, stuck for N minutes

Writing it down is what makes it output. **Being stuck is not failure; being stuck
without recording it is.**

## Done when

Tokens come out, a speed is reported, and the four deliverables are there.

You may also stop short -- any one of these:

- you judge that everything this command surface can answer has been asked, and
  going further would only be guessing
- three times in a row stuck on the same kind of thing
- the loop turns over and what is left is repetition

Either way, leave both records in the working directory.
```

The two records it asked for are an account of the run rather than part of the
example, so they are not here.

## 3. How to use it

    python run.py --ckpt <checkpoint dir> --prompt "The capital of France is" --max-new-tokens 32

`--ckpt` is required unless `$QWEN35_CKPT` is set — where the weights live is a
fact about the machine, and a default that exists on only one machine is a guess.

| flag | |
|---|---|
| `--greedy` | argmax instead of sampling; use it when diffing token streams |
| `--seed N` | seeds the sampler |
| `--temperature` / `--top-p` / `--top-k` | sampling controls |
| `--no-graph` | launch each kernel per step instead of replaying the captured CUDA graph |
| `--impl torch` | swap the tilelang kernels for the torch spelling of the same interface, one leaf at a time |
| `--layers N` | a cut-down stack — same kernels, same shapes, fewer layers. Turns the loop over quickly on real weights |
| `--capacity N` | the fixed KV capacity of the graphed path |
| `--verify-config` | assert the reconstructed config still matches the published `config.json` |
| `--compare` / `--verbose` | working switches, not part of using it |

The first run compiles the tilelang kernels (once, a few minutes).

    model.py            the model as authored HIR — the reference
    runtime_model.py    its runtime twin, plus the Session that drives one step into many
    kernels/            attn / gdn / moe / basic in tilelang, and torch_ref.py — the
                        torch spelling of the same interface they were written against
    run.py              the entry point
    config.py           the published shape configuration, reconstructed (see below)
    weights.py          the resource that answers canonical names from the raw checkpoint
    graph.py            the CUDA-graph capture of one decode step

`config.py` exists because a shipped authored file has to be self-contained:
`tilefoundry models qwen3_5_35b_a3b --source` prints a `model.py` whose first
import is the test package's config module, and that package is not installed.
Every value in it is either published in the checkpoint's `config.json` or read
back out of the signatures `tilefoundry models` prints — nothing is guessed, and a
published dimension is published rather than derived (`head_dim` is 256 while
`hidden / num_heads` is 128).

## 4. Where it stands

**Measured, on the environment above**, re-run after the port:

```
prompt        5 tokens   16.6 ms   (301.7 tok/s)
generated    32 tokens  102.3 ms   (3.20 ms/token)
             312.8 tok/s
```

`tok/s` is decode throughput: new tokens over the time to produce them, measured
after the prompt is consumed. It excludes weight loading and kernel compilation,
which happen once, and prefill, which is reported separately. One token per step,
batch one — the regime `model.py` declares (`S = 1`) — so it is a latency number.

The floor is one pass over the active parameters: 5.89 GB of weight reads per
token, which at 4.8 TB/s of HBM is 1.23 ms/token (815 tok/s). This run moves them
at 1.84 TB/s, 38% of peak.

**What the model is.** 40 layers on a repeating cycle of three gated-delta-net
mixers to one full-attention layer. Every layer then runs a 256-expert top-8 MoE
block with a shared expert beside it. Attention is partially rotary: `head_dim` is
256 of which only 64 carry RoPE, with 16 query heads over 2 key/value heads.

**Precision.** Activations and every accumulator are f32; the weights a matmul
contracts over are bf16, which is what the checkpoint stores. Nine small weights
stay f32 — the gammas, `a_log`, `dt_bias` and the convolution kernel — because
they are added to and exponentiated rather than contracted over, and `1 + w` in
bf16 would throw away the low bits of exactly the correction their converter
exists to apply. `weights.py` is `Module.prepare` without the directory: the same
walk, the same converters, one source of truth, but the converter runs on demand
on the GPU. Preparing to a directory at the declared precision would be ~280 GB
of transposes nothing reads twice.

**Not done.** One GPU. Sampling controls are present but the numbers above are
greedy.
