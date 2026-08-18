<p align="center">
  <img src="https://github.com/user-attachments/assets/3e29ee3e-2fae-4243-ba73-0efc04ac7645" alt="TileFoundry" width="100%">
</p>

<h3 align="center">Hand the compiler to the agent.</h3>

<p align="center">
  <i>Not an agent that becomes the compiler,<br>
  and not an agent plugged in as one of its passes.<br>
  The compiler stays a tool — the agent is simply the one holding it.</i>
</p>

<p align="center">
  <b>One prompt in</b>&nbsp; · &nbsp;<b>612 tok/s out</b>&nbsp; · &nbsp;<b>nobody in the loop</b>
</p>

---

<p align="center">
  <a href="https://pypi.org/project/tilefoundry/"><img src="https://img.shields.io/pypi/v/tilefoundry.svg" alt="PyPI"></a>
  <a href="https://codecov.io/gh/tile-ai/TileFoundry"><img src="https://codecov.io/gh/tile-ai/TileFoundry/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://github.com/tile-ai/TileOPs/pulls?q=is%3Apr+is%3Amerged+label%3Afoundry"><img src="https://img.shields.io/github/issues-search/tile-ai/TileOPs?query=is%3Apr%20is%3Amerged%20label%3Afoundry&amp;label=Shipped%20to%20TileOPs&amp;color=8250df&amp;logo=github" alt="TileFoundry optimizations shipped to TileOPs"></a>
  <a href="https://github.com/tile-ai/TileFoundry/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://tile-ai.github.io/TileFoundry.github.io/">Documentation</a> &middot;
  <a href="https://github.com/tile-ai/TileFoundry#quick-start">Quick Start</a> &middot;
  <a href="https://github.com/tile-ai/TileFoundry/tree/main/examples">Examples</a>
</p>

## Latest News

- 08/2026 🎉: **TileFoundry 0.0.1 is on PyPI** — the first public release.
- 08/2026 📦: Four [worked examples](https://github.com/tile-ai/TileFoundry/tree/main/examples) added — Qwen3-1.7B (tilelang), Qwen3.5-35B-A3B (tilelang), MiniCPM3-4B (CuTeDSL) and granite-4.0-h-small (CUDA C) — each one a real agent run kept whole, with the decode throughput it measured.

## Quick Start

### 1 · Install

```sh
pip install tilefoundry    # needs Python 3.12 or newer
tilefoundry                # check the install: the commands an agent will ask
```

This run also needs one NVIDIA GPU, `pip install tilelang`, the published
`Qwen/Qwen3-1.7B` checkpoint on disk (3.8 GB), and a coding agent started in an
empty directory.

### 2 · Hand it the prompt

There is **no API to learn** first. Give your coding agent this, with a
checkpoint directory of your own:

```text
Get real tokens out of Qwen3-1.7B on TileFoundry, and make it fast.
Weights and config: <checkpoint directory>
Backend: tilelang.

Everything about TileFoundry is to be asked of the `tilefoundry` command -- do not
ask a person, do not go looking elsewhere. The model itself is yours to research.

Done when this runs from outside, prints the continuation, and reports a
tokens-per-second number measured over the whole generation:

    python run.py \
        --prompt "Write a detailed explanation of how a GPU executes a matrix multiplication." \
        --max-new-tokens 2048

Measure over a long generation -- 2048 new tokens, more than 2000 characters of
text. A 32-token sample is too short for the number to mean anything.
```

That is the whole input — nothing under it is written by hand.

### 3 · Come back in two hours

Claude Opus 5 at xhigh reasoning effort worked 2.1 hours and 177 tool calls
without a single interaction, and left a `run.py` behind — it prints the
continuation, and the number it measured: **612 tok/s** on one H200.

```sh
python run.py --ckpt <checkpoint directory> \
    --prompt "Write a detailed explanation of how a GPU executes a matrix multiplication." \
    --max-new-tokens 2048
```

The first run compiles the kernels — once, a few minutes.

## Where to go next

That run is kept whole in
[`examples/qwen3_1_7b-tilelang/`](https://github.com/tile-ai/TileFoundry/tree/main/examples/qwen3_1_7b-tilelang),
with three more beside it. The specifications are meant to be argued with:
[open an issue](https://github.com/tile-ai/TileFoundry/issues/new), or start from
[`docs/develop.md`](https://github.com/tile-ai/TileFoundry/blob/main/docs/develop.md).

## License

This project is licensed under the [MIT License](https://github.com/tile-ai/TileFoundry/blob/main/LICENSE).
