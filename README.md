<p align="center">
  <img src="https://github.com/user-attachments/assets/3e29ee3e-2fae-4243-ba73-0efc04ac7645" alt="TileFoundry" width="100%">
</p>

---

<p align="center">
  <a href="https://pypi.org/project/tilefoundry/"><img src="https://img.shields.io/pypi/v/tilefoundry.svg" alt="PyPI"></a>
  <a href="https://codecov.io/gh/tile-ai/TileFoundry"><img src="https://codecov.io/gh/tile-ai/TileFoundry/branch/main/graph/badge.svg" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/status-early%20development-orange" alt="Status: early development">
  <a href="https://github.com/tile-ai/TileFoundry/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://tile-ai.github.io/TileFoundry.github.io/">Documentation</a> &middot;
  <a href="https://github.com/tile-ai/TileFoundry#installation">Installation</a> &middot;
  <a href="https://github.com/tile-ai/TileFoundry/tree/main/examples">Examples</a>
</p>

**TileFoundry** is a tile-based, agentic platform for automatic high-performance program generation across hardware.

## Latest News

- 08/2026 🎉: **TileFoundry 0.0.1 is on PyPI** — the first public release.
- 08/2026 📦: Four [worked examples](https://github.com/tile-ai/TileFoundry/tree/main/examples) added — Qwen3-1.7B (tilelang), Qwen3.5-35B-A3B (tilelang), MiniCPM3-4B (CuTeDSL) and granite-4.0-h-small (CUDA C) — each one a real agent run kept whole, with the decode throughput it measured.

## Installation

TileFoundry needs Python 3.12 or newer.

```sh
pip install tilefoundry
```

Check the install — it prints the commands an agent will ask:

```sh
tilefoundry
```

Running a model an agent generates additionally needs one NVIDIA GPU and the
checkpoint already on disk.

## Quick Start

There is **no API to learn** first. Give your coding agent this, with a checkpoint
directory of your own:

```text
Get real tokens out of Qwen3-1.7B on TileFoundry, and make it fast.
Weights and config: <checkpoint directory>
Backend: tilelang.

Everything about TileFoundry is to be asked of the `tilefoundry` command -- do not
ask a person, do not go looking elsewhere. The model itself is yours to research.

Done when this runs from outside, prints the continuation, and reports a
tokens-per-second number measured over the whole generation:

    python run.py --prompt "Write a detailed explanation of how a GPU executes a matrix multiplication." --max-new-tokens 2048

Measure over a long generation -- 2048 new tokens, more than 2000 characters of
text. A 32-token sample is too short for the number to mean anything.
```

That is the whole input — nothing under it is written by hand.

Claude Opus 5 at xhigh reasoning effort ran this prompt for 2.1 hours with **no
interaction**, and reached **612 tok/s** on one H200. What it wrote is
[`examples/qwen3_1_7b-tilelang/`](https://github.com/tile-ai/TileFoundry/tree/main/examples/qwen3_1_7b-tilelang).

## License

This project is licensed under the [MIT License](https://github.com/tile-ai/TileFoundry/blob/main/LICENSE).
