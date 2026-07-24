"""Torch oracle for DeepSeek-V4-Flash's real MLA attention decode path.

Faithful port of the ``Attention`` / ``Compressor`` / ``Indexer`` classes and
their free-function helpers from the official
``/data2/models/DeepSeek-V4-Flash/inference/model.py`` (827 lines) +
``kernel.py`` (536 lines), restricted to B=1 / single attention layer /
world_size=1, with checkpoint-driven weights (real FP8 checkpoint at
``/data2/models/DeepSeek-V4-Flash-FP8/``). This file owns its own weight
loader — it does not import the sibling ``hf_weights.py`` (a different agent
owns that file / moe.py / decode_step.py / torch_impl.py / test_decode_step_e2e.py
in this worktree).

Scope note: DeepSeek-V4's ``Block`` wraps ``Attention`` in Hyper-Connections
(``hc_pre``/``hc_post``, Sinkhorn mixing) and an ``attn_norm`` RMSNorm applied
*before* calling ``Attention.forward``. Hyper-Connections are block-level, not
attention-level, and are out of scope here (task is attention only); the
``x`` this module's ``AttentionRef.forward`` receives is the *already-normed*
hidden state ``Attention.forward`` itself would see.

Deliberate substitutions from the official code (custom-kernel / parallelism
removal — see task instructions: replace kernel.py kernels with equivalent
torch, drop dist/tensor-parallel since world_size=1 always):

| official                                              | this file                                                        |
|--------------------------------------------------------|-------------------------------------------------------------------|
| `world_size>1` sharding + `dist.all_reduce`             | dropped (world_size=1 always; those branches are dead code)       |
| `Linear`/`ColumnParallelLinear`/`RowParallelLinear` + `linear()` fp8/fp4 GEMM dispatch (`kernel.fp8_gemm`/`fp4_gemm`) | weights dequantized ONCE at load time (128x128 block-scale, "bf16 fallback path" semantics per task instructions) to plain bf16/fp32 tensors; plain `F.linear` at call time (no runtime activation quantization for the weight matmuls) |
| `kernel.sparse_attn` (tilelang flash-attention-style kernel: index-gathered online-softmax + attn_sink denominator term) | `sparse_attn_torch`: dense gather + masked softmax computing the identical math without the online/blocked-softmax bookkeeping (see its docstring for the exact correspondence) |
| `rotate_activation` -> `fast_hadamard_transform.hadamard_transform` (external CUDA package; **not installed** in this env) | `rotate_activation` here: explicit Sylvester +-1 Hadamard matrix matmul (mathematically identical for power-of-2 sizes) |
| `kernel.act_quant(..., inplace=True)` (real FP8 e4m3 QAT-noise fake-quant on activations, block=64) | `_fake_quant_fp8_block`: same amax/power-of-2-scale math, using a real `torch.float8_e4m3fn` dtype round-trip (exact, torch has native e4m3 support) |
| `kernel.fp4_act_quant(..., inplace=True)` (real FP4 e2m1 QAT-noise fake-quant, block=32) | `_fake_quant_fp4_block`: same amax/power-of-2-scale math, nearest-neighbor snap to the explicit 8-level E2M1 magnitude grid (torch has no elementwise FP4 dtype, only packed `float4_e2m1fn_x2`) |

Checkpoint quirks discovered empirically (see report for the verification
commands): (1) despite `config.json`'s `quantization_config.scale_fmt ==
"ue8m0"`, every `*.scale` tensor actually on disk (34123 of them, scanned
across all 46 shards) is stored as plain **F32** — already-decoded block-scale
values, not `float8_e8m0fnu`-encoded — so no e8m0 bit-decoding is needed, just
use them directly. (2) `model.safetensors.index.json` references a
`layers.<n>.attn.wo_a.scale` key for every layer that does not actually exist
in any shard (44 such phantom keys, one per layer incl. MTP) — `wo_a` is
genuinely bf16-only on disk (matches `model.py`'s own explicit
`dtype=torch.bfloat16` for `wo_a`); the loader below tolerates this by only
resolving a `.scale` sibling if it is actually present in the shard's header.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open

# ─────────────────────────────────────────────────────────────────────────
# Checkpoint locations (task-provided; FP8-real-weights dir is the source of
# truth we compute with — the fp4 dir is config/index-only reference, H200
# cannot run fp4 math).
# ─────────────────────────────────────────────────────────────────────────
FP8_CKPT_DIR = "/data2/models/DeepSeek-V4-Flash-FP8"
FP4_CKPT_DIR = "/data2/models/DeepSeek-V4-Flash"  # config-only reference

# kernel.py module-level constants (block sizes for the two activation
# fake-quant call sites; see model.py Attention/Compressor/Indexer.forward).
FP8_ACT_BLOCK_SIZE = 64
FP4_BLOCK_SIZE = 32
WEIGHT_BLOCK = 128  # fp8 weight block-scale granularity (model.py `block_size`)


# ═════════════════════════════════════════════════════════════════════════
# Pure-math helpers ported from kernel.py / model.py (no nn.Module, no
# parallelism, no fp8/fp4 GEMM dispatch — see module docstring table).
# ═════════════════════════════════════════════════════════════════════════


def _fake_quant_fp8_block(x: torch.Tensor, block_size: int, round_scale: bool = True) -> torch.Tensor:
    """Equivalent of `kernel.act_quant(x, block_size, scale_fmt="ue8m0", ...,
    inplace=True)`: per-`block_size`-group (last dim) absmax-scaled fake FP8
    (e4m3) quantize-dequantize, returned in ``x``'s original dtype.
    `round_scale=True` (this checkpoint's deployment default, scale_fmt=
    "ue8m0") rounds the scale up to a power of 2 — exactly kernel.py's
    `fast_round_scale` (bit-manipulation ceil(log2) done here via plain
    log2/ceil, which is numerically identical for finite positive inputs).
    """
    orig_dtype = x.dtype
    *lead, n = x.shape
    assert n % block_size == 0, f"{n} not divisible by block_size={block_size}"
    xf = x.float().reshape(*lead, n // block_size, block_size)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    fp8_max = 448.0
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(amax / fp8_max)))
    else:
        scale = amax / fp8_max
    q = (xf / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).to(torch.float32)
    y = (q * scale).reshape(*lead, n)
    return y.to(orig_dtype)


_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)  # E2M1 magnitude grid (convert.py FP4_TABLE, positive half)


def _fake_quant_fp4_block(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Equivalent of `kernel.fp4_act_quant(x, block_size, inplace=True)`:
    per-`block_size`-group absmax-scaled fake FP4 (e2m1) quantize-dequantize.
    torch has no elementwise FP4 dtype (only packed `float4_e2m1fn_x2`), so
    quantization here is nearest-neighbor snap to the explicit 8-level E2M1
    magnitude grid instead of a real dtype round-trip (see module docstring).
    """
    orig_dtype = x.dtype
    *lead, n = x.shape
    assert n % block_size == 0, f"{n} not divisible by block_size={block_size}"
    xf = x.float().reshape(*lead, n // block_size, block_size)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=6 * 2.0**-126)
    fp4_max = 6.0
    scale = torch.exp2(torch.ceil(torch.log2(amax / fp4_max)))
    normed = (xf / scale).clamp(-fp4_max, fp4_max)
    levels = _FP4_LEVELS.to(device=x.device, dtype=torch.float32)
    mag = normed.abs().unsqueeze(-1)                                  # [...,block,1]
    nearest = levels[torch.argmin((mag - levels).abs(), dim=-1)]      # [...,block]
    q = nearest * torch.sign(normed)
    y = (q * scale).reshape(*lead, n)
    return y.to(orig_dtype)


@lru_cache(maxsize=8)
def _hadamard_matrix(n: int, device_str: str) -> torch.Tensor:
    """Sylvester-construction +-1 Hadamard matrix of order n (n a power of
    2). Replaces `fast_hadamard_transform`'s fused CUDA kernel (package not
    installed in this environment — verified: `import fast_hadamard_transform`
    raises ModuleNotFoundError here)."""
    assert n & (n - 1) == 0, "Hadamard order must be a power of 2"
    h = torch.ones((1, 1), dtype=torch.float32, device=device_str)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Equivalent of model.py's `rotate_activation` (randomized-orientation
    +-1 Hadamard rotation along the last dim, used before FP4 fake-quant to
    spread information across dims). ``fast_hadamard_transform.hadamard_transform(x,
    scale=x.size(-1)**-0.5)`` -> explicit matmul against a constructed
    (symmetric) Hadamard matrix."""
    assert x.dtype == torch.bfloat16
    n = x.shape[-1]
    h = _hadamard_matrix(n, str(x.device))
    y = (x.float() @ h) * (n ** -0.5)
    return y.to(torch.bfloat16)


@lru_cache(maxsize=2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Verbatim port of model.py `precompute_freqs_cis` (YaRN-scaled RoPE
    frequency table, complex-exponential form)."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(lo, hi, dim):
        if lo == hi:
            hi += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Verbatim port of model.py `apply_rotary_emb`: interleaved-pairs
    (complex-multiplication) RoPE, mutating ``x`` in place via the
    ``y.copy_(x)`` at the end (``x`` is typically a basic-indexing view/slice
    of a caller's larger tensor — the write goes through). NOTE the
    interleaved-pairs convention here ((x0,x1),(x2,x3),... each rotated as one
    complex number) differs from the "rotate_half" / half-split convention
    (see report's HIR-gap section on tilefoundry's `tf.rope`)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Verbatim port of model.py `RMSNorm.forward` (weight is fp32; input
    upcast to fp32 for the reduction, result cast back to input's dtype)."""
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (weight * xf).to(dtype)


@lru_cache(maxsize=1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """Verbatim port of model.py `get_window_topk_idxs` (sliding-window
    candidate indices, causal, -1 sentinel for not-yet-existing slots)."""
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(maxsize=2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int) -> torch.Tensor:
    """Verbatim port of model.py `get_compress_topk_idxs` (dense/all-slots
    selection used when a layer has `compress_ratio` but no learned Indexer,
    e.g. ratio==128 layers). Not exercised by layer 2 (ratio==4, has an
    Indexer) but ported for generality / any-layer_id support."""
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def sparse_attn_torch(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float,
) -> torch.Tensor:
    """Equivalent of `kernel.sparse_attn` (tilelang blocked/online-softmax
    index-gather attention kernel). Computed here as one dense masked softmax
    per query position instead of the blocked kernel's running max/sum, but
    reproduces the exact same math:
      - `topk_idxs == -1` -> masked out (kernel: `acc_s` seeded to -inf for
        those slots, KV rows zeroed so their dot product contributes 0 -> stays
        -inf after the (accumulating) `T.gemm`).
      - `attn_sink` is a *denominator-only* extra logit, normalized against the
        running max of the REAL (non-sink) scores only (kernel: `scores_max`
        is reduced only from real KV blocks *before* `sum_exp[i] +=
        exp(attn_sink[i] - scores_max[i])` is added post-hoc) — replicated here
        by excluding attn_sink from the `amax` reduction too.
      - `kv` is used as both K (for the score dot product) and V (for the
        weighted sum) — the MLA "absorbed" design: a single shared latent per
        position, no separate value projection until after this call
        (`Attention.forward`'s `wo_a`/`wo_b`).
      - kernel.py pads heads to 16 for GPU-efficiency reasons only; irrelevant
        to a plain torch computation, so omitted here.

    q: [b,m,h,d]; kv: [b,n,d] (single shared latent, n_kv_heads=1);
    attn_sink: [h] f32; topk_idxs: [b,m,K] int, -1 = invalid slot.
    Returns o: [b,m,h,d] in q's dtype.
    """
    b, m, h, d = q.shape
    k = topk_idxs.shape[-1]
    valid = topk_idxs >= 0                                              # [b,m,K]
    gather_idx = topk_idxs.clamp(min=0).long()                          # [b,m,K]
    kv_f32 = kv.float()
    n = kv_f32.shape[1]
    kv_sel = torch.gather(
        kv_f32.unsqueeze(1).expand(b, m, n, d),
        2,
        gather_idx.unsqueeze(-1).expand(b, m, k, d),
    )                                                                    # [b,m,K,d]
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), kv_sel) * softmax_scale  # [b,m,h,K]
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    real_max = scores.amax(dim=-1, keepdim=True)
    real_max = torch.where(torch.isfinite(real_max), real_max, torch.zeros_like(real_max))
    exp_scores = torch.where(valid.unsqueeze(2), torch.exp(scores - real_max), torch.zeros_like(scores))
    sink = torch.exp(attn_sink.float().view(1, 1, h, 1) - real_max)      # [b,m,h,1]
    denom = exp_scores.sum(dim=-1, keepdim=True) + sink
    probs = exp_scores / denom
    out = torch.einsum("bmhk,bmkd->bmhd", probs, kv_sel)
    return out.to(q.dtype)


# ═════════════════════════════════════════════════════════════════════════
# Config: manual config.json -> ModelArgs-field mapping.
#
# generate.py's `ModelArgs(**json.load(f))` cannot actually run against this
# HF-style config.json as-is (it has keys — architectures, rope_scaling as a
# nested dict, quantization_config, ... — that are not flat ModelArgs fields;
# ModelArgs is a plain @dataclass with no **extra-kwargs tolerance). There is
# no adapter script among the 4 provided files, so this mapping is done by
# hand here, key by key, cross-checked against config.json (see report).
# ═════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AttnConfig:
    dim: int = 4096                      # hidden_size
    n_heads: int = 64                    # num_attention_heads
    q_lora_rank: int = 1024              # q_lora_rank
    head_dim: int = 512                  # head_dim
    rope_head_dim: int = 64              # qk_rope_head_dim
    norm_eps: float = 1e-6                # rms_norm_eps
    o_groups: int = 8                    # o_groups
    o_lora_rank: int = 1024              # o_lora_rank
    window_size: int = 128               # sliding_window
    compress_ratios: tuple = ()          # compress_ratios (per-layer list, len == num_hidden_layers)
    compress_rope_theta: float = 160000.0  # compress_rope_theta
    original_seq_len: int = 65536        # rope_scaling.original_max_position_embeddings
    rope_theta: float = 10000.0          # rope_theta
    rope_factor: float = 16.0            # rope_scaling.factor
    beta_fast: int = 32                  # rope_scaling.beta_fast
    beta_slow: int = 1                   # rope_scaling.beta_slow
    index_n_heads: int = 64              # index_n_heads
    index_head_dim: int = 128            # index_head_dim
    index_topk: int = 512                # index_topk

    @staticmethod
    def from_config_json(path: str) -> "AttnConfig":
        with open(path) as f:
            c = json.load(f)
        rs = c["rope_scaling"]
        assert rs["type"] == "yarn"
        return AttnConfig(
            dim=c["hidden_size"],
            n_heads=c["num_attention_heads"],
            q_lora_rank=c["q_lora_rank"],
            head_dim=c["head_dim"],
            rope_head_dim=c["qk_rope_head_dim"],
            norm_eps=c["rms_norm_eps"],
            o_groups=c["o_groups"],
            o_lora_rank=c["o_lora_rank"],
            window_size=c["sliding_window"],
            compress_ratios=tuple(c["compress_ratios"]),
            compress_rope_theta=c["compress_rope_theta"],
            original_seq_len=rs["original_max_position_embeddings"],
            rope_theta=c["rope_theta"],
            rope_factor=rs["factor"],
            beta_fast=rs["beta_fast"],
            beta_slow=rs["beta_slow"],
            index_n_heads=c["index_n_heads"],
            index_head_dim=c["index_head_dim"],
            index_topk=c["index_topk"],
        )


# ═════════════════════════════════════════════════════════════════════════
# Weight loading: safetensors index.json -> shard lookup -> block-scale
# dequant. Self-contained (no import of the sibling agent's hf_weights.py).
# ═════════════════════════════════════════════════════════════════════════


def _read_index(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _load_raw_tensors(ckpt_dir: str, keys: list, device) -> dict:
    """Loads exactly the requested keys, grouping by shard file. Silently
    skips a key if it is declared in index.json but not actually present in
    its shard header — this is a real, verified checkpoint quirk (every
    `*.attn.wo_a.scale` key across all 44 attention layers is indexed but
    does not exist in any shard; `wo_a` is genuinely bf16-only on disk)."""
    weight_map = _read_index(ckpt_dir)
    by_shard: dict = {}
    for k in keys:
        shard = weight_map.get(k)
        if shard is not None:
            by_shard.setdefault(shard, []).append(k)
    out = {}
    for shard, shard_keys in by_shard.items():
        path = os.path.join(ckpt_dir, shard)
        with safe_open(path, framework="pt", device=str(device)) as f:
            present = set(f.keys())
            for k in shard_keys:
                if k in present:
                    out[k] = f.get_tensor(k)
    return out


def _dequant_128_block(w_f8: torch.Tensor, scale_f32: torch.Tensor) -> torch.Tensor:
    """128x128 block-scale FP8 (e4m3) -> bf16 dequant. `scale_f32` is used
    as-is (already plain F32 on disk in this checkpoint, not ue8m0-encoded —
    see module docstring)."""
    rows, cols = w_f8.shape
    assert rows % WEIGHT_BLOCK == 0 and cols % WEIGHT_BLOCK == 0
    w = w_f8.to(torch.float32).reshape(rows // WEIGHT_BLOCK, WEIGHT_BLOCK, cols // WEIGHT_BLOCK, WEIGHT_BLOCK)
    s = scale_f32.to(torch.float32).reshape(rows // WEIGHT_BLOCK, 1, cols // WEIGHT_BLOCK, 1)
    return (w * s).reshape(rows, cols).to(torch.bfloat16)


def _resolve_linear(raw: dict, key: str, out_dtype: torch.dtype) -> torch.Tensor:
    """Returns a `[out,in]` weight ready for `F.linear`, dequantizing 128x128
    block-scaled FP8 if a sibling `<key>.scale` tensor is actually present
    (the "bf16 fallback path" semantics: dequantize once at load time, plain
    matmul at call time, no runtime activation quantization)."""
    w = raw[f"{key}.weight"]
    s = raw.get(f"{key}.scale")
    if s is not None:
        return _dequant_128_block(w, s).to(out_dtype)
    return w.to(out_dtype)


def load_attention_weights(layer_id: int, cfg: AttnConfig, ckpt_dir: str = FP8_CKPT_DIR, device="cuda") -> dict:
    """Loads and resolves every checkpoint tensor `AttentionRef` needs for one
    real transformer layer, keyed to match the official module hierarchy
    (`attn_sink`, `wq_a`, `q_norm.weight`, ..., `compressor.*`, `indexer.*`).
    Weight-carrying tensors are dequantized to bf16 (main projections) or f32
    (compressor's `Linear(..., dtype=torch.float32)` projections, per
    model.py) as appropriate; norm/ape/attn_sink tensors upcast to f32
    (matching model.py's own "checkpoint stores bf16, param is fp32" comments
    on RMSNorm / Compressor.ape)."""
    ratio = cfg.compress_ratios[layer_id]
    p = f"layers.{layer_id}.attn"
    keys = [
        f"{p}.attn_sink",
        f"{p}.wq_a.weight", f"{p}.wq_a.scale",
        f"{p}.q_norm.weight",
        f"{p}.wq_b.weight", f"{p}.wq_b.scale",
        f"{p}.wkv.weight", f"{p}.wkv.scale",
        f"{p}.kv_norm.weight",
        f"{p}.wo_a.weight", f"{p}.wo_a.scale",  # .scale is the known-phantom key; tolerated by _load_raw_tensors
        f"{p}.wo_b.weight", f"{p}.wo_b.scale",
    ]
    if ratio:
        keys += [
            f"{p}.compressor.ape",
            f"{p}.compressor.norm.weight",
            f"{p}.compressor.wgate.weight",
            f"{p}.compressor.wkv.weight",
        ]
    if ratio == 4:
        keys += [
            f"{p}.indexer.compressor.ape",
            f"{p}.indexer.compressor.norm.weight",
            f"{p}.indexer.compressor.wgate.weight",
            f"{p}.indexer.compressor.wkv.weight",
            f"{p}.indexer.weights_proj.weight",
            f"{p}.indexer.wq_b.weight", f"{p}.indexer.wq_b.scale",
        ]
    raw = _load_raw_tensors(ckpt_dir, keys, device)

    out = {
        "attn_sink": raw[f"{p}.attn_sink"].to(torch.float32),
        "wq_a": _resolve_linear(raw, f"{p}.wq_a", torch.bfloat16),
        "q_norm.weight": raw[f"{p}.q_norm.weight"].to(torch.float32),
        "wq_b": _resolve_linear(raw, f"{p}.wq_b", torch.bfloat16),
        "wkv": _resolve_linear(raw, f"{p}.wkv", torch.bfloat16),
        "kv_norm.weight": raw[f"{p}.kv_norm.weight"].to(torch.float32),
        "wo_a": _resolve_linear(raw, f"{p}.wo_a", torch.bfloat16),
        "wo_b": _resolve_linear(raw, f"{p}.wo_b", torch.bfloat16),
    }
    if ratio:
        out["compressor.ape"] = raw[f"{p}.compressor.ape"].to(torch.float32)
        out["compressor.norm.weight"] = raw[f"{p}.compressor.norm.weight"].to(torch.float32)
        out["compressor.wgate"] = raw[f"{p}.compressor.wgate.weight"].to(torch.float32)
        out["compressor.wkv"] = raw[f"{p}.compressor.wkv.weight"].to(torch.float32)
    if ratio == 4:
        out["indexer.compressor.ape"] = raw[f"{p}.indexer.compressor.ape"].to(torch.float32)
        out["indexer.compressor.norm.weight"] = raw[f"{p}.indexer.compressor.norm.weight"].to(torch.float32)
        out["indexer.compressor.wgate"] = raw[f"{p}.indexer.compressor.wgate.weight"].to(torch.float32)
        out["indexer.compressor.wkv"] = raw[f"{p}.indexer.compressor.wkv.weight"].to(torch.float32)
        out["indexer.weights_proj"] = raw[f"{p}.indexer.weights_proj.weight"].to(torch.bfloat16)
        out["indexer.wq_b"] = _resolve_linear(raw, f"{p}.indexer.wq_b", torch.bfloat16)
    return out


# ═════════════════════════════════════════════════════════════════════════
# Ported modules: CompressorRef / IndexerRef / AttentionRef.
#
# Structural simplification vs. official (non-numerical): official
# Compressor/Indexer lazily receive their `kv_cache`/`freqs_cis` from the
# owning Attention on first forward() call (`if self.compressor.kv_cache is
# None: ...`), because __init__ order doesn't yet know the final buffer. This
# port wires everything at construction time instead (same buffers, same
# slicing, no lazy-assignment dance) since AttentionRef.__init__ controls the
# whole construction order.
# ═════════════════════════════════════════════════════════════════════════


class CompressorRef:
    """Port of model.py `Compressor` (learned per-dimension gated-softmax
    pooling over `compress_ratio` consecutive tokens; overlapping 2x-wide
    windows when ratio==4). `rotate=True` is the Indexer's own internal
    compressor (128-dim index space: Hadamard + fake FP4 quant before
    caching); `rotate=False` is Attention's own top-level compressor
    (512-dim value space: fake FP8 quant on the non-rope dims only, rope dims
    left bf16 "for positional precision" per model.py's own comment)."""

    def __init__(
        self, *, compress_ratio: int, head_dim: int, rope_head_dim: int, eps: float, rotate: bool,
        ape: torch.Tensor, norm_weight: torch.Tensor, wgate_weight: torch.Tensor, wkv_weight: torch.Tensor,
        kv_cache: torch.Tensor, max_batch_size: int, device,
    ):
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.eps = eps
        coff = 1 + self.overlap
        self.ape = ape
        self.norm_weight = norm_weight
        self.wgate_weight = wgate_weight
        self.wkv_weight = wkv_weight
        self.kv_cache = kv_cache  # [max_batch_size, cache_len, head_dim] bf16 (caller-owned buffer/view)
        self.kv_state = torch.zeros(max_batch_size, coff * compress_ratio, coff * head_dim, dtype=torch.float32, device=device)
        self.score_state = torch.full((max_batch_size, coff * compress_ratio, coff * head_dim), float("-inf"), dtype=torch.float32, device=device)
        self.freqs_cis: Optional[torch.Tensor] = None  # wired by AttentionRef before first forward()

    def overlap_transform(self, tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        b, s, _, _ = tensor.shape
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int) -> Optional[torch.Tensor]:
        bsz, seqlen, _ = x.shape
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x = x.float()
        kv = F.linear(x, self.wkv_weight)
        score = F.linear(x, self.wgate_weight)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio:cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio:cutoff] + self.ape
            if remainder > 0:
                kv, tail_kv = kv.split([cutoff, remainder], dim=1)
                self.kv_state[:bsz, offset:offset + remainder] = tail_kv
                self.score_state[:bsz, offset:offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0.0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        kv = _rms_norm(kv.to(dtype), self.norm_weight, self.eps)
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
            kv = _fake_quant_fp4_block(kv, FP4_BLOCK_SIZE)
        else:
            kv[..., :-rd] = _fake_quant_fp8_block(kv[..., :-rd], FP8_ACT_BLOCK_SIZE)
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class IndexerRef:
    """Port of model.py `Indexer` (learned top-k compressed-KV-position
    selection; owns its own `CompressorRef(rotate=True)` operating in a
    separate 128-dim "index" space, distinct from Attention's own 512-dim
    value-space compressor)."""

    def __init__(
        self, *, n_heads: int, head_dim: int, rope_head_dim: int, index_topk: int,
        wq_b_weight: torch.Tensor, weights_proj_weight: torch.Tensor, softmax_scale: float,
        compressor: CompressorRef, kv_cache: torch.Tensor,
    ):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.wq_b_weight = wq_b_weight
        self.weights_proj_weight = weights_proj_weight
        self.softmax_scale = softmax_scale
        self.compress_ratio = compressor.compress_ratio
        self.compressor = compressor
        self.kv_cache = kv_cache  # [max_batch_size, max_seq_len//ratio, head_dim=128] bf16
        self.freqs_cis: Optional[torch.Tensor] = None  # wired by AttentionRef before first forward()

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen

        q = F.linear(qr, self.wq_b_weight)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        q = _fake_quant_fp4_block(q, FP4_BLOCK_SIZE)
        self.compressor.forward(x, start_pos)  # side effect only: advances the index-space compressed-kv cache
        weights = F.linear(x, self.weights_proj_weight) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1) >= (
                torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            index_score = index_score + torch.where(mask, float("-inf"), 0.0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= (torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio)
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


class AttentionRef:
    """Port of model.py `Attention` (Multi-head Latent Attention: low-rank Q
    (wq_a -> q_norm -> wq_b), single shared 512-dim KV latent (wkv -> kv_norm,
    MQA-style: n_kv_heads==1), sliding-window + learned-indexer KV-compression
    sparsity, grouped low-rank O projection (wo_a -> wo_b)). See module
    docstring for the kernel/parallelism substitutions."""

    def __init__(self, *, layer_id: int, cfg: AttnConfig, weights: dict, device, max_seq_len: int, max_batch_size: int = 1):
        self.layer_id = layer_id
        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.o_lora_rank = cfg.o_lora_rank
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.o_groups
        self.window_size = cfg.window_size
        self.compress_ratio = cfg.compress_ratios[layer_id]
        self.eps = cfg.norm_eps
        self.max_batch_size = max_batch_size

        self.attn_sink = weights["attn_sink"]
        self.wq_a_weight = weights["wq_a"]
        self.q_norm_weight = weights["q_norm.weight"]
        self.wq_b_weight = weights["wq_b"]
        self.wkv_weight = weights["wkv"]
        self.kv_norm_weight = weights["kv_norm.weight"]
        self.wo_a_weight = weights["wo_a"]
        self.wo_b_weight = weights["wo_b"]
        self.softmax_scale = self.head_dim ** -0.5

        kv_cache_size = self.window_size + (max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.kv_cache = torch.zeros(max_batch_size, kv_cache_size, self.head_dim, dtype=torch.bfloat16, device=device)

        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = CompressorRef(
                compress_ratio=self.compress_ratio, head_dim=self.head_dim, rope_head_dim=self.rope_head_dim,
                eps=self.eps, rotate=False,
                ape=weights["compressor.ape"], norm_weight=weights["compressor.norm.weight"],
                wgate_weight=weights["compressor.wgate"], wkv_weight=weights["compressor.wkv"],
                kv_cache=self.kv_cache[:, self.window_size:], max_batch_size=max_batch_size, device=device,
            )
            if self.compress_ratio == 4:
                index_kv_cache = torch.zeros(
                    max_batch_size, max_seq_len // self.compress_ratio, cfg.index_head_dim,
                    dtype=torch.bfloat16, device=device,
                )
                index_compressor = CompressorRef(
                    compress_ratio=self.compress_ratio, head_dim=cfg.index_head_dim, rope_head_dim=self.rope_head_dim,
                    eps=self.eps, rotate=True,
                    ape=weights["indexer.compressor.ape"], norm_weight=weights["indexer.compressor.norm.weight"],
                    wgate_weight=weights["indexer.compressor.wgate"], wkv_weight=weights["indexer.compressor.wkv"],
                    kv_cache=index_kv_cache, max_batch_size=max_batch_size, device=device,
                )
                self.indexer = IndexerRef(
                    n_heads=cfg.index_n_heads, head_dim=cfg.index_head_dim, rope_head_dim=self.rope_head_dim,
                    index_topk=cfg.index_topk,
                    wq_b_weight=weights["indexer.wq_b"], weights_proj_weight=weights["indexer.weights_proj"],
                    softmax_scale=cfg.index_head_dim ** -0.5,
                    compressor=index_compressor, kv_cache=index_kv_cache,
                )

        if self.compress_ratio:
            original_seq_len, rope_theta = cfg.original_seq_len, cfg.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, cfg.rope_theta
        self.freqs_cis = precompute_freqs_cis(
            self.rope_head_dim, max_seq_len + 1, original_seq_len, rope_theta,
            cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
        ).to(device)
        if self.compressor is not None:
            self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis
            self.indexer.compressor.freqs_cis = self.freqs_cis  # official wires this lazily inside Indexer.forward; done eagerly here (see class docstring)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # q
        qr = q = _rms_norm(F.linear(x, self.wq_a_weight), self.q_norm_weight, self.eps)
        q = F.linear(q, self.wq_b_weight).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # win kv & topk_idxs
        kv = F.linear(x, self.wkv_weight)
        kv = _rms_norm(kv, self.kv_norm_weight, self.eps)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        kv[..., :-rd] = _fake_quant_fp8_block(kv[..., :-rd], FP8_ACT_BLOCK_SIZE)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)
        if self.compress_ratio:
            offset = kv.shape[1] if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer.forward(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset).to(x.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                kv_compress = self.compressor.forward(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn_torch(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor.forward(x, start_pos)
            o = sparse_attn_torch(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a_weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return F.linear(o.flatten(2), self.wo_b_weight)


def build_attention_ref(
    layer_id: int, device="cuda", max_seq_len: int = 4096, max_batch_size: int = 1,
    ckpt_dir: str = FP8_CKPT_DIR, config_path: Optional[str] = None,
) -> AttentionRef:
    """Convenience one-shot builder: config + weights + module for one real
    transformer layer's attention."""
    if config_path is None:
        config_path = os.path.join(ckpt_dir, "config.json")
    cfg = AttnConfig.from_config_json(config_path)
    weights = load_attention_weights(layer_id, cfg, ckpt_dir=ckpt_dir, device=device)
    return AttentionRef(layer_id=layer_id, cfg=cfg, weights=weights, device=device, max_seq_len=max_seq_len, max_batch_size=max_batch_size)


__all__ = [
    "FP8_CKPT_DIR",
    "FP4_CKPT_DIR",
    "AttnConfig",
    "CompressorRef",
    "IndexerRef",
    "AttentionRef",
    "load_attention_weights",
    "build_attention_ref",
    "sparse_attn_torch",
    "apply_rotary_emb",
    "precompute_freqs_cis",
    "get_window_topk_idxs",
    "get_compress_topk_idxs",
    "rotate_activation",
]
