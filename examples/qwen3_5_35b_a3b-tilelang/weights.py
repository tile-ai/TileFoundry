"""Binding the published checkpoint to what `model.py` declares.

`Module.load(resource)` reads a weight by its canonical name and expects it
already in the declared shape; the raw->declared transform lives in the
per-weight **converters**, which only `Module.prepare` runs (runtime §1.1.2).
Prepare writes a directory, and for this model at the declared f32 that
directory is ~280 GB of transposes nothing reads twice.

So `HFResource` below is `prepare` without the directory: same walk, same
converters, same one source of truth, but the converter runs on demand, on the
GPU, at the dtype the runtime actually wants. Two consequences worth stating:

* The converters in `model.py` are the *only* place the raw->declared transform
  is written. A second copy here -- "load the checkpoint and transpose it" -- is
  the thing that goes stale the first time a converter changes, so there isn't
  one. `evaluate(conv, *raws)` is how this file gets its answer, exactly as
  `_prepare_into` does.
* The runtime dtype is **bf16 for the matmul weights, f32 for everything else**.
  bf16 is what the checkpoint stores, so nothing is lost: the reference declares
  f32 and `SafetensorsResource(dtype=torch.float32)` would honour it, but 35 G
  parameters at f32 is 140 GB and the values in them are bf16 either way. The
  gammas, `A_log`, `dt_bias` and the convolution kernel stay f32 -- they are
  ~300 K elements in total, they are added to and exponentiated rather than
  contracted over, and `1 + w` in bf16 would throw away the low bits of exactly
  the correction the converter exists to apply.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

import model as sem  # noqa: E402
from config import CKPT, REAL  # noqa: E402
from tilefoundry.evaluator import evaluate  # noqa: E402
from tilefoundry.runtime import SafetensorsResource  # noqa: E402

#: Weights that stay f32 whatever the matmuls run at. Small, and read by
#: something other than a contraction: a scale, a bias, an exponent, a kernel.
F32_WEIGHTS = frozenset({
    "gamma_in", "gamma_q", "gamma_k", "gamma_gdn", "gamma_post", "gamma_final",
    "a_log", "dt_bias", "conv_w",
})


class HFResource:
    """A `RuntimeResource` over the raw checkpoint that answers canonical names.

    Implements the protocol of runtime §1.5 -- `load` / `load_group` / `subtree`
    -- so both a `Module` and its `RuntimeModule` twin can be loaded from it.
    """

    #: (raw resource, name, tensor) of the last raw read, shared across subtree
    #: views because a view is a new object per child. See `_fetch`.
    _last = None

    def __init__(
        self,
        node,
        raw,
        *,
        dtype: torch.dtype = torch.bfloat16,
        report=None,
        _converters=None,
    ) -> None:
        self._node = node
        self._raw = raw
        self._dtype = dtype
        self._report = report
        self._converters = (
            _converter_map(node) if _converters is None else _converters
        )

    def _fetch(self, name: str) -> torch.Tensor:
        """One raw tensor, with a one-entry cache.

        `w_gate` and `w_up` are two converters over the *same* raw tensor
        (`mlp.experts.gate_up_proj`, 1.07 GB), and they are asked for
        back to back. Without this the checkpoint is read twice per layer for
        them -- 43 GB of the 229 s first measured. One entry is enough because
        the reuse is always immediate; holding more would pin gigabytes.
        """
        cached = HFResource._last
        if cached is not None and cached[0] is self._raw and cached[1] == name:
            return cached[2]
        value = self._raw.load(name)
        HFResource._last = (self._raw, name, value)
        return value

    def load(self, name: str) -> torch.Tensor:
        conv = self._converters.get(name)
        want = torch.float32 if name in F32_WEIGHTS else self._dtype
        if conv is None:
            value = self._fetch(name).to(want)
        else:
            # The converter's parameters are raw-checkpoint names; the alias
            # table maps them. Feed it at `want` so a `1 + w` lands in the
            # dtype it will be used at rather than being rounded twice.
            raws = [self._fetch(p.name).to(want) for p in conv.params]
            # `evaluate` materialises the converter's *declared* return dtype,
            # which is `config.dt` (f32) -- so feeding it bf16 does not keep the
            # result bf16 and the cast has to happen on the way out too. The f32
            # transient is 2x the final tensor; the largest is `w_gate` at 1.07
            # GB, which is why this is a cast and not a complaint.
            value = evaluate(conv, *raws).to(want)
        value = value.contiguous()
        if self._report is not None:
            self._report(name, value)
        return value

    def load_group(self, name: str):
        # No weight of this model is a one-to-many alias: the published
        # checkpoint already stores the 256 experts as one fused tensor.
        return None

    def subtree(self, seg: str) -> "HFResource":
        for child in self._node.modules:
            if child.name == seg:
                return HFResource(
                    child,
                    self._raw.subtree(seg),
                    dtype=self._dtype,
                    report=self._report,
                )
        raise KeyError(f"{self._node.name!r} has no child module {seg!r}")


def _converter_map(node) -> dict:
    """This node's per-weight converters, unioned over its functions.

    Same rule as `Module._prepare_into`: a converter is registered on the
    function that declares the weight, and one weight may be declared by
    several functions of one Module, so the map is per Module.
    """
    found: dict[str, object] = {}
    for fn in node.functions:
        for weight_name, conv in getattr(fn, "converters", ()):
            found[weight_name] = conv
    return found


def raw_resource(ckpt=CKPT, cfg=REAL, *, device="cuda"):
    """The published checkpoint, alias table attached, nothing converted yet."""
    return SafetensorsResource(str(ckpt), device=device, alias=sem.hf_alias(cfg))


def decoder_resource(
    node=None, ckpt=CKPT, cfg=REAL, *, device="cuda", dtype=torch.bfloat16,
    verbose=False,
):
    """What `Qwen3_5Decoder.load(...)` / the twin's `load(...)` reads.

    *node* is the authored decoder the resource walks -- the published one by
    default, or a truncated `model.build(cfg)["Qwen3_5Decoder"]` for a short loop.
    It has to match *cfg*, since the alias table is generated per layer index.
    """
    node = sem.Qwen3_5Decoder if node is None else node
    total = {"bytes": 0, "n": 0, "t0": time.perf_counter()}

    def report(name, value):
        total["bytes"] += value.numel() * value.element_size()
        total["n"] += 1
        if verbose and total["n"] % 200 == 0:
            gb = total["bytes"] / 1e9
            dt = time.perf_counter() - total["t0"]
            print(
                f"  loaded {total['n']:5d} tensors  {gb:6.2f} GB  "
                f"{dt:5.1f}s  ({gb / max(dt, 1e-9):.2f} GB/s)",
                flush=True,
            )

    return HFResource(
        node,
        raw_resource(ckpt, cfg, device=device),
        dtype=dtype,
        report=report,
    ), total


def leaf_resource(
    node, kind, layer_index, ckpt=CKPT, cfg=REAL, *, device="cuda",
    dtype=torch.float32,
):
    """One leaf Module's own weights, for a per-leaf comparison.

    `tilefoundry check model.py:Qwen3_5LinearAttention` stands over the bare
    mixer, so its resource has no `layerN.mixer` prefix to qualify against;
    `model.leaf_alias` is `hf_alias` re-rooted for that case. Defaults to f32
    because a leaf check wants the tight comparison, not the fast one.
    """
    raw = SafetensorsResource(
        str(ckpt), device=device, alias=sem.leaf_alias(kind, layer_index, cfg)
    )
    return HFResource(node, raw, dtype=dtype)


def prepare_leaf(node, kind, layer_index, out_dir, ckpt=CKPT, cfg=REAL, *, device="cuda"):
    """Write one leaf's declared weights to *out_dir*, for `check --ckpt DIR`.

    `tilefoundry check` builds its resource as `SafetensorsResource(ckpt)` with
    no alias and no dtype, so `--ckpt` has to be a **prepared** directory: clean
    dot-joined canonical names, at the declared dtype. `Module.prepare` is what
    writes one.

    The `dtype=torch.float32` here is load-bearing, not tidiness. `prepare`
    validates every weight's dtype against the declaration strictly, and this
    checkpoint is **mixed**: `linear_attn.A_log` and `linear_attn.norm.weight`
    are stored f32 while `dt_bias` and every projection are bf16. A weight with
    no converter is validated against its raw value unchanged, so `dt_bias` at
    bf16 against a declared f32 is refused outright:

        ValueError: Module 'Qwen3_5LinearAttention': raw weight 'dt_bias' has
        dtype torch.bfloat16, declared FloatDType(name='f32', ...)

    Reading the whole checkpoint as f32 is exactly what runtime §1.5 says the
    `dtype` argument is for -- "what lets one checkpoint serve modules that
    declare a different precision than it holds" -- and it is cheaper than a
    per-weight converter whose only work is a cast.
    """
    raw = SafetensorsResource(
        str(ckpt),
        device=device,
        alias=sem.leaf_alias(kind, layer_index, cfg),
        dtype=torch.float32,
    )
    node.prepare(raw, str(out_dir), device=device)
    return out_dir


def rope_caches(cfg=REAL, *, device="cuda", dtype=torch.float32):
    """`cos_cache` / `sin_cache`, both `(max_ctx, rotary_dim)`.

    Built the way `Qwen3_5MoeTextRotaryEmbedding` builds them, at
    `partial_rotary_factor` of the head:

        inv_freq[j] = theta ** (-2j / rotary_dim),  j < rotary_dim / 2
        row p       = cos(concat(p * inv_freq, p * inv_freq))

    The published rope is `mrope_interleaved` with sections (11, 11, 10). That
    matters for image and video tokens, whose three position streams differ. For
    a text-only decode all three carry the same position, so
    `apply_interleaved_mrope` selects between three identical values and the
    result is the ordinary 1-D rope -- which is why this returns one cache and
    not three. Text is the whole scope here; a vision tower would need the other
    two streams and is out of scope by the same decision that drops
    `model.visual.*`.
    """
    half = cfg.rotary_dim // 2
    j = torch.arange(0, half, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (cfg.rope_theta ** (2.0 * j / cfg.rotary_dim))
    pos = torch.arange(cfg.max_ctx, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)                      # (max_ctx, half)
    emb = torch.cat((freqs, freqs), dim=-1)                 # (max_ctx, rotary_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def tokenizer(ckpt=CKPT):
    """The published tokenizer. `tokenizers` directly rather than
    `transformers.AutoTokenizer`, which would want the whole model class."""
    from tokenizers import Tokenizer  # noqa: PLC0415

    if ckpt is None:
        raise SystemExit("where the checkpoint is has to be said: pass --ckpt")
    return Tokenizer.from_file(os.path.join(str(ckpt), "tokenizer.json"))


__all__ = [
    "F32_WEIGHTS",
    "HFResource",
    "decoder_resource",
    "leaf_resource",
    "raw_resource",
    "rope_caches",
    "tokenizer",
]
