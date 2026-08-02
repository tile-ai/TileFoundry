"""Qwen3.5-35B-A3B's published shape configuration, as `model.py` reads it.

Why this file exists: `tilefoundry models qwen3_5_35b_a3b --source` prints a
`model.py` whose first import is

    from tests.models.qwen3_5_35b_a3b.config import REAL as config

and that package does not ship with the installation -- `tilefoundry models`
answers from a precomputed `catalog.json`, so the CLI never imports the model it
prints. Since `tilefoundry check` / `analyze` execute a source file with
`runpy.run_path`, an authored file has to be self-contained (parser §2.7 says so
outright: "every name its class bodies read MUST resolve within that file").

So the config object is reconstructed here. Nothing in it is guessed:

* every **field value** is either published in the checkpoint's `config.json`,
  or read back out of the signatures `tilefoundry models qwen3_5_35b_a3b`
  prints. The catalog is the authority for the fixture's own choices --
  `cos_cache: Tensor[(4096, 64), "f32"]` is where `max_ctx = 4096`,
  `rotary_dim = 64` and `dt = "f32"` come from, and
  `conv_state: Tensor[(1, 8192, 3), "f32"]` is where `gdn_conv_context = 3`
  comes from.
* every **field name** is read off the shipped `model.py` body, which names all
  of them.

The migrate tutorial's third rule applies throughout: a published dimension is
published, never derived. `head_dim` is 256 while `hidden / num_heads` is 128.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

#: Where the real weights live, if the environment says. There is deliberately no
#: fallback: a path that exists on one machine is not a setting, and defaulting to
#: it behaves like a hardcoded path for everybody else. `run.py --ckpt` is the
#: other way to state it, and it is required.
_ENV_CKPT = os.environ.get("QWEN35_CKPT")
CKPT = Path(_ENV_CKPT) if _ENV_CKPT else None


@dataclass(frozen=True)
class Qwen3_5Config:
    """One structural configuration of the published text decoder."""

    # ---- global ---------------------------------------------------------
    hidden: int
    vocab: int
    rms_eps: float
    dt: str
    #: The dtype of the weights a matmul contracts over. Separate from `dt`
    #: because a declared type is now checked against the tensor that arrives
    #: (`RuntimeModule.load` -> `_validate_declared`), and the two differ: the
    #: checkpoint stores those weights bf16 and 35 G parameters at f32 is 140 GB,
    #: while activations and accumulators stay `dt`. The nine weights listed in
    #: `weights.F32_WEIGHTS` -- gammas, `a_log`, `dt_bias`, `conv_w` -- keep `dt`:
    #: they are added to and exponentiated rather than contracted over.
    dt_w: str
    layer_types: tuple[str, ...]
    #: One row per position a step may be decoded at. A fixture choice, not the
    #: model's `max_position_embeddings` (262144): the rope caches are
    #: materialised, and `ctx_len`'s exclusive upper bound is this.
    max_ctx: int
    rope_theta: float

    # ---- full attention -------------------------------------------------
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    rotary_dim: int

    # ---- gated delta net ------------------------------------------------
    gdn_n_k_heads: int
    gdn_n_v_heads: int
    gdn_head_k_dim: int
    gdn_head_v_dim: int
    gdn_conv_kernel: int

    # ---- mixture of experts ---------------------------------------------
    n_experts: int
    top_k: int
    moe_intermediate: int
    shared_intermediate: int

    # ---- derived (arithmetic on published fields, never a substitute) ----
    @property
    def pass_dim(self) -> int:
        """The part of a head RoPE does not touch."""
        return self.head_dim - self.rotary_dim

    @property
    def gqa_group(self) -> int:
        return self.n_q_heads // self.n_kv_heads

    @property
    def gdn_key_dim(self) -> int:
        return self.gdn_n_k_heads * self.gdn_head_k_dim

    @property
    def gdn_value_dim(self) -> int:
        return self.gdn_n_v_heads * self.gdn_head_v_dim

    @property
    def gdn_conv_dim(self) -> int:
        """q and k (one per key head) plus v (one per value head)."""
        return 2 * self.gdn_key_dim + self.gdn_value_dim

    @property
    def gdn_conv_context(self) -> int:
        """The prior columns a step's window carries; the step supplies the last."""
        return self.gdn_conv_kernel - 1

    @property
    def gdn_v_per_k(self) -> int:
        return self.gdn_n_v_heads // self.gdn_n_k_heads

    @property
    def n_layers(self) -> int:
        return len(self.layer_types)

    @property
    def attn_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    def replace(self, **changes) -> "Qwen3_5Config":
        from dataclasses import replace as _replace

        return _replace(self, **changes)


_CYCLE = ("linear_attention",) * 3 + ("full_attention",)

#: The published configuration.
#:
#: `dt = "f32"` is what this implementation computes in: activations, the
#: accumulators inside every tilelang kernel, and the nine small weights that are
#: added to or exponentiated rather than contracted over. `dt_w = "bf16"` is what
#: the contracted weights are, which is what the checkpoint stores.
#:
#: These were one field when this was written, against a guarantee that has since
#: been withdrawn: `SafetensorsResource(dtype=...)` widened bf16 to the declared
#: f32 on the read side, and runtime §1.5 said so in as many words -- "this is
#: what lets one checkpoint serve modules that declare a different precision than
#: it holds". `#48` deleted the flag, the sentence, and the permissive `load`, and
#: `_validate_declared` now refuses a tensor whose dtype is not the declared one.
#: So the two precisions have to be declared separately, which is what splitting
#: this field does. No value changes: bf16 stored, bf16 declared, f32 accumulated.
REAL = Qwen3_5Config(
    hidden=2048,
    vocab=248320,
    rms_eps=1e-6,
    dt="f32",
    dt_w="bf16",
    layer_types=_CYCLE * 10,
    max_ctx=4096,
    rope_theta=1.0e7,
    n_q_heads=16,
    n_kv_heads=2,
    head_dim=256,
    rotary_dim=64,
    gdn_n_k_heads=16,
    gdn_n_v_heads=32,
    gdn_head_k_dim=128,
    gdn_head_v_dim=128,
    gdn_conv_kernel=4,
    n_experts=256,
    top_k=8,
    moe_intermediate=512,
    shared_intermediate=512,
)


def from_checkpoint(
    ckpt: Path | str = CKPT, *, dt: str = "f32", dt_w: str = "bf16", max_ctx: int = 4096
):
    """`REAL`, but with every field it can read taken from the checkpoint itself.

    Used to assert that `REAL` above is not drifting from the published file --
    `run.py --verify-config` calls it.
    """
    text = json.loads((Path(ckpt) / "config.json").read_text())["text_config"]
    rope = text["rope_parameters"]
    return Qwen3_5Config(
        hidden=text["hidden_size"],
        vocab=text["vocab_size"],
        rms_eps=text["rms_norm_eps"],
        dt=dt,
        dt_w=dt_w,
        layer_types=tuple(text["layer_types"]),
        max_ctx=max_ctx,
        rope_theta=rope["rope_theta"],
        n_q_heads=text["num_attention_heads"],
        n_kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        rotary_dim=int(text["head_dim"] * rope["partial_rotary_factor"]),
        gdn_n_k_heads=text["linear_num_key_heads"],
        gdn_n_v_heads=text["linear_num_value_heads"],
        gdn_head_k_dim=text["linear_key_head_dim"],
        gdn_head_v_dim=text["linear_value_head_dim"],
        gdn_conv_kernel=text["linear_conv_kernel_dim"],
        n_experts=text["num_experts"],
        top_k=text["num_experts_per_tok"],
        moe_intermediate=text["moe_intermediate_size"],
        shared_intermediate=text["shared_expert_intermediate_size"],
    )


#: A cut-down stack, for turning the loop over quickly on real weights. Same
#: kernels, same shapes, fewer layers -- `layer_types` is the only field that
#: moves, so a Module built from it is the published one with the tail removed.
def truncated(n_layers: int, base: Qwen3_5Config = REAL) -> Qwen3_5Config:
    if n_layers > base.n_layers:
        raise ValueError(f"{n_layers} > the published {base.n_layers}")
    return base.replace(layer_types=base.layer_types[:n_layers])


__all__ = ["CKPT", "REAL", "Qwen3_5Config", "from_checkpoint", "truncated"]
