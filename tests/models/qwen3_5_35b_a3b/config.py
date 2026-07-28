"""Qwen3.5-35B-A3B (text tower) dimensions, and the Hugging Face oracle every
test in this package compares against.

The shape is not written out here. It is read from ``text_config.json``, which
is the ``text_config`` subtree of the published ``config.json`` verbatim, and
the provenance of that file -- where it came from, at which revision, and a
digest of the fields this package actually depends on -- is stated below and
asserted by ``test_provenance.py``. A model fixture that claims a checkpoint's
name has to be checkable against that checkpoint, and a number retyped by hand
is not checkable: it agrees with the source until someone edits it.

The published model is *multimodal* -- ``Qwen3_5MoeForConditionalGeneration``
with a ``vision_config`` alongside the ``text_config``. Only the text tower is
in scope, so only the ``text_config`` subtree is carried, and ``model_type``
inside it (``qwen3_5_moe_text``) is what names the architecture the tests build.

What the architecture is, in the two shapes that matter here:

- ``layer_types`` alternates ``linear_attention`` and ``full_attention`` on a
  strict period of ``full_attention_interval`` = 4 (three linear, then one
  full). The two are different token mixers with different state, so they are
  separate boundaries, not one boundary at two settings.
- a ``linear_attention`` layer is a Gated DeltaNet: its state is a short causal
  convolution window and a fixed-size recurrent matrix, neither of which grows
  with the context.
- a ``full_attention`` layer is GQA (16 query heads over 2 key/value heads at
  ``head_dim`` 256), with two things Qwen3 did not have: a *partial* rotary
  embedding (``partial_rotary_factor`` 0.25, so 64 of the 256 head dims rotate)
  and an output gate (``attn_output_gate``).
- every layer's MLP is the same 256-expert top-8 MoE block, plus a shared
  expert that every token goes through, itself gated by a scalar per token.

Weights are synthetic and seeded (``oracle.randomised``). No checkpoint is
downloaded or read: the tests run offline, and the same call twice returns the
same weights, so a disagreement is a disagreement about the compiler.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tests.models import decode_oracle as oracle

# ── Provenance of text_config.json ──────────────────────────────────────
#: Where the published configuration was fetched from.
SOURCE_URL = "https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json"

#: The commit on that repository the fetch resolved to. A branch name alone
#: would make this file's provenance move under it.
SOURCE_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"

#: sha256 of the *whole* published ``config.json`` as fetched -- the multimodal
#: one, including ``vision_config``. ``text_config.json`` next to this module is
#: its ``text_config`` subtree re-serialised (sorted keys, two-space indent), so
#: it does not carry this digest itself; what ties the two together is
#: ``FIELDS_DIGEST`` below, over the fields this package depends on.
SOURCE_SHA256 = "5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944"

#: sha256 of ``text_config.json`` as it is checked in beside this module. The
#: re-serialisation is what makes the digest above not apply to it, and a subtree
#: nobody hashed is a subtree that can be edited without anything noticing.
LOCAL_SHA256 = "4893a05b080e069c5e70debe1477ef0085d5427b7cde6e71bb9e08bccda44117"

#: The path the shape is read from.
TEXT_CONFIG_PATH = Path(__file__).parent / "text_config.json"

#: The fields this package's kernels and oracles are shaped by. Named rather
#: than digesting the whole file, because the whole file also carries things no
#: test here reads (``eos_token_id``, ``initializer_range``, ...) and a digest
#: that moved when those moved would report a change nobody made.
DEPENDED_FIELDS: tuple[str, ...] = (
    "attention_bias",
    "attn_output_gate",
    "full_attention_interval",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "layer_types",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "model_type",
    "moe_intermediate_size",
    "mtp_num_hidden_layers",
    "num_attention_heads",
    "num_experts",
    "num_experts_per_tok",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_parameters",
    "shared_expert_intermediate_size",
    "vocab_size",
)

#: sha256 over the depended-on fields, canonically serialised. Bumping a shape
#: means bumping this, which is the point: the change becomes a visible edit to
#: the fixture's stated provenance rather than a silent retype.
FIELDS_DIGEST = "225208b1d25b32fd4d151112bcac69f08ffc6591fd258c127a92bb40296dd910"


def published_text_config() -> dict:
    """The published ``text_config`` subtree, as data."""
    return json.loads(TEXT_CONFIG_PATH.read_text())


def fields_digest(text_config: dict | None = None) -> str:
    """sha256 over ``DEPENDED_FIELDS`` of *text_config*, canonically serialised."""
    config = published_text_config() if text_config is None else text_config
    selected = {name: config[name] for name in DEPENDED_FIELDS}
    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


_PUBLISHED = published_text_config()


# ── The shape, read from the published configuration ────────────────────


@dataclass(frozen=True)
class Qwen35Shape:
    """The dimensions of one layer of each type, plus the MoE block they share.

    Derived from the published configuration by ``from_published``; nothing here
    is a literal, so no field can disagree with the source.
    """

    hidden: int
    vocab: int
    rms_eps: float
    n_layers: int
    layer_types: tuple[str, ...]
    full_attention_interval: int

    # full attention
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    rope_theta: float
    partial_rotary_factor: float
    attention_bias: bool
    max_pos: int

    # gated delta net (linear attention)
    gdn_n_k_heads: int
    gdn_head_k_dim: int
    gdn_n_v_heads: int
    gdn_head_v_dim: int
    gdn_conv_kernel: int

    # moe
    n_experts: int
    top_k: int
    moe_intermediate: int
    shared_intermediate: int

    # multi-token prediction: carried because the published model states it,
    # not because anything here executes it (see test_provenance.py).
    mtp_n_layers: int

    #: The largest context the full-attention kernels are authored for. Not a
    #: published field: ``max_position_embeddings`` is 262144, and a decode
    #: kernel authored to that envelope is the same kernel as one authored to a
    #: smaller one -- what the envelope has to be is larger than any context a
    #: test draws, and stated rather than implied.
    max_ctx: int

    #: The dtype every kernel in this package is authored at. f32, because the
    #: question these tests ask is whether the computation is the one Hugging
    #: Face performs, and a bf16 comparison answers it only to within bf16.
    dt: str = "f32"

    @classmethod
    def from_published(cls, config: dict, *, max_ctx: int = 4096) -> "Qwen35Shape":
        rope = config["rope_parameters"]
        return cls(
            hidden=config["hidden_size"],
            vocab=config["vocab_size"],
            rms_eps=config["rms_norm_eps"],
            n_layers=config["num_hidden_layers"],
            layer_types=tuple(config["layer_types"]),
            full_attention_interval=config["full_attention_interval"],
            n_q_heads=config["num_attention_heads"],
            n_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            rope_theta=float(rope["rope_theta"]),
            partial_rotary_factor=float(rope["partial_rotary_factor"]),
            attention_bias=config["attention_bias"],
            max_pos=config["max_position_embeddings"],
            gdn_n_k_heads=config["linear_num_key_heads"],
            gdn_head_k_dim=config["linear_key_head_dim"],
            gdn_n_v_heads=config["linear_num_value_heads"],
            gdn_head_v_dim=config["linear_value_head_dim"],
            gdn_conv_kernel=config["linear_conv_kernel_dim"],
            n_experts=config["num_experts"],
            top_k=config["num_experts_per_tok"],
            moe_intermediate=config["moe_intermediate_size"],
            shared_intermediate=config["shared_expert_intermediate_size"],
            mtp_n_layers=config["mtp_num_hidden_layers"],
            max_ctx=max_ctx,
        )

    # ── derived extents ──────────────────────────────────────────────────

    @property
    def gqa_group(self) -> int:
        """Query heads sharing one key/value head."""
        return self.n_q_heads // self.n_kv_heads

    @property
    def rotary_dim(self) -> int:
        """How many of ``head_dim``'s entries rotate. The rest pass through."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def pass_dim(self) -> int:
        """The tail of each head that carries no position at all."""
        return self.head_dim - self.rotary_dim

    @property
    def q_gate_proj(self) -> int:
        """``q_proj``'s fan-out: two ``head_dim`` blocks per query head, one the
        query and one the output gate, chunked apart after the projection."""
        return self.n_q_heads * self.head_dim * 2

    @property
    def q_proj(self) -> int:
        return self.n_q_heads * self.head_dim

    @property
    def kv_proj(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gdn_key_dim(self) -> int:
        return self.gdn_n_k_heads * self.gdn_head_k_dim

    @property
    def gdn_value_dim(self) -> int:
        return self.gdn_n_v_heads * self.gdn_head_v_dim

    @property
    def gdn_conv_dim(self) -> int:
        """The convolution's channel count: query, key and value in one tensor."""
        return 2 * self.gdn_key_dim + self.gdn_value_dim

    @property
    def gdn_conv_context(self) -> int:
        """How many earlier positions the causal convolution needs.

        The kernel spans ``gdn_conv_kernel`` positions ending at the one being
        decoded, so the state handed in is the ``kernel - 1`` before it. Hugging
        Face stores ``kernel`` columns and drops the oldest on use; carrying
        ``kernel - 1`` says the same thing without a column no step reads.
        """
        return self.gdn_conv_kernel - 1

    @property
    def gdn_v_per_k(self) -> int:
        """Value heads sharing one key head."""
        return self.gdn_n_v_heads // self.gdn_n_k_heads

    def first_layer_of(self, block_type: str) -> int:
        """The lowest index in the published stack whose type is *block_type*."""
        return self.layer_types.index(block_type)


#: One token per step: the literal 1, not a range.
SEQ_LEN = 1

#: The published shape. ``max_ctx`` is the authoring envelope, not a published
#: field -- see ``Qwen35Shape.max_ctx``.
REAL = Qwen35Shape.from_published(_PUBLISHED)


# ── The Hugging Face oracle ─────────────────────────────────────────────
# Every builder below instantiates ONE submodule, never a stack. At the
# published dimensions a single MoE block already holds 256 experts, and the
# published stack holds forty of them; a fixture that built the stack to check
# one block would be measuring memory, not semantics.


def build_hf_config(shape: Qwen35Shape = REAL, *, layers: int | None = None):
    """A ``Qwen3_5MoeTextConfig`` from the published fields, verbatim.

    The published dictionary is passed through as it is rather than field by
    field, so a field this module does not know about still reaches Hugging
    Face. ``layers`` truncates ``num_hidden_layers`` and ``layer_types``
    together, which is what lets a caller build the smallest stack that still
    contains a layer of the type it wants.
    """
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: PLC0415
        Qwen3_5MoeTextConfig,
    )

    fields = dict(_PUBLISHED)
    if layers is not None:
        fields["num_hidden_layers"] = layers
        fields["layer_types"] = list(shape.layer_types[:layers])
    return Qwen3_5MoeTextConfig(**fields)


def build_hf_decoder_layer(block_type: str, seed=0, device="cpu", dtype=None,
                           shape: Qwen35Shape = REAL):
    """One ``Qwen3_5MoeDecoderLayer`` of *block_type*, weights drawn at *seed*.

    ``layer_idx`` is the lowest published index of that type, because the layer
    reads its own type out of ``config.layer_types[layer_idx]`` -- the index is
    how the type is chosen, not a decoration.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
        Qwen3_5MoeDecoderLayer,
    )

    index = shape.first_layer_of(block_type)
    config = build_hf_config(shape, layers=index + 1)
    return oracle.randomised(
        lambda: Qwen3_5MoeDecoderLayer(config, layer_idx=index), seed, device, dtype
    )


def build_hf_mixer(block_type: str, seed=0, device="cpu", dtype=None,
                   shape: Qwen35Shape = REAL):
    """One token mixer of *block_type* and the norm in front of it, and no more.

    A whole ``Qwen3_5MoeDecoderLayer`` is 3.3 GB in f32, of which 3.2 GB is its
    256-expert MoE block -- and the mixer boundaries do not touch the MoE at all.
    Building the layer to test the mixer put that 3.2 GB in every parallel test
    worker at once and exhausted a 140 GB device when the rest of the suite ran
    alongside; measured, as an out-of-memory failure in eight of this package's
    tests under ``-n 8``. So the mixer is built on its own.

    The classes are the published ones -- ``Qwen3_5MoeAttention``,
    ``Qwen3_5MoeGatedDeltaNet``, ``Qwen3_5MoeRMSNorm`` -- at the published
    dimensions, held in a container that exposes them under the attribute names a
    decoder layer uses. So an oracle written against a layer reads this without
    knowing the difference, and what it is comparing against is still Hugging
    Face's own module rather than a reimplementation of it.
    """
    import torch  # noqa: PLC0415
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
        Qwen3_5MoeAttention,
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeRMSNorm,
    )

    index = shape.first_layer_of(block_type)
    config = build_hf_config(shape, layers=index + 1)

    class MixerOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = Qwen3_5MoeRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            if block_type == "linear_attention":
                self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, index)
            else:
                self.self_attn = Qwen3_5MoeAttention(config, index)

    return oracle.randomised(MixerOnly, seed, device, dtype)


def rope_caches(shape: Qwen35Shape = REAL, total: int = 64, device="cpu", dtype=None):
    """cos / sin caches ``[total, rotary_dim]`` from the published rotary module.

    Narrower than ``head_dim``: ``partial_rotary_factor`` is 0.25, so the caches
    cover the 64 entries of each head that rotate and nothing else. That is what
    Hugging Face's own ``apply_rotary_pos_emb`` reads -- it slices ``q`` to
    ``cos.shape[-1]`` and concatenates the untouched tail back on.

    ``mrope`` is not exercised by a text-only fixture and this does not pretend
    it is. The published rotary embedding assigns a position triple per token and
    interleaves the three axes' frequencies by ``mrope_section``; with no image
    the three are the same number, so every branch of the interleave selects the
    same frequency and the result is ordinary RoPE at ``rotary_dim``. What these
    caches do cover is the partial factor. ``test_provenance.py`` measures the
    degeneracy rather than asserting it.
    """
    import torch  # noqa: PLC0415
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
        Qwen3_5MoeTextRotaryEmbedding,
    )

    config = build_hf_config(shape, layers=1)
    with torch.device(device):
        rotary = Qwen3_5MoeTextRotaryEmbedding(config)
    reference = torch.zeros(1, total, shape.hidden, device=device)
    cos, sin = rotary(reference, torch.arange(total, device=device).unsqueeze(0))
    cos, sin = cos[0], sin[0]
    return (cos, sin) if dtype is None else (cos.to(dtype), sin.to(dtype))


# ── Weight layout ───────────────────────────────────────────────────────


def linear_weight(linear):
    """HF ``nn.Linear.weight`` ``[out, in]`` -> the kernels' ``[1, in, out]``."""
    return oracle.linear_weight(linear)


def matrix_weight(weight):
    """A bare ``[out, in]`` parameter -> the kernels' ``[in, out]``.

    The MoE router and Hugging Face's expert tensors are ``nn.Parameter``s
    consumed by ``F.linear``, not ``nn.Linear`` modules, so they need the same
    transpose without the module wrapper.
    """
    return weight.t().contiguous()


def norm_gamma(norm):
    """A ``Qwen3_5MoeRMSNorm``'s scale as ``tf.rms_norm`` wants it.

    Hugging Face scales the normalised value by ``1 + weight``; ``tf.rms_norm``
    scales by the vector it is handed. The offset is therefore weight
    preprocessing, and it belongs on this side of the boundary -- the same place
    the projection transposes live -- rather than inside a kernel that would
    then only be correct for this one family.
    """
    return (1.0 + norm.weight.float()).to(norm.weight.dtype)


__all__ = [
    "DEPENDED_FIELDS",
    "FIELDS_DIGEST",
    "REAL",
    "SEQ_LEN",
    "SOURCE_REVISION",
    "SOURCE_SHA256",
    "SOURCE_URL",
    "TEXT_CONFIG_PATH",
    "Qwen35Shape",
    "build_hf_config",
    "build_hf_decoder_layer",
    "fields_digest",
    "linear_weight",
    "matrix_weight",
    "norm_gamma",
    "published_text_config",
    "rope_caches",
]
