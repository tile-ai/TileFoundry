"""Qwen dimensions, stage computation and architecture-specific parameter roles."""

import copy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Shape:
    layers: int = 48
    hidden: int = 2048
    q_heads: int = 32
    kv_heads: int = 4
    head_dim: int = 128
    experts: int = 128
    top_k: int = 8
    expert_hidden: int = 768
    vocab: int = 151936

    def parameter_count(self):
        """Count bias-free GQA, per-head norms, expert matrices and untied embeddings."""
        q, kv = self.q_heads * self.head_dim, self.kv_heads * self.head_dim
        attention = self.hidden * (2 * q + 2 * kv) + 2 * self.head_dim
        moe = self.experts * (3 * self.hidden * self.expert_hidden + self.hidden)
        return 2 * self.vocab * self.hidden + self.hidden + self.layers * (attention + moe + 2 * self.hidden)

    def transformers_config(self):
        """Build the reference configuration without fetching model weights."""
        from transformers import Qwen3MoeConfig

        config = Qwen3MoeConfig(
            num_hidden_layers=self.layers, hidden_size=self.hidden,
            num_attention_heads=self.q_heads, num_key_value_heads=self.kv_heads,
            head_dim=self.head_dim, num_experts=self.experts,
            num_experts_per_tok=self.top_k, moe_intermediate_size=self.expert_hidden,
            vocab_size=self.vocab, norm_topk_prob=True, tie_word_embeddings=False,
            rms_norm_eps=1e-6, rope_theta=1e6, attention_bias=False,
            attention_dropout=0.0, output_router_logits=True, router_aux_loss_coef=0.001,
        )
        config._attn_implementation = "eager"
        return config


FULL = Shape()
TINY = Shape(layers=2, hidden=128, q_heads=16, kv_heads=2, head_dim=16,
             experts=16, top_k=8, expert_hidden=64, vocab=256)
SOURCES = {
    "qwen": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    "dmuon": "083ea977c045c5c15881845b3a3c465b6d2c4bb2",
    "qwen_reference": "transformers.models.qwen3_moe; runtime version recorded per run",
}



class QwenStage(nn.Module):
    """Own a contiguous layer interval and carry earlier router outputs to the loss."""

    def __init__(self, reference, index, stages):
        super().__init__()
        self.config = reference.config
        self.pack = None
        self.route_oracle = None
        self.microbatch = 0
        self.routing_comparisons = []
        self.first, self.last = index == 0, index == stages - 1
        width = self.config.num_hidden_layers // stages
        self.begin, self.end = index * width, (index + 1) * width
        self.model = nn.Module()
        if self.first:
            self.model.embed_tokens = copy.deepcopy(reference.model.embed_tokens)
        self.model.layers = nn.ModuleDict({str(i): copy.deepcopy(reference.model.layers[i])
                                          for i in range(self.begin, self.end)})
        self.model.rotary_emb = copy.deepcopy(reference.model.rotary_emb)
        if self.last:
            self.model.norm = copy.deepcopy(reference.model.norm)
            self.lm_head = copy.deepcopy(reference.lm_head)

    def parameter_roles(self):
        """Describe matrix semantics without importing an optimizer backend."""
        result = {}
        for name, layer in self.model.layers.items():
            prefix = f"model.layers.{name}"
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                result[f"{prefix}.self_attn.{projection}.weight"] = ("attention", None)
            experts = layer.mlp.experts
            count = experts.gate_up_proj.shape[0]
            width = experts.down_proj.shape[-1]
            hidden = self.config.hidden_size
            result[f"{prefix}.mlp.experts.gate_up_proj"] = ("expert", (count, 2, width, hidden))
            result[f"{prefix}.mlp.experts.down_proj"] = ("expert", (count, hidden, width))
        return result

    def parameter_domains(self):
        """Actual executing modules whose parameters share a gradient domain."""
        dense, expert = [], []
        if self.first:
            dense.append(("model.embed_tokens", self.model.embed_tokens))
        for name, layer in self.model.layers.items():
            prefix = f"model.layers.{name}"
            dense.extend((f"{prefix}.{part}", module) for part, module in (
                ("input_layernorm", layer.input_layernorm), ("self_attn", layer.self_attn),
                ("post_attention_layernorm", layer.post_attention_layernorm), ("mlp.gate", layer.mlp.gate)))
            expert.append((f"{prefix}.mlp.experts", layer.mlp.experts))
        if self.last:
            dense.extend((("model.norm", self.model.norm), ("lm_head", self.lm_head)))
        return dense, expert

    def forward(self, value, earlier_routers=None):
        hidden = self.model.embed_tokens(value) if self.first else value
        batch, tokens, _ = hidden.shape
        positions = torch.arange(tokens, device=hidden.device)
        position_ids = self.pack.local(self.pack.positions)
        rotary = self.model.rotary_emb(hidden, position_ids=position_ids)
        routers = [] if earlier_routers is None else list(earlier_routers.unbind(dim=1))

        def capture(module, inputs, output):
            if self.route_oracle is not None:
                output = self.route_oracle.check(self, name, output, batch, tokens)
            routers.append(output[0].reshape(batch, tokens, self.config.num_experts))
            return output

        for name, layer in self.model.layers.items():
            handle = layer.mlp.gate.register_forward_hook(capture)
            try:
                with torch.profiler.record_function(f"qwen/layer/{name}"):
                    hidden = layer(hidden, attention_mask=None, position_ids=position_ids,
                                   cache_position=positions, position_embeddings=rotary, use_cache=False)
            finally:
                handle.remove()
        output = self.lm_head(self.model.norm(hidden)) if self.last else hidden
        return output, torch.stack(routers, dim=1)
