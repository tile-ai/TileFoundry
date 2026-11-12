"""Pinned Qwen3-MoE dimensions and scaled numerical workload."""

from dataclasses import dataclass


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
