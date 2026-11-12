"""Pinned Qwen3-MoE dimensions and scaled numerical workload."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


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


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", choices=("full", "tiny"), default="full")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1, help="expert partition of the data axis")
    parser.add_argument("--packed", action="store_true", help="multiple documents with unequal padding per data lane")
    parser.add_argument("--routing-check", choices=("strict", "replay-near-ties"), default="strict",
                        help="validation only: optionally replay oracle choices after verifying near-tie margins")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--optimizer", choices=("sgd", "dmuon"), default="sgd")
    parser.add_argument("--checkpoint", choices=("auto", "stage", "none"), default="auto")
    parser.add_argument("--boundary", choices=("stage", "layer"), default="layer")
    parser.add_argument("--overlap", choices=("off", "on"), default="off")
    parser.add_argument("--ns-backend", choices=("direct", "gram"), default="direct")
    parser.add_argument("--forward-unshard", choices=("broadcast", "all_gather"), default="broadcast",
                        help="attention matrix publication; explicit expert layouts require broadcast in the pin")
    parser.add_argument("--muon-max-norm", type=float,
                        help="clip the global Muon-only norm; AdamW parameters are excluded")
    parser.add_argument("--replicas", type=int, default=1, help="HSDP replica dimension inside each stage's DP group")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="number of diagnostic steps after all unprofiled timing windows")
    parser.add_argument("--timing-steps", type=int, help="steps per timing window; defaults to profile-steps")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timing-rounds", type=int, default=1, help="independent multi-step timing windows before profiling")
    parser.add_argument("--reference-offload", action="store_true",
                        help="keep only stage-owned reference gradients and offload the reference between checks")
    parser.add_argument("--compare-final", type=Path,
                        help="require bitwise final weights matching a prior run with the same data steps")
    parser.add_argument("--transport", choices=("native", "socket"), default="native")
    parser.add_argument("--output", type=Path, default=Path(".tmp/qwen3-dmuon/pipeline"))
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", 0))
    if args.stages < 2 or args.layers < args.stages or args.layers % args.stages:
        parser.error("layers must be a positive multiple of at least two pipeline stages")
    if not world or world % args.stages:
        parser.error("launch with torchrun; world size must be a multiple of stages")
    if min(args.cp, args.ep) < 1 or (world // args.stages) % (args.cp * args.ep):
        parser.error("CP times EP must divide ranks per pipeline stage")
    args.packed = args.packed or args.cp > 1 or args.ep > 1
    if args.packed and (args.tokens < 32 or args.tokens % args.cp):
        parser.error("packed sequences need >= 32 tokens divisible by CP")
    if args.packed and (args.optimizer != "dmuon" or (args.ep > 1 and args.replicas != 1) or args.muon_max_norm is not None
                        or args.boundary != "layer"):
        parser.error("packed training requires dmuon and layer boundaries; EP with HSDP and norm clipping are not supported")
    if args.replicas < 1 or (world // args.stages) % args.replicas:
        parser.error("HSDP replicas must divide the DP group size")
    if args.replicas > 1 and args.forward_unshard == "all_gather":
        parser.error("pinned dmuon exports stale replica-row weights after HSDP all-gather publication; use broadcast")
    if args.profile_steps < 0 or args.warmup_steps < 0 or args.timing_rounds < 1:
        parser.error("profile/warmup counts must be nonnegative and timing rounds positive")
    if args.timing_steps is not None and args.timing_steps < 1:
        parser.error("timing steps must be positive")
    if args.muon_max_norm is not None and not 0 < args.muon_max_norm < float("inf"):
        parser.error("Muon max norm must be finite and positive")
    if args.optimizer != "dmuon" and (args.overlap == "on" or args.replicas > 1):
        parser.error("HSDP and optimizer overlap require --optimizer dmuon")
    if args.optimizer != "dmuon" and (args.muon_max_norm is not None or args.ns_backend != "direct"
                                     or args.forward_unshard != "broadcast"):
        parser.error("Muon options require --optimizer dmuon")
    if args.compare_final and (args.optimizer != "dmuon" or args.compare_final.resolve() == args.output.resolve()):
        parser.error("final-state comparison needs dmuon and a different baseline output directory")
    if args.microbatches < args.stages or args.tokens < 2 or args.steps < 1:
        parser.error("require microbatches >= stages, tokens >= 2, and steps >= 1")
    return args
