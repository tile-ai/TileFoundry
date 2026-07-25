"""Assemble the DeepSeek-V4-Flash tree at a given scale: each node's declarative
source under ``model/`` is loaded with *config* injected."""
from __future__ import annotations

from pathlib import Path

from tests.models.deepseek_v4_flash.config import DSV4Config
from tests.models.loader import load_model
from tilefoundry.ir.core.module import Module

_MODEL_DIR = Path(__file__).parent / "model"

__all__ = ["load_causal_lm"]


def _load_layer(config: DSV4Config, index: int) -> Module:
    """One decoder layer with its own freshly loaded attention / moe:
    ``Module.renamed`` is a shallow copy, so a shared child would let this
    layer's ``load()`` clobber another's."""
    attention_module = load_model(_MODEL_DIR / "attention.py", config=config).DeepseekV4Attention
    moe_module = load_model(_MODEL_DIR / "moe.py", config=config).DeepseekV4MoE
    layer = load_model(
        _MODEL_DIR / "decoder_layer.py",
        config=config, attention_module=attention_module, moe_module=moe_module,
    ).DeepseekV4DecoderLayer
    return layer.renamed(f"layer{index}")


def load_causal_lm(config: DSV4Config) -> Module:
    """The full tree at *config*'s scale."""
    decoder_layers = tuple(_load_layer(config, i) for i in range(config.n_layers))
    return load_model(
        _MODEL_DIR / "causal_lm.py", config=config, decoder_layers=decoder_layers,
    ).DeepseekV4ForCausalLM
