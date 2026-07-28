"""One Qwen3.5-35B-A3B decoder layer of either published type, at the published
shape.

``build_decoder_layer("full_attention")`` and
``build_decoder_layer("linear_attention")`` load ``model/decoder_layer.py`` with
the matching token mixer injected. Each call re-executes the model sources, so
two layers are two Module trees rather than one tree referenced twice -- a Module
is the execution domain of the functions it owns, and a shared instance would
give two layers one analysis.

There is deliberately no decoder *stack* here. The published model is 40 layers
of 35 billion parameters; the reference for a model this size is declared,
boundary-complete submodules, and a 40-layer walk is neither cheap nor more
informative than the layer it repeats. What a stack would observe and this does
not -- layer order, the thread of residuals running through 40 layers, the final
norm -- is stated as untested rather than approximated.
"""
from __future__ import annotations

from pathlib import Path

from tests.models.loader import load_model
from tests.models.qwen3_5_35b_a3b.config import REAL, Qwen35Shape

_MODEL_DIR = Path(__file__).parent / "model"

#: Which model source authors each published token mixer.
MIXER_SOURCE = {
    "full_attention": "full_attention.py",
    "linear_attention": "linear_attention.py",
}


def build_mixer(block_type: str, shape: Qwen35Shape = REAL):
    """The token-mixer Module for *block_type*, freshly loaded."""
    try:
        source = MIXER_SOURCE[block_type]
    except KeyError:
        known = ", ".join(sorted(MIXER_SOURCE))
        raise KeyError(
            f"no token mixer {block_type!r}; this model publishes {known}"
        ) from None
    loaded = load_model(_MODEL_DIR / source, config=shape)
    return loaded.Qwen3_5FullAttention if block_type == "full_attention" \
        else loaded.Qwen3_5LinearAttention


def build_moe(shape: Qwen35Shape = REAL):
    """The MoE block Module, freshly loaded."""
    return load_model(_MODEL_DIR / "moe.py", config=shape).Qwen3_5MoE


def build_decoder_layer(block_type: str, shape: Qwen35Shape = REAL):
    """One decoder layer of *block_type*: the mixer and the MoE block as its
    children, and the residual additions that close each half."""
    loaded = load_model(
        _MODEL_DIR / "decoder_layer.py",
        config=shape,
        mixer_module=build_mixer(block_type, shape).renamed("mixer"),
        moe_module=build_moe(shape).renamed("moe"),
    )
    return loaded.Qwen3_5DecoderLayer


__all__ = ["MIXER_SOURCE", "build_decoder_layer", "build_mixer", "build_moe"]
