"""Corpus selections for three real authored-loop matrix kernels."""

from __future__ import annotations

from tests.models.access_footprint.model import (
    FlashSplitKDecode,
    GroupedMoEGEMM,
    TiledQKVProjection,
)
from tests.models.corpus import FunctionCase, ModelCase

MODEL = "access_footprint"

QKV = ModelCase(
    id="access_footprint.qkv",
    model=MODEL,
    prototype=TiledQKVProjection,
    analyze=(
        FunctionCase(
            id="access_footprint/analyze/qkv_projection",
            selector="qkv_projection",
        ),
    ),
)

SPLIT_K = ModelCase(
    id="access_footprint.split_k",
    model=MODEL,
    prototype=FlashSplitKDecode,
    analyze=(
        FunctionCase(
            id="access_footprint/analyze/split_k_decode",
            selector="flash_split_k_decode",
            dims={"ctx": 4096},
        ),
    ),
)

GROUPED_MOE = ModelCase(
    id="access_footprint.grouped_moe",
    model=MODEL,
    prototype=GroupedMoEGEMM,
    analyze=(
        FunctionCase(
            id="access_footprint/analyze/grouped_moe_gemm",
            selector="grouped_gemm",
        ),
    ),
)

CASES = (QKV, SPLIT_K, GROUPED_MOE)

__all__ = ["CASES", "GROUPED_MOE", "MODEL", "QKV", "SPLIT_K"]
