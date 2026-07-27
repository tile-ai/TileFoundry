"""Real-scale DeepSeek-V4-Flash MoE: loads ``model/moe.py`` with ``REAL`` and
re-exports its hash-router and learned-router entries for ``test_moe.py`` and
``tests/schedule/*``."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tests.models.deepseek_v4_flash.config import REAL
from tests.models.loader import load_model
from tilefoundry.target import CudaTarget

_MODEL_PATH = Path(__file__).parent / "model" / "moe.py"
_loaded = load_model(_MODEL_PATH, config=REAL)

# The authored modules declare no Target: they are nested as children of a
# decoder layer, and only a root module declares the Target its tree runs on.
# Standalone analysis/schedule callers select them as their own root, so those
# re-exports declare it here.
moe_hash_module = replace(_loaded.DeepseekV4MoE, target=CudaTarget())
deepseek_v4_flash_module = replace(_loaded.DeepseekV4NoauxTcMoE, target=CudaTarget())

# Real-scale model constants, re-exported for callers that only need the
# numbers, not a built Module.
DIM = REAL.dim
N_ROUTED = REAL.n_routed
N_ACT = REAL.n_act
MOE_INTER = REAL.moe_inter
ROUTE_SCALE = REAL.route_scale
SWIGLU_LIMIT = REAL.swiglu_limit
VOCAB = REAL.vocab

# IR Function nodes, not Modules -- test_moe.py inspects them directly
# (.params / .return_type / .body / .target).
deepseek_v4_flash_moe = deepseek_v4_flash_module.lookup("deepseek_v4_flash_moe")
moe_experts_core = deepseek_v4_flash_module.lookup("moe_experts_core")
moe_topk = deepseek_v4_flash_module.lookup("moe_topk")

__all__ = [
    "DIM",
    "MOE_INTER",
    "N_ACT",
    "N_ROUTED",
    "ROUTE_SCALE",
    "SWIGLU_LIMIT",
    "VOCAB",
    "deepseek_v4_flash_moe",
    "deepseek_v4_flash_module",
    "moe_experts_core",
    "moe_hash_module",
    "moe_topk",
]
