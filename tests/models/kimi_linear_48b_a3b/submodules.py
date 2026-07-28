"""Kimi-Linear-48B-A3B submodules at the real shape: loads
``model/submodules.py`` with ``REAL`` and re-exports the three Modules plus the
IR Function nodes the analysis and schedule tests inspect directly."""
from __future__ import annotations

from pathlib import Path

from tests.models.kimi_linear_48b_a3b.config import REAL
from tests.models.loader import load_model

_loaded = load_model(Path(__file__).parent / "model" / "submodules.py", config=REAL)

KimiMla = _loaded.KimiMla
KimiKda = _loaded.KimiKda
KimiMoe = _loaded.KimiMoe

# IR Function nodes, not the callables ``Module.__getattr__`` returns.
mla_attention = KimiMla.lookup("mla_attention")

short_conv = KimiKda.lookup("short_conv")
l2_normalize = KimiKda.lookup("l2_normalize")
kda_gate = KimiKda.lookup("kda_gate")
kda_attention = KimiKda.lookup("kda_attention")

router = KimiMoe.lookup("router")
shared_expert = KimiMoe.lookup("shared_expert")
moe = KimiMoe.lookup("moe")


_MOE_AT: dict[int, object] = {}


def moe_at(n_experts: int):
    """The same MoE source, loaded with a reduced expert count.

    Analysis wants the published 256 experts, because the gather over them is
    most of the kernel's cost profile. Evaluation does not always: 256 experts of
    f32 weights is about 7 GB, and this suite runs 8-way parallel, so the tests
    that only need to tell a right answer from a wrong one run at a smaller count.
    None of the routing properties they check depend on it -- the count only has
    to exceed `top_k` for a top-k to mean anything.
    """
    from dataclasses import replace  # noqa: PLC0415

    if n_experts not in _MOE_AT:
        loaded = load_model(
            Path(__file__).parent / "model" / "submodules.py",
            config=replace(REAL, n_experts=n_experts),
        )
        _MOE_AT[n_experts] = loaded.KimiMoe.lookup("moe")
    return _MOE_AT[n_experts]


__all__ = [
    "KimiKda",
    "KimiMla",
    "KimiMoe",
    "kda_attention",
    "kda_gate",
    "l2_normalize",
    "mla_attention",
    "moe",
    "moe_at",
    "router",
    "shared_expert",
    "short_conv",
]
