"""A tiled MLP against a naive one: the loop nest reassociates, and nothing else.

Test-only: no checkpoint, no Hugging Face oracle, no corpus or catalog entry.
The dimensions are the fixture's own, sized so the blocking divides.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tests.core.models.decode_oracle import agrees_to_one_rounding
from tilefoundry.cli.source import load_namespace
from tilefoundry.evaluator import evaluate
from tilefoundry.runtime.resource import DictResource

TILED_MLP = Path(__file__).parents[1] / "fixtures" / "dsl" / "tiled_mlp.py"
_TORCH_DT = torch.bfloat16


def _drawn(device="cpu", seed=0):
    """One input and one set of weights, shared by both spellings."""
    torch.manual_seed(seed)
    program = load_namespace(str(TILED_MLP))[0]
    S, HIDDEN, INTERMEDIATE = program["S"], program["HIDDEN"], program["INTERMEDIATE"]

    def draw(*shape):
        return (torch.randn(*shape, device=device) * 0.05).to(_TORCH_DT)

    x = draw(1, S, HIDDEN)
    loaded = program["TiledMLP"].load(
        DictResource(
            {
                "w_gate": draw(1, HIDDEN, INTERMEDIATE),
                "w_up": draw(1, HIDDEN, INTERMEDIATE),
                "w_down": draw(1, INTERMEDIATE, HIDDEN),
            }
        )
    )
    return loaded, x


def test_the_tiled_mlp_computes_what_the_naive_one_does() -> None:
    """The blocked K walk reassociates the reduction and changes nothing else."""
    loaded, x = _drawn()

    agrees_to_one_rounding(
        evaluate(loaded.tiled_mlp, x),
        evaluate(loaded.naive_mlp, x),
    )
