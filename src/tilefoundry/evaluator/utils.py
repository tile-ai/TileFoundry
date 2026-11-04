"""Helpers the Op evaluators share.

See [evaluator §6](docs/spec/evaluator.md#6-layout-domain).
"""

from __future__ import annotations

from tilefoundry.core import ShardTensor
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.types import Placement


def common_placements(*tensors: ShardTensor) -> tuple[Placement, ...]:
    """The placements the local views among *tensors* hold in common; empty when all are global."""
    held = [tensor.placements for tensor in tensors if not tensor.is_global]
    if any(placements != held[0] for placements in held[1:]):
        raise EvalError("local-view inputs are held by different placements")
    return held[0] if held else ()


__all__ = ["common_placements"]
