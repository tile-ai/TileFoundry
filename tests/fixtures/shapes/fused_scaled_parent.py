"""A root that holds no constants of its own and reaches one child that does.

``tests/evaluator`` runs it against two readings to show each keeps its own
weight; ``tests/runtime`` checks what a reading must supply before anything runs.
The child binding is named ``scaled`` because both directories address the weight
as ``scaled.w``.
"""

from __future__ import annotations

from tests.fixtures.shapes.scaled_child import ScaledChild
from tilefoundry import func, module
from tilefoundry.dsl import Tensor
from tilefoundry.target import CudaTarget


@module(entry="fused", target=CudaTarget("nvidia.h200_sxm"))
class FusedScaledParent:
    scaled = ScaledChild

    @func
    def fused(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return scaled(x)  # noqa: F821 -- class-body binding
