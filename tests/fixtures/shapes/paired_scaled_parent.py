"""One child bound twice, so each binding reads its own constants.

``tests/analysis`` checks that expansion keeps ``left.w`` and ``right.w`` distinct
and that the activation parameter stays the caller's own; ``tests/evaluator`` runs
it against a reading that gives the two bindings different weights. The entry is
named ``both`` and its parameter ``w`` because both directories address them by
those names.
"""

from __future__ import annotations

from tests.fixtures.shapes.scaled_child import ScaledChild
from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.target import CudaTarget


@module(entry="both", target=CudaTarget("nvidia.h200_sxm"))
class PairedScaledParent:
    left = ScaledChild
    right = ScaledChild

    @func
    def both(w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.add(left(w), right(w))  # noqa: F821 -- class-body bindings
