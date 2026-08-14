"""One matmul into one rms_norm: the shape two directories each wrote for themselves.

``tests/analysis`` compares the dependences against a hand-written isl map, which
is what fixes the size at (2,4)x(4,2) -- the map spells those extents out.
``tests/schedule`` only needs a body that is not a matmul, so the size does not
reach its assertion and the small one serves both.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf


@func
def gemm_rms_norm(
    x: Tensor[(2, 4), "f32"], w: Tensor[(4, 2), "f32"], weight: Tensor[(2,), "f32"]
) -> Tensor[(2, 2), "f32"]:
    h = tf.matmul(x, w)
    return tf.rms_norm(h, weight)
