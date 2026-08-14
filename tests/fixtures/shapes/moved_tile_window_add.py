"""The same tile window moved by a compile-time offset: rows 6..12 read 3 at a time.

``tests/analysis`` checks that the offset reaches the access map, ``tests/passes``
checks that it reaches the lowered address as arithmetic over the induction
variable rather than a materialized value. The offset divides here, so what
separates this from ``tile_window_add`` is the move, not the tail.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf


@func
def moved_tile_window_add(
    x: Tensor[(12, 4), "f32"], seed: Tensor[(3, 4), "f32"]
) -> Tensor[(3, 4), "f32"]:
    o = tf.add(seed, seed)
    for i in tile(6, 3):
        o = tf.add(x[i + 6, :], seed)
    return o
