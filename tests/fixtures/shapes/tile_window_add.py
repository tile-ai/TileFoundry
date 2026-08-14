"""A tile window whose step does not divide the axis: 10 rows read 4 at a time.

The tail is the point. ``tests/analysis`` extracts the access relations of the
windows, ``tests/parser`` expects evaluating it to be refused, and
``tests/passes`` expects lowering it to be refused -- three assertions over one
program, which is why it lives here rather than three times over.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf


@func
def tile_window_add(
    x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]
) -> Tensor[(4, 4), "f32"]:
    o = tf.add(seed, seed)
    for i in tile(10, 4):
        o = tf.add(x[i, :], seed)
    return o
