"""Binary has a rule, but not for MIN."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class UnsupportedKind:
    @func
    def forward(x: Tensor[(4, 3), "f32"], y: Tensor[(4, 3), "f32"]):
        low = tf.minimum(x, y)
        return low * x
