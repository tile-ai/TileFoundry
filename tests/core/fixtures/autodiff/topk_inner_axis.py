"""TopK along an axis that is not the last."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class TopkInnerAxis:
    @func
    def forward(x: Tensor[(6, 4), "f32"]):
        values, indices = tf.topk(x, k=2, axis=0)
        return values * 2.0
