"""RMSNorm over the last axis, both the input and the weight trained."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class RmsNorm:
    @func
    def forward(x: Tensor[(4, 6), "f32"], w: Tensor[(6,), "f32"]):
        return tf.rms_norm(x, w, eps=1e-5) * x
