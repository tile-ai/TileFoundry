"""A function that calls a function that calls a function, on two H200s.

Differentiating it saves each callee's values inside its caller's, so the
generated Forward returns a tuple three levels deep.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="forward", target=cuda.Target("nvidia.h200_sxm", device_count=2), topologies=(Topology("gpu", 2), Topology("cta", 1)))
class CallChain:
    mesh = Mesh(("gpu",), (2,), names=("dp",))

    @func
    def inner(x: Tensor[(8 @ mesh.dp, 16), "f32", "gmem"], w: Tensor[(16, 16), "f32", (16, 16), "gmem"]):
        with mesh:
            return tf.silu(tf.matmul(x, w))

    @func
    def middle(x: Tensor[(8 @ mesh.dp, 16), "f32", "gmem"], w: Tensor[(16, 16), "f32", (16, 16), "gmem"]):
        with mesh:
            return inner(x, w) * inner(x * x, w)

    @func
    def forward(x: Tensor[(8 @ mesh.dp, 16), "f32", "gmem"], w: Tensor[(16, 16), "f32", (16, 16), "gmem"]):
        with mesh:
            return tf.reduce(middle(x, w) + x, axes=(0, 1), kind=ReduceKind.SUM, keepdim=False)
