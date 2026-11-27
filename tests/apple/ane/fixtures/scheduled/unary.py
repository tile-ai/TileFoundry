"""An affine task with a clamp on the processing engine: clip(2 * x + 3, 0, 6)."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="unary", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def unary(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]), x),
                op=T.ane.unary(pre_scale=2.0, bias=3.0, clamp=(0.0, 6.0)),
            )
