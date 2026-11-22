"""Two matrix products on ANE; the first result stays in the on-chip buffer.

The second product reads it there, so it never returns to unified memory.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="chain", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def chain(
        a: Tensor[(8, 16), "f16", ((8, 16), (1, 8)), "gmem"],
        b: Tensor[(16, 16), "f16", ((16, 16), (16, 1)), "gmem"],
        w: Tensor[(16, 16), "f16", ((16, 16), (16, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            h = tf.schedule((tf.empty(Tensor[(8, 16), "f16", ((8, 16), (1, 8)), "smem"]), a, b), op=T.ane.conv())
            return tf.schedule((tf.empty(Tensor[(8, 16), "f16", ((8, 16), (16, 1)), "gmem"]), h, w), op=T.ane.conv())
