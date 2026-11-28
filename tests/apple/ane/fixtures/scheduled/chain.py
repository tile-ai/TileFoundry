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
            al = tf.schedule((a, tf.empty(Tensor[(8, 16), "f16", ((8, 16), (1, 8)), "smem"])), op=T.ane.load())
            h = tf.schedule((al, b, tf.empty(Tensor[(8, 16), "f16", ((8, 16), (1, 8)), "smem"])), op=T.ane.conv())
            y = tf.schedule((h, w, tf.empty(Tensor[(8, 16), "f16", ((8, 16), (16, 1)), "smem"])), op=T.ane.conv())
            return tf.schedule((y, tf.empty(Tensor[(8, 16), "f16", ((8, 16), (16, 1)), "gmem"])), op=T.ane.store())
