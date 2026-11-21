"""A core copy scheduled under ANE, which lowers only T.ane instructions; lowering refuses it."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="core", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def core(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"]), x),
                op=T.copy(),
            )
