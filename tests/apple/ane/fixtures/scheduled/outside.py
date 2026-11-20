"""A max pool outside any mesh: core evaluates it, and lowering finds no mesh to issue it in."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="outside", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def outside(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"]):
        return tf.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
