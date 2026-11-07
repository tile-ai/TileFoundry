"""Two SIMD matrices within each of four independent threadgroups."""
from tilefoundry.dsl import *
from tilefoundry_apple import MetalTarget


@module(entry="matrices", target=MetalTarget("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 4), Topology("thread", 64)))
class Program:
    @func
    def matrices(
        ordinary: Tensor[(2,8,8), "f32", ((2,8,8),(64,8,1)), "gmem"],
        odd: Tensor[(2,7,8), "f32"],
    ):
        with Mesh(("threadgroup",), layout=(4,), names=("tile",)) as _blocks:
            with Mesh(("thread",), layout=Layout((2,32),(32,1)), names=("group","lane")) as _threads:
                ordinary_mm = tf.matmul(ordinary, ordinary)
                odd_mm = tf.matmul(odd, ordinary)
                return ordinary_mm, odd_mm
