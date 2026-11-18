"""Two SIMD groups load one 8x8 matrix each, transposed, and store the fragment back as it is."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="ldmatrix", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def ldmatrix(a: Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"]):
        with Mesh(("thread",), layout=Layout(((2, 2, 4, 2, 2),), ((32, 1, 2, 8, 16),))) as simd:
            fragment = ShardLayout(
                Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (S(0), S(4), S(2), S(3), S(1)), simd
            )
            fa = tf.schedule(
                (tf.empty(Tensor[(2, 8, 8), "f32", fragment, "rmem"]), a), op=T.metal.ldmatrix(transpose=True)
            )
            return tf.schedule(
                (tf.empty(Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"]), fa), op=T.metal.stmatrix()
            )
