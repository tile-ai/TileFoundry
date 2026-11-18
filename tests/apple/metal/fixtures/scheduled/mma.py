"""Two SIMD groups compute a @ b + c into a fresh fragment and into c's own fragment."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="mma", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def mma(
        a: Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"],
        b: Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"],
        c: Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"],
    ):
        with Mesh(("thread",), layout=Layout(((2, 2, 4, 2, 2),), ((32, 1, 2, 8, 16),))) as simd:
            fragment = ShardLayout(
                Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (S(0), S(4), S(2), S(3), S(1)), simd
            )
            fa = tf.schedule((tf.empty(Tensor[(2, 8, 8), "f32", fragment, "rmem"]), a), op=T.metal.ldmatrix())
            fb = tf.schedule((tf.empty(Tensor[(2, 8, 8), "f32", fragment, "rmem"]), b), op=T.metal.ldmatrix())
            fc = tf.schedule((tf.empty(Tensor[(2, 8, 8), "f32", fragment, "rmem"]), c), op=T.metal.ldmatrix())
            fresh = tf.schedule(
                (tf.empty(Tensor[(2, 8, 8), "f32", fragment, "rmem"]), fa, fb, fc), op=T.metal.mma(dtype=DType.f32)
            )
            fc = tf.schedule((fc, fa, fb, fc), op=T.metal.mma(dtype=DType.f32))
            fresh = tf.schedule(
                (tf.empty(Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"]), fresh), op=T.metal.stmatrix()
            )
            fc = tf.schedule(
                (tf.empty(Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"]), fc), op=T.metal.stmatrix()
            )
            return fresh, fc
