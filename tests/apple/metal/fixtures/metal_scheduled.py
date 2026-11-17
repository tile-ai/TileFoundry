"""Two SIMD groups with their Metal instructions chosen: an 8x8 f32 product and a reduction.

MatMul loads A, B and the accumulator into fragments, accumulates in place and
stores the result; Reduction combines the lanes of each group. Lowering refuses
the other two: Unmeshed issues a Metal instruction with no enclosing mesh, and
CoreReduce selects core's generic T.reduce, which Metal does not lower.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 2, 4, 2, 2),), ((32, 1, 2, 8, 16),)))
FRAGMENT = ShardLayout(Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (S(0), S(4), S(2), S(3), S(1)), SIMDGROUPS)
LANES = Mesh((Topology("thread", 64),), layout=Layout(((2, 32),), ((32, 1),)))
PARTIAL_SUM = ShardLayout(Layout((2, 1), (0, 0)), (S(0), P("sum")), LANES)
PER_LANE = ShardLayout(Layout((2, 32), (32, 1)), (S(0), S(1)), LANES)


@module(entry="mma", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class MatMul:
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
            fc = tf.schedule((fc, fa, fb, fc), op=T.metal.mma())
            return tf.schedule(
                (tf.empty(Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"]), fc), op=T.metal.stmatrix()
            )


@module(entry="reduce", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Reduction:
    @func
    def reduce(partial: Tensor[(2, 1), "f32", PARTIAL_SUM, "rmem"]):
        with Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),))) as simd:
            total = tf.empty(Tensor[(2, 1), "f32", ShardLayout(Layout((2, 1), (0, 0)), (S(0), B()), simd), "rmem"])
            return tf.schedule((total, partial), op=T.metal.reduce())


@module(entry="unmeshed", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Unmeshed:
    @func
    def unmeshed(
        a: Tensor[(2, 8, 8), "f32", FRAGMENT, "rmem"],
        b: Tensor[(2, 8, 8), "f32", FRAGMENT, "rmem"],
    ):
        return tf.matmul(a, b)


@module(entry="core_reduce", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class CoreReduce:
    @func
    def core_reduce(x: Tensor[(2, 32), "f32", PER_LANE, "rmem"]):
        with Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),))) as simd:
            total = tf.empty(Tensor[(2, 1), "f32", ShardLayout(Layout((2, 1), (1, 0)), (S(0), B()), simd), "rmem"])
            return tf.schedule((x, total), op=T.reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM))
