"""The matrix product of ``matmul`` with the entry's input and output declared as [M, K] matrices.

A row-major [128, 16] fp16 matrix is byte for byte the [1, 16, 8, 16] feature map of ``matmul``:
load and store alone change rank, and every engine task keeps its feature map.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="matmul_2d", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def matmul_2d(
        a: Tensor[(128, 16), "f16", ((128, 16), (32, 1)), "gmem"],
        w: ConstTensor[(16, 16, 1, 1), "f16", ((16, 16, 1, 1), (16, 1, 1, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            al = tf.schedule((a, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.load())
            outer = tf.schedule((al, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            y = tf.schedule((outer, w, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.conv())
            rows = tf.schedule((y, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            return tf.schedule((rows, tf.empty(Tensor[(128, 16), "f16", ((128, 16), (32, 1)), "gmem"])), op=T.ane.store())
