"""Instruction forms the scheduled fixtures do not reach; each module is one case."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class UnaryOrder:
    """Bias precedes square, and one input ties at the bias."""

    @func
    def main(x: Tensor[(1, 1, 1, 8), "f16", ((1, 1, 1, 8), (8, 8, 8, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 1, 1, 8), "f16", ((1, 1, 1, 8), (8, 8, 8, 1)), "gmem"]), x),
                op=T.ane.unary(bias=1.0, square=True, clamp=(0.0, 10.0)),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class MaxPoolCeil:
    """A max pool whose last window along the width is partial."""

    @func
    def main(x: Tensor[(1, 2, 5, 6), "f16", ((1, 2, 5, 6), (60, 30, 6, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 2, 2, 3), "f16", ((1, 2, 2, 3), (12, 6, 3, 1)), "gmem"]), x),
                op=T.ane.pool(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True, kind=PoolKind.MAX),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class AveragePoolInside:
    """An average that counts only the cells inside the map."""

    @func
    def main(x: Tensor[(1, 2, 5, 6), "f16", ((1, 2, 5, 6), (60, 30, 6, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 2, 3, 3), "f16", ((1, 2, 3, 3), (18, 9, 3, 1)), "gmem"]), x),
                op=T.ane.pool(
                    kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), count_include_pad=False, kind=PoolKind.AVG
                ),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class ReduceDropped:
    """A sum over height and width that drops both axes."""

    @func
    def main(x: Tensor[(1, 2, 3, 4), "f16", ((1, 2, 3, 4), (24, 12, 4, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 2), "f16", ((1, 2), (2, 1)), "gmem"]), x),
                op=T.ane.reduce(axes=(2, 3), keepdim=False, kind=ReduceKind.SUM),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class ReduceNorm:
    """The reciprocal square root of a sum of squares over channels and width."""

    @func
    def main(x: Tensor[(1, 2, 3, 4), "f16", ((1, 2, 3, 4), (24, 12, 4, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 1, 3, 1), "f16", ((1, 1, 3, 1), (3, 3, 1, 1)), "gmem"]), x),
                op=T.ane.reduce(
                    axes=(1, 3), keepdim=True, kind=ReduceKind.SUM, square=True, post=T.ane.Post.RSQRT
                ),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class ConvStridedGrouped:
    """A convolution with a stride of 2, two groups and a bias."""

    @func
    def main(
        x: Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "gmem"],
        w: Tensor[(6, 2, 3, 3), "f16", ((6, 2, 3, 3), (18, 9, 3, 1)), "gmem"],
        b: Tensor[(6,), "f16", ((6,), (1,)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 6, 3, 3), "f16", ((1, 6, 3, 3), (54, 9, 3, 1)), "gmem"]), x, w, b),
                op=T.ane.conv(stride=(2, 2), padding=(1, 1), groups=2),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class MatrixBias:
    """A matrix product that adds its bias to each column."""

    @func
    def main(
        a: Tensor[(8, 16), "f16", ((8, 16), (1, 8)), "gmem"],
        w: Tensor[(16, 5), "f16", ((16, 5), (5, 1)), "gmem"],
        b: Tensor[(5,), "f16", ((5,), (1,)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(8, 5), "f16", ((8, 5), (5, 1)), "gmem"]), a, w, b), op=T.ane.conv()
            )
