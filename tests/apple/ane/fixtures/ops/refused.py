"""Scheduled instructions that a load refuses; each module breaks one rule."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Gapped:
    """A feature map whose W stride is 2."""

    @func
    def main(x: Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (288, 72, 12, 2)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (x, tf.empty(Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "smem"])),
                op=T.ane.load(),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Unloaded:
    """An engine task that reads its source from unified memory instead of L2."""

    @func
    def main(x: Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (x, tf.empty(Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "smem"])),
                op=T.ane.unary(abs=True),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class ClampReversed:
    """A clamp whose low bound is above its high bound."""

    @func
    def main(x: Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "smem"])), op=T.ane.load())
            return tf.schedule(
                (xs, tf.empty(Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "smem"])),
                op=T.ane.unary(clamp=(6.0, 0.0)),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class KeepdimWidth:
    """A reduction over the width alone that drops the axis."""

    @func
    def main(x: Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "smem"])), op=T.ane.load())
            return tf.schedule(
                (xs, tf.empty(Tensor[(1, 4, 6), "f16", ((1, 4, 6), (24, 6, 1)), "smem"])),
                op=T.ane.reduce(axes=(3,), keepdim=False, kind=ReduceKind.SUM),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class WeightOnChip:
    """A convolution whose weight is in the on-chip buffer."""

    @func
    def main(
        x: Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "gmem"],
        w: Tensor[(6, 4, 3, 3), "f16", ((6, 4, 3, 3), (36, 9, 3, 1)), "smem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "smem"])), op=T.ane.load())
            return tf.schedule(
                (xs, w, tf.empty(Tensor[(1, 6, 4, 4), "f16", ((1, 6, 4, 4), (96, 16, 4, 1)), "smem"])),
                op=T.ane.conv(),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class WrongResultShape:
    """A pool whose destination is not the pooled shape."""

    @func
    def main(x: Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "smem"])), op=T.ane.load())
            return tf.schedule(
                (xs, tf.empty(Tensor[(1, 4, 2, 3), "f16", ((1, 4, 2, 3), (24, 6, 3, 1)), "smem"])),
                op=T.ane.pool(kernel_size=(2, 2), stride=(2, 2), kind=PoolKind.MAX),
            )


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class GroupsUneven:
    """A convolution whose groups do not divide its channels."""

    @func
    def main(
        x: Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "gmem"],
        w: Tensor[(6, 4, 3, 3), "f16", ((6, 4, 3, 3), (36, 9, 3, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 4, 6, 6), "f16", ((1, 4, 6, 6), (144, 36, 6, 1)), "smem"])), op=T.ane.load())
            return tf.schedule(
                (xs, w, tf.empty(Tensor[(1, 6, 4, 4), "f16", ((1, 6, 4, 4), (96, 16, 4, 1)), "smem"])),
                op=T.ane.conv(groups=3),
            )
