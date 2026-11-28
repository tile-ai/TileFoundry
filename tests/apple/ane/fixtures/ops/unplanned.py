"""Scheduled transfers that lowering or code generation refuses; each breaks one transfer rule."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class StoreLoaded:
    """A store whose source is a load, with no computation between them."""

    @func
    def main(x: Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (768, 192, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (384, 96, 16, 1)), "smem"])), op=T.ane.load())
            return tf.schedule((xs, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (768, 192, 32, 1)), "gmem"])), op=T.ane.store())


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class StoreRead:
    """A stored result that a later task reads again."""

    @func
    def main(x: Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (768, 192, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (384, 96, 16, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((xs, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (384, 96, 16, 1)), "smem"])), op=T.ane.unary(abs=True))
            zs = tf.schedule((ys, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (384, 96, 16, 1)), "smem"])), op=T.ane.unary(bias=1.0))
            out = tf.empty(Tensor[(1, 8, 6, 16), "f16", ((1, 8, 6, 16), (1536, 192, 32, 1)), "gmem"])
            out = tf.insert_slice(out, tf.schedule((ys, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (1536, 192, 32, 1)), "gmem"])), op=T.ane.store()), (0, 0, 0, 0))
            return tf.insert_slice(out, tf.schedule((zs, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (1536, 192, 32, 1)), "gmem"])), op=T.ane.store()), (0, 4, 0, 0))


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Unstored:
    """An output that the entry returns from L2 instead of storing it."""

    @func
    def main(x: Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (768, 192, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (384, 96, 16, 1)), "smem"])), op=T.ane.load())
            return tf.schedule((xs, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (384, 96, 16, 1)), "smem"])), op=T.ane.unary(abs=True))


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class StoredTwice:
    """One L2 result stored into two windows of the output."""

    @func
    def main(x: Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (768, 192, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (384, 96, 16, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((xs, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (384, 96, 16, 1)), "smem"])), op=T.ane.unary(abs=True))
            out = tf.empty(Tensor[(1, 8, 6, 16), "f16", ((1, 8, 6, 16), (1536, 192, 32, 1)), "gmem"])
            out = tf.insert_slice(out, tf.schedule((ys, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (1536, 192, 32, 1)), "gmem"])), op=T.ane.store()), (0, 0, 0, 0))
            return tf.insert_slice(out, tf.schedule((ys, tf.empty(Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (1536, 192, 32, 1)), "gmem"])), op=T.ane.store()), (0, 4, 0, 0))


@module(entry="main", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Copied:
    """An input inserted into a window of the output, which needs a copy."""

    @func
    def main(x: Tensor[(1, 4, 6, 16), "f16", ((1, 4, 6, 16), (768, 192, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            out = tf.empty(Tensor[(1, 8, 6, 16), "f16", ((1, 8, 6, 16), (1536, 192, 32, 1)), "gmem"])
            return tf.insert_slice(out, x, (0, 4, 0, 0))
