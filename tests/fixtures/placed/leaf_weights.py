"""Modules whose sibling-only weights must stay lazy when one leaf runs."""

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, tf
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.target import CpuTarget, CudaTarget

D, W = 64, 4
PER = 48 * (1 << 30) // 2 // D


@module(entry="entry", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", W),))
class Mod:
    """Sibling weights total 96 GiB, so checking any leaf loads too much today."""

    @func
    def leaf(x: Tensor[(1, D), "f32"]) -> Tensor[(1, D), "f32"]:
        """A leaf with no constants, beside functions with medium weights."""
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            return tf.reshard(xs + xs, (1, D), "gmem")

    @func
    def entry(
        x: Tensor[(1, D), "f32"], w_a: ConstTensor[(PER, D), "bf16"]
    ) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            ws = tf.reshard(w_a[0:1, :], (1, D @ m.w), "smem")
            return tf.reshard(xs + tf.cast(ws, dtype="f32"), (1, D), "gmem")

    @func
    def other(
        x: Tensor[(1, D), "f32"], w_b: ConstTensor[(PER, D), "bf16"]
    ) -> Tensor[(1, D), "f32"]:
        """The second declaration keeps each function below the memory budget."""
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            ws = tf.reshard(w_b[0:1, :], (1, D @ m.w), "smem")
            return tf.reshard(xs * tf.cast(ws, dtype="f32"), (1, D), "gmem")


@module(entry="entry", target=CpuTarget(), topologies=(Topology("cta", W),))
class Small:
    @func
    def leaf(x: Tensor[(1, D), "f32"]) -> Tensor[(1, D), "f32"]:
        """A runnable leaf with no constants."""
        with Mesh(("cta",), layout=(W,), names=("b",)) as m:
            xs = tf.reshard(x, (1, D @ m.b), "rmem")
            return tf.reshard(xs + xs, (1, D), "gmem")

    @func
    def entry(
        x: Tensor[(1, D), "f32"], w: ConstTensor[(1, D), "f32"]
    ) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("b",)) as m:
            xs = tf.reshard(x, (1, D @ m.b), "rmem")
            ws = tf.reshard(w, (1, D @ m.b), "rmem")
            return tf.reshard(tf.mul(xs, ws), (1, D), "gmem")


@runtime_module(Small)
class SmallTwin:
    @runtime_func
    def leaf(self, x):
        return x + x

    @runtime_func
    def entry(self, x, w):
        return x * w
