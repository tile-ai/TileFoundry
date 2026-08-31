"""A dispatch on a callee: check and analyze both select its implementation."""

from tilefoundry import func, module
from tilefoundry.dsl import DimVar, DimVarRangePat, Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

D, W, BOUND = 64, 4, 128
N = DimVar("n", 1, 1024)
_CUDA = CudaTarget("nvidia.h200_sxm")
_CTA = Topology("cta", W)


@module(entry="run", target=_CUDA, topologies=(_CTA,))
class ToCallee:
    """The dispatch is on a callee; the entry calls the prototype."""

    @func
    def pick(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        pass

    @pick.specialize(DimVarRangePat("n", 1, BOUND))
    def pick_small(
        x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]
    ) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            return tf.reshard(xs + xs, (1, D), "gmem")

    @pick.specialize(DimVarRangePat("n", BOUND, 1024))
    def pick_big(
        x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]
    ) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            return tf.reshard(xs + xs + xs, (1, D), "gmem")

    @func
    def run(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        return pick(x, k)


@module(entry="run", target=_CUDA, topologies=(_CTA,))
class Direct:
    """The entry calls one variant body directly."""

    @func
    def pick(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        pass

    @pick.specialize(DimVarRangePat("n", 1, BOUND))
    def pick_small(
        x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]
    ) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            return tf.reshard(xs + xs, (1, D), "gmem")

    @pick.specialize(DimVarRangePat("n", BOUND, 1024))
    def pick_big(
        x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]
    ) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            return tf.reshard(xs + xs + xs, (1, D), "gmem")

    @func
    def run(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        return pick_big(x, k)
