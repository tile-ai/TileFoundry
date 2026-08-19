"""One SM80 16x8x16 BF16 MMA tile with an FP32 result, on one CTA."""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.target import CudaTarget

_THREAD = Topology("thread", 32)
_THREAD_MESH = Mesh(
    topologies=(_THREAD,),
    layout=Layout(shape=(4, 8), strides=(1, 4)),
    names=("x", "y"),
)

_A_FRAG = ShardLayout(
    layout=Layout(shape=(2, 4, 2, 8, 2), strides=(1, 2, 8, 16, 128)),
    attrs=(Split(1), Split(3)),
    mesh=_THREAD_MESH,
)
_B_FRAG = ShardLayout(
    layout=Layout(shape=(8, 2, 4, 2), strides=(1, 8, 16, 64)),
    attrs=(Split(2), Split(0)),
    mesh=_THREAD_MESH,
)
_C_FRAG = ShardLayout(
    layout=Layout(shape=(2, 4, 8, 2), strides=(1, 2, 8, 64)),
    attrs=(Split(1), Split(2)),
    mesh=_THREAD_MESH,
)


@module(
    entry="matmul_16x8x16",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), _THREAD),
)
class MatmulModule:
    @func
    def matmul_16x8x16(
        a: Tensor[(16, 16), "bf16"],
        b: Tensor[(16, 8), "bf16"],
    ) -> Tensor[(16, 8), "f32"]:
        with Mesh(("cta",), (1,), ("tile",)) as _cta:
            a_frag = reshard(a, layout=_A_FRAG, storage="rmem")
            b_frag = reshard(b, layout=_B_FRAG, storage="rmem")
            c_frag = mma_sm80_16x8x16(
                a_frag,
                b_frag,
                dtype_a="bf16",
                dtype_b="bf16",
                dtype_acc="f32",
            )
            return reshard(c_frag, layout=_C_FRAG, storage="gmem")
