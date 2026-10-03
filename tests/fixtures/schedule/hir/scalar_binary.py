"""Lower scalar and lhs size-one broadcasts through binary instructions.

Literals stay unmaterialized in HIR; explicit and automatic instructions must
allocate and fill rank-zero register tensors. The final scheduled multiply
broadcasts its size-one lhs over the register tile.
"""

from tilefoundry import func, module
from tilefoundry.dsl import BinaryKind, Mesh, T, Tensor, Topology, tf
from tilefoundry.ir.types import Broadcast, Layout, ShardLayout
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

_THREADS = ThreadMesh((Topology("thread", 32),), Layout((32,), (1,)))
_REG = ShardLayout(Layout((32,), (1,)), (Broadcast(),), _THREADS)
_LHS = ShardLayout(Layout((1,), (1,)), (Broadcast(),), _THREADS)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class ScalarBinary:
    @func
    def gemm(x: Tensor[(32,), "f32"], lhs: Tensor[(1,), "f32"]) -> Tensor[(32,), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                held = tf.schedule((x,), op=T.copy(rmem_layout=_REG))
                shifted = tf.schedule((held, 0.25), op=T.binary(kind=BinaryKind.ADD))
                offset = 1.0 - shifted
                left = tf.schedule((lhs,), op=T.copy(rmem_layout=_LHS))
                return tf.schedule((left, offset), op=T.binary(kind=BinaryKind.MUL))
