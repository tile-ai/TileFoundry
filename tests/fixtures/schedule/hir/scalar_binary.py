"""Materialize scalar operands for explicit and automatic binary instructions.

The literal stays unmaterialized in HIR. Both instruction paths must allocate
and fill a rank-zero register tensor before issuing T.binary.
"""

from tilefoundry import func, module
from tilefoundry.dsl import BinaryKind, Mesh, T, Tensor, Topology, tf
from tilefoundry.ir.types import Broadcast, Layout, ShardLayout
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

_THREADS = ThreadMesh((Topology("thread", 32),), Layout((32,), (1,)))
_REG = ShardLayout(Layout((32,), (1,)), (Broadcast(),), _THREADS)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class ScalarBinary:
    @func
    def gemm(x: Tensor[(32,), "f32"]) -> Tensor[(32,), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as _threads:
                held = tf.schedule((x,), op=T.copy(rmem_layout=_REG))
                shifted = tf.schedule((held, 0.25), op=T.binary(kind=BinaryKind.ADD))
                return 1.0 - shifted
