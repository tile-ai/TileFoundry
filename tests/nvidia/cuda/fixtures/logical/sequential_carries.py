from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry='gemm', target=CudaTarget('nvidia.h200_sxm'),
        topologies=(Topology('cta', 1), Topology('thread', 1)))
class SequentialCarries:
    @func
    def gemm(x: Tensor[(2, 4), 'f32'], y: Tensor[(2, 4), 'f32'], z: Tensor[(2, 4), 'f32']):
        with Mesh(('cta',), layout=(1,), names=('block',)) as _cta:
            with Mesh(('thread',), layout=(1,), names=('lane',)) as _thread:
                out = tf.zeros(Tensor[(6, 4), 'f32'])
                for j in range(2):
                    out = tf.insert_slice(out, x[j:j + 1, :], (j, 0))
                for j in range(2):
                    out = tf.insert_slice(out, y[j:j + 1, :], (2 + j, 0))
                    out = tf.insert_slice(out, z[j:j + 1, :], (4 + j, 0))
                return out
