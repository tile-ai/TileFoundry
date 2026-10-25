from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry='gemm', target=CudaTarget('nvidia.h200_sxm'),
        topologies=(Topology('cta', 1), Topology('thread', 1)))
class FreshReturn:
    @func
    def doubled(x: Tensor[(2, 4), 'f32']):
        return x + x

    @func
    def gemm():
        with Mesh(('cta',), layout=(1,), names=('block',)) as _cta:
            a = tf.zeros(Tensor[(2, 4), 'f32'])
            return doubled(a), a
