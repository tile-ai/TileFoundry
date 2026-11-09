from tests.nvidia.cuda.fixtures.logical.hir_composition import Expert
from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="fused", target=cuda.Target("nvidia.h200_sxm"))
class _Composed:
    first = Expert
    second = Expert

    @func
    def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return tf.add(first(x), second(x))
