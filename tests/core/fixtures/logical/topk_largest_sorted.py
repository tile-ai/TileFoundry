from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="f", target=CpuTarget())
class TopkLargestSorted:
    @func
    def f(x: Tensor[(4, 256), "f32"]):
        v = tf.topk(x, k=6, axis=-1, largest=False, sorted=True)
        return v
