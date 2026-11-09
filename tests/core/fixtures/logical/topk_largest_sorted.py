from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="f", target=cpu.Target())
class TopkLargestSorted:
    @func
    def f(x: Tensor[(4, 256), "f32"]):
        v = tf.topk(x, k=6, axis=-1, largest=False, sorted=True)
        return v
