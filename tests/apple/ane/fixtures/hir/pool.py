"""Max and average pooling offered to ANE, with padding and ceil mode.

ANE refuses a stride beyond the engine's strides. A ceil-mode average divides an
overhanging window by its valid elements only when padding is not counted.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="hir_pool", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def hir_pool(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            maximum = tf.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            average = tf.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            padded = tf.max_pool2d(x, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ceiled = tf.max_pool2d(x, kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
            wide_step = tf.max_pool2d(x, kernel_size=(5, 5), stride=(3, 3))
            ceiled_average = tf.avg_pool2d(x, kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
            ceiled_valid = tf.avg_pool2d(
                x, kernel_size=(3, 3), stride=(2, 2), ceil_mode=True, count_include_pad=False
            )
            return maximum, average, padded, ceiled, wide_step, ceiled_average, ceiled_valid
