"""The same matmul-into-rms_norm at the size that reaches an MMA atom.

``tests/schedule``'s partition and pipeline tests both need bf16 on a CUDA target
at (64,128)x(128,64): that is what makes the SM80 atom a candidate at all. The
untargeted f32 (2,4) twin in ``gemm_rms_norm`` cannot stand in for it, and it
cannot stand in for that one either -- the sizes carry different subjects.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.target import CudaTarget


@func(target=CudaTarget("nvidia.h200_sxm"))
def bf16_gemm_rms_norm(
    x: Tensor[(64, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
    weight: Tensor[(64,), "f32"],
) -> Tensor[(64, 64), "bf16"]:
    h = tf.matmul(x, w)
    return tf.rms_norm(h, weight)
