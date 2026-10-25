"""``@func`` with and without declared execution context.

A Target or a topology declaration (an explicit empty one included) makes the
decorator yield the Module that owns the function; ``plain`` declares neither
and stays a bare Function, which is what the test reads.
"""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.logical.module_context import CONTEXT_CTA
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(CONTEXT_CTA,))
def standalone(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return tf.relu(x)


@func(topologies=())
def empty(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return tf.relu(x)


@func
def plain(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return tf.relu(x)
