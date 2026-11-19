"""The megatron block program of the core fixtures, declared for a 2-GPU CUDA target."""

from dataclasses import replace

from tests.core.fixtures.autodiff.megatron_block import TensorParallel
from tilefoundry.dsl import Topology
from tilefoundry_nvidia import cuda

TensorParallel = replace(
    TensorParallel,
    target=cuda.Target("nvidia.h200_sxm", device_count=2),
    topologies=(*TensorParallel.topologies, Topology("cta", 1)),
)
