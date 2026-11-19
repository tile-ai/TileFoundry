"""The hsdp block program of the core fixtures, declared for a 4-GPU CUDA target."""

from dataclasses import replace

from tests.core.fixtures.autodiff.hsdp_block import HsdpBlock
from tilefoundry.dsl import Topology
from tilefoundry_nvidia import cuda

HsdpBlock = replace(
    HsdpBlock,
    target=cuda.Target("nvidia.h200_sxm", device_count=4),
    topologies=(*HsdpBlock.topologies, Topology("cta", 1)),
)
