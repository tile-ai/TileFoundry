"""The expert-parallel program of the core fixtures, declared for a two-GPU CUDA target."""

from dataclasses import replace

from tests.core.fixtures.autodiff.expert_parallel import ExpertParallel
from tilefoundry.dsl import Topology
from tilefoundry_nvidia import cuda

ExpertParallel = replace(
    ExpertParallel,
    target=cuda.Target("nvidia.h200_sxm", device_count=2),
    topologies=(*ExpertParallel.topologies, Topology("cta", 1)),
)
