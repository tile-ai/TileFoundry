"""The two-stage pipeline of the core fixtures, declared for a two-GPU CUDA target."""

from dataclasses import replace

from tests.core.fixtures.autodiff.pipeline import Pipeline
from tilefoundry.dsl import Topology
from tilefoundry_nvidia import cuda

Pipeline = replace(
    Pipeline,
    target=cuda.Target("nvidia.h200_sxm", device_count=2),
    topologies=(*Pipeline.topologies, Topology("cta", 1)),
)
