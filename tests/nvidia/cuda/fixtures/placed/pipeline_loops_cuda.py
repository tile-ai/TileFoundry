"""The two microbatch schedules of the core fixtures, declared for a two-GPU CUDA target."""

from dataclasses import replace

from tests.core.fixtures.autodiff.pipeline_loops import OneLoop, TwoLoops
from tilefoundry.dsl import Topology
from tilefoundry_nvidia import cuda

OneLoop, TwoLoops = (
    replace(
        module,
        target=cuda.Target("nvidia.h200_sxm", device_count=2),
        topologies=(*module.topologies, Topology("cta", 1)),
    )
    for module in (OneLoop, TwoLoops)
)
