"""The placed weighted product of the core fixtures, declared for a CUDA target."""

from dataclasses import replace

from tests.core.fixtures.placed.weighted_twin import Weighted
from tilefoundry_nvidia import CudaTarget

Weighted = replace(Weighted, target=CudaTarget("nvidia.h200_sxm"))
