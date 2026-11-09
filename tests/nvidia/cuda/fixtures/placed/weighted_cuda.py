"""The placed weighted product of the core fixtures, declared for a CUDA target."""

from dataclasses import replace

from tests.core.fixtures.placed.weighted_twin import Weighted
from tilefoundry_nvidia import cuda

Weighted = replace(Weighted, target=cuda.Target("nvidia.h200_sxm"))
