"""A placed weighted child reached through an authored parent and its reference."""

from weighted_twin import Weighted, WeightedTwin

from tilefoundry import module
from tilefoundry.runtime import runtime_module
from tilefoundry.target import CpuTarget


@module(target=CpuTarget())
class Nested:
    child = Weighted


@runtime_module(Nested)
class NestedTwin:
    child = WeightedTwin
