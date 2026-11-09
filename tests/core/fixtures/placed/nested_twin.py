"""A placed weighted child reached through an authored parent and its reference."""

from weighted_twin import Weighted, WeightedTwin

from tilefoundry import runtime
from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(target=cpu.Target())
class Nested:
    child = Weighted


@runtime.module(Nested)
class NestedTwin:
    child = WeightedTwin
