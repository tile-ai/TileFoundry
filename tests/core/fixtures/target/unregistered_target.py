"""A Module declared on an instance of a Target class that was never registered."""

from __future__ import annotations

from tilefoundry import target as tilefoundry_target
from tilefoundry.dsl import *


class UnregisteredTarget(tilefoundry_target.Target):
    name = "tests.core.target.unregistered"


target = UnregisteredTarget()


@module(target=target)
class Decorated:
    def forward(self):
        return None
