"""A Module declared on an instance of a Target class that was never registered."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import Target


class UnregisteredTarget(Target):
    name = "tests.core.target.unregistered"


target = UnregisteredTarget()


@module(target=target)
class Decorated:
    def forward(self):
        return None
