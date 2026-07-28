"""The machines the corpus can ask a model about.

Each fixture states its levels by asking the target what it has rather than by
repeating a number: a device whose SM count changes should move the fixture with
it, and a fixture that disagreed with the hardware documents would be measuring
something nobody owns.

H200 is the acceptance machine. The Apple fixture is a local convenience, and
saying so here is deliberate -- it produces no CUDA evidence and no performance
claim, so a report that shows it must not read as if it did.
"""

from __future__ import annotations

from tests.models.corpus import TargetFixture
from tilefoundry.analysis.facts import ParallelCapacityFacts
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget
from tilefoundry.target.amx.target import AmxTarget
from tilefoundry.target.base import Target


def _parallel_units(target: Target, level: str) -> int:
    """How many positions of *level* the hardware documents state."""
    facts = target.as_facts(ParallelCapacityFacts)
    if facts.topology != level:
        raise ValueError(
            f"{type(target).__name__} states parallel capacity for "
            f"{facts.topology!r}, not {level!r}"
        )
    return facts.parallel_units


def h200_sxm(*, threads_per_cta: int = 128) -> TargetFixture:
    """One H200 SXM, at both levels CUDA schedules.

    The CTA extent is the device's own SM count, so the fixture divides work over
    exactly as much machine as the documents say exists. The thread extent is a
    launch shape rather than a hardware total, so it is a stated choice, bounded
    by what the architecture admits.
    """
    target = CudaTarget()
    limit = target.topology_limit("thread")
    if limit is not None and not 1 <= threads_per_cta <= limit:
        raise ValueError(
            f"h200_sxm: {threads_per_cta} threads per CTA is outside the "
            f"architecture's limit of {limit}"
        )
    return TargetFixture(
        id="h200_sxm",
        target=target,
        topologies=(
            Topology("cta", _parallel_units(target, "cta")),
            Topology("thread", threads_per_cta),
        ),
    )


def apple_m2_pro() -> TargetFixture:
    """One Apple M2 Pro with its AMX units, for local feedback only.

    Both levels come from the target: the performance-core count it publishes,
    and the single AMX unit a core issues to.
    """
    target = AmxTarget()
    amx_limit = target.topology_limit("amx")
    return TargetFixture(
        id="apple_m2_pro",
        target=target,
        topologies=(
            Topology("core", _parallel_units(target, "core")),
            Topology("amx", amx_limit if amx_limit is not None else 1),
        ),
    )


#: The machine every model in the corpus is accepted on.
ACCEPTANCE = h200_sxm


__all__ = ["ACCEPTANCE", "apple_m2_pro", "h200_sxm"]
