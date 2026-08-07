"""The machines a model can be asked about besides the one its own source declares.

Each fixture states its levels by asking the target what it has rather than by
repeating a number: a device whose SM count changes should move the fixture with
it, and a fixture that disagreed with the hardware documents would be measuring
something nobody owns.

No fixture is a default. Every model declares the machine its tree runs on and a
build answers with that declaration, so a fixture serves only the separate question
of aiming one model at a second machine in the same run. `apple_m2_pro` is the one
asked that way, and saying so here is deliberate -- it produces no CUDA evidence and
no performance claim, so a report that shows it must not read as if it did.
`h200_sxm` is reached only by the tests that check a fixture agrees with the hardware
documents: asking a corpus model about H200 is what its own source already does.
"""

from __future__ import annotations

from tests.models.corpus import TargetFixture
from tilefoundry.analysis.facts import ParallelCapacityFacts
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget, TopologyLimitFacts
from tilefoundry.target.amx.target import AmxTarget
from tilefoundry.target.base import Target


def _parallel_units(target: Target, level: str) -> int:
    """How many positions of *level* the hardware documents state."""
    facts = target.get_facts(ParallelCapacityFacts)
    if facts.topology != level:
        raise ValueError(
            f"{type(target).__name__} states parallel capacity for "
            f"{facts.topology!r}, not {level!r}"
        )
    return facts.parallel_units


def h200_sxm(*, threads_per_cta: int = 512) -> TargetFixture:
    """One H200 SXM, at both levels CUDA schedules.

    The CTA extent is the device's own SM count, so the fixture divides work over
    exactly as much machine as the documents say exists. The thread extent is a
    launch shape rather than a hardware total, so it is a stated choice, bounded
    by what the architecture admits.
    """
    target = CudaTarget("nvidia.h200_sxm")
    limit = target.get_facts(TopologyLimitFacts, "thread").max_static_extent
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
    amx_limit = target.get_facts(TopologyLimitFacts, "amx").max_static_extent
    return TargetFixture(
        id="apple_m2_pro",
        target=target,
        topologies=(
            Topology("core", _parallel_units(target, "core")),
            Topology("amx", amx_limit if amx_limit is not None else 1),
        ),
    )


__all__ = ["apple_m2_pro", "h200_sxm"]
