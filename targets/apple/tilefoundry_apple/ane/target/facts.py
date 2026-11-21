"""What an ANE target tells the analysis families.

Only the memory hierarchy is projected: it fixes which topology owns each
storage. No ANE throughput is measured yet, so throughput and per-unit
service facts stay unsupported rather than guessed.
"""

from __future__ import annotations

from tilefoundry import target


def memory_hierarchy(tgt: target.Target, query: object = None) -> target.MemoryHierarchyFacts:
    """Unified memory shared by the target, and the on-chip L2 buffer of the engine."""
    device = tgt.device
    return target.MemoryHierarchyFacts(
        explicit_levels=(
            target.ExplicitMemoryLevelFacts(
                name="gmem",
                capacity_bytes=device.unified_memory_capacity_bytes,
                scope="device",
                owner=target.SHARED_OWNER,
            ),
            target.ExplicitMemoryLevelFacts(
                name="smem",
                capacity_bytes=None,
                scope="engine",
                owner="engine",
            ),
        ),
        implicit_levels=(),
        relations=(),
    )


__all__ = ["memory_hierarchy"]
