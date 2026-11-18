"""What a Metal target tells the analysis families.

Only the memory hierarchy is projected: it fixes which topology owns each
storage. No Metal throughput is measured yet, so throughput and per-unit
service facts stay unsupported rather than guessed.
"""

from __future__ import annotations

from tilefoundry import target
from tilefoundry.ir import expr


def memory_hierarchy(tgt: target.Target, query: object = None) -> target.MemoryHierarchyFacts:
    """Unified memory shared by the target, threadgroup memory, and per-thread registers."""
    device = tgt.device
    return target.MemoryHierarchyFacts(
        explicit_levels=(
            target.ExplicitMemoryLevelFacts(
                name="gmem",
                capacity_bytes=device.unified_memory_capacity_bytes,
                scope="device",
                owner=expr.TARGET_MEMORY_OWNER,
            ),
            target.ExplicitMemoryLevelFacts(
                name="smem",
                capacity_bytes=device.threadgroup_memory_bytes,
                scope="threadgroup",
                owner="threadgroup",
            ),
            target.ExplicitMemoryLevelFacts(
                name="rmem",
                capacity_bytes=None,
                scope="thread",
                owner="thread",
            ),
        ),
        implicit_levels=(),
        relations=(),
    )


__all__ = ["memory_hierarchy"]
