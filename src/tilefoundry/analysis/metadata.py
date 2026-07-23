from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.metadata import IRMetadata


@dataclass(frozen=True)
class TrafficBytes:
    """Read/write byte counts for one memory hierarchy level."""

    read_bytes: int = 0
    write_bytes: int = 0


@dataclass(frozen=True)
class RooflineMetadata(IRMetadata):
    flops: tuple[tuple[str, int], ...] = ()
    traffic: tuple[tuple[str, TrafficBytes], ...] = ()
    theoretical_ns: int = 0

    def format_comment(self) -> str:
        flop_text = ",".join(f"{name}:{value}" for name, value in self.flops) or "0"
        memory_text = ",".join(
            f"{level}:r{traffic.read_bytes}/w{traffic.write_bytes}"
            for level, traffic in self.traffic
        ) or "0"
        return f"roofline flops={flop_text} bytes={memory_text} bound={self.theoretical_ns}ns"


@dataclass(frozen=True)
class FootprintMetadata(IRMetadata):
    live_bytes: tuple[tuple[str, int], ...] = ()

    def format_comment(self) -> str:
        text = ",".join(f"{level}:{value}" for level, value in self.live_bytes) or "0"
        return f"footprint live={text}"


@dataclass(frozen=True)
class TimelineMetadata(IRMetadata):
    grid_ctas: int = 1
    waves: int = 1
    start_ns: int = 0
    end_ns: int = 0

    def format_comment(self) -> str:
        return (
            f"timeline ctas={self.grid_ctas} waves={self.waves} "
            f"start={self.start_ns}ns end={self.end_ns}ns"
        )


__all__ = [
    "FootprintMetadata",
    "RooflineMetadata",
    "TimelineMetadata",
    "TrafficBytes",
]
