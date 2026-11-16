"""One named hardware level with its explicit extent."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Topology", "topology_name"]


@dataclass(frozen=True)
class Topology:
    """Name one hardware level and its explicit static or symbolic size."""

    name: str

    size: "ShapeDim"

    def __post_init__(self) -> None:
        """Refuse a level whose extent is left implicit."""
        if self.size is None:
            raise ValueError(
                f"Topology {self.name!r}: extent must be explicit; None is not "
                "a ShapeDim. The rule: "
                "tilefoundry spec target topology-levels"
            )


def topology_name(topology: "Topology | str") -> str:
    """The name of a topology, or the string itself when one was written as a string."""
    return getattr(topology, "name", topology)
