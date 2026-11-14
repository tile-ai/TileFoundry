"""Typed host resources used by asynchronous communication statements."""

from dataclasses import dataclass

from .mesh import Mesh

__all__ = ["CommunicationHandleType"]


@dataclass(frozen=True)
class CommunicationHandleType:
    """A communicator, stream or event identity, independent of a native handle.

    See [tir §2.4](docs/spec/tir.md#24-host-communication).
    """

    kind: str
    name: str
    participants: Mesh | None = None

    def __post_init__(self):
        if self.kind not in {"communicator", "stream", "event"} or not self.name:
            raise ValueError("communication handles require a kind and nonempty identity")
        if (self.kind == "communicator") != isinstance(self.participants, Mesh):
            raise ValueError("only communicator handles require a participant mesh")
