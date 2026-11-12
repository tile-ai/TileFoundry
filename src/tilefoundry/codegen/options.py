"""Compile-wide source emission options."""

from dataclasses import dataclass
from typing import Literal

from .communication import Communication

EmitGranularity = Literal["full", "skeleton"]


@dataclass(frozen=True)
class Options:
    """Source granularity and the explicit host communication provider."""

    emit: EmitGranularity = "full"
    communication: type[Communication] | None = None

    def __post_init__(self) -> None:
        if self.emit not in ("full", "skeleton"):
            raise ValueError(f"invalid codegen emit granularity: {self.emit!r}")
        if self.communication is not None and (
            not isinstance(self.communication, type) or not issubclass(self.communication, Communication)
        ):
            raise TypeError("codegen communication must be a Communication provider class")


__all__ = ["EmitGranularity", "Options"]
