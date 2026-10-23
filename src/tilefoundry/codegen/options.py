"""Compile-wide source emission options."""

from dataclasses import dataclass
from typing import Literal

EmitGranularity = Literal["full", "skeleton"]


@dataclass(frozen=True)
class CodegenOptions:
    """Whether emitted bodies contain statements or placeholders."""

    emit: EmitGranularity = "full"

    def __post_init__(self) -> None:
        if self.emit not in ("full", "skeleton"):
            raise ValueError(f"invalid codegen emit granularity: {self.emit!r}")
