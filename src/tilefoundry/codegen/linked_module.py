"""A linked library and the functions it exports."""

from dataclasses import dataclass
from pathlib import Path

from tilefoundry.codegen.linked_function import LinkedFunction


@dataclass(frozen=True)
class LinkedModule:
    """Library artifact, diagnostic source and exported host functions."""

    library_path: Path
    source: str
    functions: tuple[LinkedFunction, ...]


__all__ = ["LinkedModule"]
