"""The documented Apple GPU 8 SIMD32 compilation profile."""
from dataclasses import dataclass
from tilefoundry.target.base import Architecture

@dataclass(frozen=True)
class MetalArchitecture(Architecture):
    name: str
    capabilities: tuple[str, ...]
    simd_width: int
    language_version: int
