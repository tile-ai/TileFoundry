"""Metal descriptions and evaluators; no code generation or GPU initialization."""
from .target import MetalArchitecture, MetalDevice, MetalTarget

__all__ = ["MetalArchitecture", "MetalDevice", "MetalTarget"]

from . import ir as ir
from . import passes as passes
