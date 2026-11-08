"""Apple targets for TileFoundry; each target is a subpackage."""

from tilefoundry_apple.amx import AmxTarget, AppleAmx, AppleM2Pro
from tilefoundry_apple.metal import MetalArchitecture, MetalDevice, MetalTarget

__all__ = ["AmxTarget", "AppleAmx", "AppleM2Pro", "MetalArchitecture", "MetalDevice", "MetalTarget"]
