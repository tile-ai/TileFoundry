"""AMX target facts."""

from .architecture import AppleAmx
from .device import AppleM2Pro
from .target import AmxTarget

__all__ = ["AmxTarget", "AppleAmx", "AppleM2Pro"]
