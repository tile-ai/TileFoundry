"""Apple AMX target and compiler registrations for TileFoundry."""

from tilefoundry_amx.target import AmxTarget, AppleAmx, AppleM2Pro

__all__ = ["AmxTarget", "AppleAmx", "AppleM2Pro"]

from tilefoundry_amx import ir, visitor_registry  # noqa: E402,F401
