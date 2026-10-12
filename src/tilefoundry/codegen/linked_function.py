"""A linked function's registration identity and host calling convention."""

from dataclasses import dataclass

from tilefoundry.codegen.signature import CallableSignature


@dataclass(frozen=True)
class LinkedFunction:
    """One library export, preserving its C++ signature separately from its registration name."""

    name: str
    signature: CallableSignature


__all__ = ["LinkedFunction"]
