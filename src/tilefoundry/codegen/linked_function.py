"""A linked function's registration identity and host calling convention."""

from dataclasses import dataclass

from .signature import CallableSignature


@dataclass(frozen=True)
class LinkedFunction:
    """One library export, preserving its C++ signature separately from its registration name."""

    name: str
    signature: CallableSignature

    @classmethod
    def create(cls, target, symbol) -> "LinkedFunction":
        """Keep the export name and host signature of a resolved symbol as distinct identities."""
        return cls(name=symbol.export_name, signature=symbol.callable)


__all__ = ["LinkedFunction"]
