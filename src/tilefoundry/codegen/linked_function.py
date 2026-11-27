"""A linked function's logical identity, DLL symbol and host calling convention."""

from dataclasses import dataclass

from .signature import CallableSignature


@dataclass(frozen=True)
class LinkedFunction:
    """One library export with distinct logical and native names."""

    name: str
    symbol: str
    signature: CallableSignature

    @classmethod
    def create(cls, target, symbol) -> "LinkedFunction":
        """Keep the logical name, native symbol and host signature of a resolved symbol."""
        return cls(name=symbol.export_name, symbol=symbol.symbol, signature=symbol.callable)


__all__ = ["LinkedFunction"]
