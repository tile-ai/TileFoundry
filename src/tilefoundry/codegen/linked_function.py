"""A linked function's registration identity and host calling convention."""

from dataclasses import dataclass

from tilefoundry.codegen.signature import CallableSignature


@dataclass(frozen=True)
class LinkedFunction:
    """One library export, preserving its C++ signature separately from its registration name."""

    name: str
    signature: CallableSignature

    @classmethod
    def create(cls, target, symbol) -> "LinkedFunction":
        """Create default export metadata from a resolved function symbol."""
        from tilefoundry.codegen.link_function import link_function  # noqa: PLC0415

        return link_function(symbol)


__all__ = ["LinkedFunction"]
