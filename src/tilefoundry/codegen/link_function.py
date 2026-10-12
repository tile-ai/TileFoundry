"""Resolve one host symbol into linked export metadata."""

from tilefoundry.codegen.context_builder import FunctionSymbol
from tilefoundry.codegen.linked_function import LinkedFunction


def link_function(symbol: FunctionSymbol) -> LinkedFunction:
    """Keep the export name and host signature as distinct identities."""
    return LinkedFunction(name=symbol.export_name, signature=symbol.callable)


__all__ = ["link_function"]
