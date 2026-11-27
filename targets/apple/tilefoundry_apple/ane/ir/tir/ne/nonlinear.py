"""What a neural-engine task applies to its result."""

from enum import Enum
from types import MappingProxyType

from tilefoundry.ir import expr


@expr.register_type(dialect="T", namespace="ane")
class Nonlinear(Enum):
    """What a neural-engine task applies after its arithmetic."""

    NONE = "none"
    RELU = "relu"


KINDS = MappingProxyType(
    {
        Nonlinear.NONE: lambda value: value,
        Nonlinear.RELU: lambda value: value.relu(),
    }
)


def apply(kind: Nonlinear, x):
    """x, or max(x, 0)."""
    return KINDS[kind](x)


__all__ = ["Nonlinear", "apply"]
