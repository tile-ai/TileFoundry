"""Destination parameter convention shared by HIR lowering and host launches."""

from tilefoundry.ir.core import Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import TensorType, TupleType, UnitType
from tilefoundry.ir.types.storage import StorageKind


def fresh_name(stem: str, used: set[str]) -> str:
    """Reserve the stem or its first available decimal suffix."""
    candidate = stem
    index = 0
    while candidate in used:
        index += 1
        candidate = f"{stem}_{index}"
    used.add(candidate)
    return candidate


def destination_params(fn: Function) -> tuple[Var, ...]:
    """Declared inputs followed by tensor result leaves as destination parameters."""
    used = {param.name for param in fn.params}

    def outputs(ty):
        if isinstance(ty, TensorType):
            yield Var(
                fresh_name("out", used),
                type=TensorType(tuple(ty.shape), ty.dtype, None, StorageKind.GMEM),
            )
        elif isinstance(ty, TupleType):
            for field in ty.fields:
                yield from outputs(field)
        elif not isinstance(ty, UnitType):
            raise TypeError(f"destination_params: unsupported result type {ty!r} in {fn.name!r}")

    return (*fn.params, *outputs(fn.return_type))
