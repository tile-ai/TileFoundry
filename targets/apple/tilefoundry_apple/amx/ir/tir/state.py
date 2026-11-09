"""AMX SET and CLR Ops."""

from tilefoundry.ir import expr, pattern

from .common import (
    AmxOp,
)
from .semantics import register_semantics

R = expr.MemoryEffect.READ
W = expr.MemoryEffect.WRITE


@expr.register_op(dialect="T", namespace="amx", category="control", name="set")
class Set(AmxOp):
    """Logical enabled-state token; physical scratch registers are evaluator-internal."""

    mnemonic = "SET"
    capability = expr.OpCapability("amx_set")
    resource = "amx_engine"
    roles = ("token",)
    writes = ("token",)
    register_roles = {"token": "lifecycle"}
    token = expr.ParamDef(
        kind="input",
        effect=R | W,
        pattern=pattern.Tensor(
            shape=(1,), dtype=expr.DType.u8, storage=expr.StorageKind.RMEM, layout=pattern.Layout((1,), (1,))
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="control", name="clr")
class Clr(AmxOp):
    """Logical enabled-state token; physical scratch registers are evaluator-internal."""

    mnemonic = "CLR"
    capability = expr.OpCapability("amx_clr")
    resource = "amx_engine"
    roles = ("token",)
    writes = ("token",)
    register_roles = {"token": "lifecycle"}
    token = expr.ParamDef(
        kind="input",
        effect=R | W,
        pattern=pattern.Tensor(
            shape=(1,), dtype=expr.DType.u8, storage=expr.StorageKind.RMEM, layout=pattern.Layout((1,), (1,))
        ),
    )

register_semantics((Set, Clr,))


__all__ = ["Set", "Clr"]
