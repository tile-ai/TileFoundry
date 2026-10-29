from tilefoundry.ir.core import OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    LayoutPattern,
    TensorPattern,
)
from tilefoundry.ir.types import DType, StorageKind

from .common import (
    AmxOp,
    bind,
)

R = MemoryEffect.READ
W = MemoryEffect.WRITE


@register_op(dialect="T", namespace="amx", category="control", name="set")
class Set(AmxOp):
    """Logical enabled-state token; physical scratch registers are evaluator-internal."""

    mnemonic = "SET"
    capability = OpCapability("amx_set")
    resource = "amx_engine"
    roles = ("token",)
    writes = ("token",)
    register_roles = {"token": "lifecycle"}
    token = ParamDef(
        kind="input",
        effect=R | W,
        pattern=TensorPattern(
            shape=(1,), dtype=DType.u8, storage=StorageKind.RMEM, layout=LayoutPattern((1,), (1,))
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="control", name="clr")
class Clr(AmxOp):
    """Logical enabled-state token; physical scratch registers are evaluator-internal."""

    mnemonic = "CLR"
    capability = OpCapability("amx_clr")
    resource = "amx_engine"
    roles = ("token",)
    writes = ("token",)
    register_roles = {"token": "lifecycle"}
    token = ParamDef(
        kind="input",
        effect=R | W,
        pattern=TensorPattern(
            shape=(1,), dtype=DType.u8, storage=StorageKind.RMEM, layout=LayoutPattern((1,), (1,))
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)
