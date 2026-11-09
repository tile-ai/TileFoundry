"""AMX GENLUT Ops."""

from tilefoundry.ir import expr, pattern

from .common import (
    AmxOp,
    Bank,
    LutMode,
    instruction_versions,
)
from .semantics import register_semantics

R = expr.MemoryEffect.READ
W = expr.MemoryEffect.WRITE


class Genlut(AmxOp):
    """Packed index source, typed table, and only the written destination register."""

    mnemonic = "GENLUT"
    capability = expr.OpCapability("amx_genlut")
    resource = "amx_engine"
    roles = ("source", "table", "dst")
    writes = ("dst",)
    register_roles = {"source": "source_bank", "table": "table_bank", "dst": "destination"}
    mode = expr.ParamDef(kind="attribute", annotation=LutMode, pattern=pattern.Wildcard("mode"))
    bf16 = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "bf16",
            predicates=(
                (pattern.In(pattern.variable("bf16"), (False,)))
                | ((pattern.variable("amx_generation") >= 2) & (pattern.variable("mode") == LutMode.F16)),
            ),
        ),
        default=False,
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard(
            "dtype",
            predicates=(
                pattern.Switch(
                    "mode",
                    {
                        LutMode.F32: expr.DType.f32,
                        LutMode.F16: pattern.Switch("bf16", {False: expr.DType.f16, True: expr.DType.bf16}),
                        LutMode.F64: expr.DType.f64,
                        LutMode.I32: expr.DType.i32,
                        LutMode.I16: expr.DType.i16,
                        LutMode.U32: expr.DType.u32,
                        LutMode.U16: expr.DType.u16,
                        LutMode.LOOKUP32_2: pattern.Or(expr.DType.i32, expr.DType.u32, expr.DType.f32),
                        LutMode.LOOKUP16_2: pattern.Or(expr.DType.i16, expr.DType.u16, expr.DType.f16, expr.DType.bf16),
                        LutMode.LOOKUP8_2: pattern.Or(expr.DType.i8, expr.DType.u8),
                        LutMode.LOOKUP64_4: expr.DType.f64,
                        LutMode.LOOKUP32_4: pattern.Or(expr.DType.i32, expr.DType.u32, expr.DType.f32),
                        LutMode.LOOKUP16_4: pattern.Or(expr.DType.i16, expr.DType.u16, expr.DType.f16, expr.DType.bf16),
                        LutMode.LOOKUP8_4: pattern.Or(expr.DType.i8, expr.DType.u8),
                        LutMode.LOOKUP16_5: pattern.Or(expr.DType.i16, expr.DType.u16, expr.DType.f16, expr.DType.bf16),
                        LutMode.LOOKUP8_5: pattern.Or(expr.DType.i8, expr.DType.u8),
                    },
                ),
            ),
        ),
    )
    source_bank = expr.ParamDef(
        kind="attribute",
        annotation=Bank,
        pattern=pattern.Wildcard("source_bank", predicates=(pattern.Or(Bank.X, Bank.Y),)),
        default=Bank.X,
    )
    table_bank = expr.ParamDef(
        kind="attribute",
        annotation=Bank,
        pattern=pattern.Wildcard("table_bank", predicates=(pattern.Or(Bank.X, Bank.Y),)),
        default=Bank.Y,
    )
    destination = expr.ParamDef(
        kind="attribute",
        annotation=Bank,
        pattern=pattern.Wildcard(
            "destination",
            predicates=(
                pattern.Or(Bank.X, Bank.Y, Bank.Z),
                (pattern.variable("destination") != Bank.Z) | (pattern.variable("mode") >= LutMode.LOOKUP32_2),
            ),
        ),
        default=Bank.X,
    )
    source_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("source_offset", predicates=(pattern.Range(lo=0, hi=511),)),
        default=0,
    )
    table_register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("table_register", predicates=(pattern.Range(lo=0, hi=7),)),
        default=1,
    )
    destination_register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "destination_register",
            predicates=(
                pattern.Switch(
                    "destination",
                    {
                        Bank.X: pattern.Range(lo=0, hi=7),
                        Bank.Y: pattern.Range(lo=0, hi=7),
                        Bank.Z: pattern.Range(lo=0, hi=63),
                    },
                ),
                pattern.variable("table_lanes") == 512 // pattern.Bits("dtype"),
                pattern.Switch(
                    "mode",
                    {
                        m: pattern.Wildcard(
                            predicates=(
                                pattern.variable("source_lanes") == 512 // pattern.Bits("dtype"),
                                pattern.variable("destination_lanes") == 64,
                            )
                        )
                        if int(m) <= 6
                        else pattern.Wildcard(
                            predicates=(
                                pattern.variable("source_lanes")
                                == {
                                    7: 4,
                                    8: 8,
                                    9: 16,
                                    10: 4,
                                    11: 8,
                                    12: 16,
                                    13: 32,
                                    14: 20,
                                    15: 40,
                                }[int(m)],
                                pattern.variable("destination_lanes") == 512 // pattern.Bits("dtype"),
                            )
                        )
                        for m in LutMode
                    },
                ),
            ),
        ),
        default=2,
    )
    source = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("source_lanes"),),
            dtype=pattern.Switch(
                "mode", {m: pattern.Wildcard("dtype") if int(m) <= 6 else expr.DType.u8 for m in LutMode}
            ),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout((pattern.Wildcard("source_lanes"),), (1,)),
        ),
    )
    table = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("table_lanes"),),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout((pattern.Wildcard("table_lanes"),), (1,)),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("destination_lanes"),),
            dtype=pattern.Switch(
                "mode", {m: expr.DType.u8 if int(m) <= 6 else pattern.Wildcard("dtype") for m in LutMode}
            ),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout((pattern.Wildcard("destination_lanes"),), (1,)),
        ),
    )


GenlutV1, GenlutV2 = instruction_versions(Genlut, (1, 2), name="genlut", category="arith")

register_semantics((GenlutV1, GenlutV2,))


__all__ = ["GenlutV1", "GenlutV2"]
