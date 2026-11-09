"""AMX EXTRX and EXTRY Ops."""

from tilefoundry.interfaces import schedule
from tilefoundry.ir import expr, pattern
from tilefoundry.ir.hir import sharding, tensor

from .common import (
    I8,
    I16,
    I32,
    TYPES,
    AmxOp,
    Bank,
    ExtractMode,
    Repeat,
    instruction_versions,
)
from .semantics import register_semantics

R = expr.MemoryEffect.READ
W = expr.MemoryEffect.WRITE


EXTRACT_SRC_DTYPE = pattern.Wildcard(
    "src_dtype",
    predicates=(
        pattern.Switch(
            "mode",
            {
                ExtractMode.COPY: pattern.Or(*TYPES),
                ExtractMode.LEGACY: pattern.Or(
                    expr.DType.i16, expr.DType.u16, expr.DType.f16, expr.DType.i32, expr.DType.u32, expr.DType.f32, expr.DType.f64
                ),
                ExtractMode.INTEGER: pattern.Or(*I8, *I16, *I32),
                ExtractMode.FLOAT: pattern.Or(expr.DType.f16, expr.DType.f32, expr.DType.f64),
            },
        ),
    ),
)

EXTRACT_DST_DTYPE = pattern.Wildcard(
    "dst_dtype",
    predicates=(
        pattern.Switch(
            "mode",
            {
                ExtractMode.COPY: pattern.Wildcard("src_dtype"),
                ExtractMode.LEGACY: pattern.Wildcard("src_dtype"),
                ExtractMode.INTEGER: pattern.Wildcard(
                    predicates=(
                        pattern.In(pattern.variable("dst_dtype"), (*I8, *I16, *I32)),
                        pattern.Bits("dst_dtype") <= pattern.Bits("src_dtype"),
                    )
                ),
                ExtractMode.FLOAT: pattern.Switch(
                    "src_dtype",
                    {
                        expr.DType.f16: expr.DType.f16,
                        expr.DType.f32: pattern.Or(expr.DType.f32, expr.DType.f16, expr.DType.bf16),
                        expr.DType.f64: expr.DType.f64,
                    },
                ),
            },
        ),
        (pattern.variable("mode") != ExtractMode.FLOAT)
        | (pattern.variable("amx_generation") >= 2)
        | (pattern.variable("dst_dtype") == pattern.variable("src_dtype")),
    ),
)


class Extrx(AmxOp):
    hir_matches = (
        schedule.HirMatch(
            sharding.Reshard, attributes={"mode": ExtractMode.COPY}, operands={"src": "x", "dst": "result"}
        ),
        schedule.HirMatch(
            tensor.Cast,
            attributes={
                "mode": ExtractMode.INTEGER,
                "shift": 0,
                "rounding": False,
                "saturate": False,
            },
            operands={"src": "x", "dst": "result"},
        ),
        schedule.HirMatch(
            tensor.Cast, attributes={"mode": ExtractMode.FLOAT}, operands={"src": "x", "dst": "result"}
        ),
    )
    mnemonic = "EXTRX"
    capability = expr.OpCapability("amx_extrx")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "extract_source", "dst": "destination"}
    mode = expr.ParamDef(
        kind="attribute",
        annotation=ExtractMode,
        pattern=pattern.Wildcard("mode"),
        default=ExtractMode.LEGACY,
    )
    src_dtype = expr.ParamDef(
        kind="attribute", annotation=expr.DType, pattern=EXTRACT_SRC_DTYPE, default=expr.DType.f32
    )
    dst_dtype = expr.ParamDef(
        kind="attribute", annotation=expr.DType, pattern=EXTRACT_DST_DTYPE, default=expr.DType.f32
    )
    destination = expr.ParamDef(
        kind="attribute",
        annotation=Bank,
        pattern=pattern.Wildcard(
            "destination",
            predicates=(
                pattern.Or(Bank.X, Bank.Y),
                pattern.In(pattern.variable("mode"), (ExtractMode.INTEGER, ExtractMode.FLOAT))
                | (pattern.variable("destination") == Bank.X),
            ),
        ),
        default=Bank.X,
    )
    z_selector = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "z_selector",
            predicates=(
                pattern.Range(lo=0, hi=63),
                (pattern.variable("mode") != ExtractMode.COPY) | (pattern.variable("z_selector") <= 7),
            ),
        ),
        default=0,
    )
    destination_register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "destination_register",
            predicates=(
                pattern.Range(lo=0, hi=7),
                (pattern.variable("mode") == ExtractMode.COPY) | (pattern.variable("destination_register") == 0),
            ),
        ),
        default=0,
    )
    destination_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "destination_offset",
            predicates=(
                pattern.Range(lo=0, hi=511),
                (pattern.variable("mode") != ExtractMode.COPY) | (pattern.variable("destination_offset") == 0),
            ),
        ),
        default=0,
    )
    repeat = expr.ParamDef(
        kind="attribute",
        annotation=Repeat,
        pattern=pattern.Wildcard(
            "repeat",
            predicates=(
                (pattern.variable("repeat") == Repeat.ONE)
                | (
                    (pattern.variable("amx_generation") >= 2)
                    & pattern.In(pattern.variable("mode"), (ExtractMode.INTEGER, ExtractMode.FLOAT))
                ),
            ),
        ),
        default=Repeat.ONE,
    )
    alternating = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "alternating",
            predicates=(
                (pattern.In(pattern.variable("alternating"), (False,)))
                | (
                    pattern.In(pattern.variable("mode"), (ExtractMode.INTEGER, ExtractMode.FLOAT))
                    & (pattern.Bits("src_dtype") == 32)
                ),
            ),
        ),
        default=False,
    )
    low_byte_only = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "low_byte_only",
            predicates=(
                (pattern.In(pattern.variable("low_byte_only"), (False,)))
                | ((pattern.variable("mode") == ExtractMode.LEGACY) & (pattern.Bits("src_dtype") == 16)),
            ),
        ),
        default=False,
    )
    shift = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "shift",
            predicates=(
                pattern.Range(lo=0, hi=31),
                (pattern.variable("mode") == ExtractMode.INTEGER) | (pattern.variable("shift") == 0),
            ),
        ),
        default=0,
    )
    rounding = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "rounding",
            predicates=(
                (pattern.In(pattern.variable("rounding"), (False,))) | (pattern.variable("mode") == ExtractMode.INTEGER),
            ),
        ),
        default=False,
    )
    saturate = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "saturate",
            predicates=(
                (pattern.In(pattern.variable("saturate"), (False,))) | (pattern.variable("mode") == ExtractMode.INTEGER),
            ),
        ),
        default=False,
    )
    mask = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "mask",
            predicates=(
                pattern.Switch(
                    "mode",
                    {
                        ExtractMode.COPY: 0,
                        ExtractMode.LEGACY: pattern.Range(lo=0, hi=127),
                        ExtractMode.INTEGER: pattern.Range(lo=0, hi=511),
                        ExtractMode.FLOAT: pattern.Range(lo=0, hi=511),
                    },
                ),
                pattern.variable("src_lanes") == 64 // (pattern.Bits("src_dtype") // 8),
                pattern.variable("dst_lanes") == 64 // (pattern.Bits("dst_dtype") // 8),
                pattern.variable("rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("repeat")],
                pattern.variable("extract_ratio") == pattern.Bits("src_dtype") // pattern.Bits("dst_dtype"),
                pattern.variable("extract_inner") == 512 // pattern.Bits("src_dtype"),
                pattern.variable("source_row")
                == (64 // pattern.Table((0, 1, 2, 0, 4))[pattern.variable("repeat")])
                * 64
                // (pattern.Bits("src_dtype") // 8),
                pattern.Switch(
                    "alternating",
                    {
                        False: pattern.Wildcard(
                            predicates=(pattern.variable("extract_step") == 64 // (pattern.Bits("src_dtype") // 8),)
                        ),
                        True: pattern.Switch(
                            "dst_dtype",
                            {
                                d: pattern.Wildcard(
                                    predicates=(
                                        pattern.variable("extract_step")
                                        == (2 if d.bit_width == 16 else 1)
                                        * 64
                                        // (pattern.Bits("src_dtype") // 8),
                                    )
                                )
                                for d in TYPES
                            },
                        ),
                    },
                ),
            ),
        ),
        default=0,
    )
    src = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Switch(
            "mode",
            {
                ExtractMode.COPY: pattern.Tensor(
                    shape=(1, pattern.Wildcard("src_lanes")),
                    dtype=pattern.Wildcard("src_dtype"),
                    storage=expr.StorageKind.RMEM,
                    layout=pattern.Layout((1, pattern.Wildcard("src_lanes")), (0, 1)),
                ),
                ExtractMode.LEGACY: pattern.Tensor(
                    shape=(1, 1, pattern.Wildcard("src_lanes")),
                    dtype=pattern.Wildcard("src_dtype"),
                    storage=expr.StorageKind.RMEM,
                    layout=pattern.Layout(
                        (1, 1, pattern.Wildcard("src_lanes")),
                        (pattern.Wildcard("source_row"), pattern.Wildcard("src_lanes"), 1),
                    ),
                ),
                **{
                    m: pattern.Tensor(
                        shape=(pattern.Wildcard("rows"), pattern.Wildcard("dst_lanes")),
                        dtype=pattern.Wildcard("src_dtype"),
                        storage=expr.StorageKind.RMEM,
                        layout=pattern.Switch(
                            "extract_ratio",
                            {
                                1: pattern.Layout(
                                    (pattern.Wildcard("rows"), pattern.Wildcard("dst_lanes")),
                                    (pattern.Wildcard("source_row"), 1),
                                ),
                                **{
                                    ratio: pattern.Layout(
                                        (
                                            pattern.Wildcard("rows"),
                                            (
                                                pattern.Wildcard("extract_inner"),
                                                pattern.Wildcard("extract_ratio"),
                                            ),
                                        ),
                                        (
                                            pattern.Wildcard("source_row"),
                                            (1, pattern.Wildcard("extract_step")),
                                        ),
                                    )
                                    for ratio in (2, 4)
                                },
                            },
                        ),
                    )
                    for m in (ExtractMode.INTEGER, ExtractMode.FLOAT)
                },
            },
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=R | W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("rows"), pattern.Wildcard("dst_lanes")),
            dtype=pattern.Wildcard("dst_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("rows"), pattern.Wildcard("dst_lanes")),
                (pattern.Wildcard("dst_lanes"), 1),
            ),
        ),
    )


class Extry(AmxOp):
    hir_matches = (
        schedule.HirMatch(
            sharding.Reshard, attributes={"mode": ExtractMode.COPY}, operands={"src": "x", "dst": "result"}
        ),
        schedule.HirMatch(
            tensor.Cast,
            attributes={
                "mode": ExtractMode.INTEGER,
                "shift": 0,
                "rounding": False,
                "saturate": False,
            },
            operands={"src": "x", "dst": "result"},
        ),
        schedule.HirMatch(
            tensor.Cast, attributes={"mode": ExtractMode.FLOAT}, operands={"src": "x", "dst": "result"}
        ),
    )
    mnemonic = "EXTRY"
    capability = expr.OpCapability("amx_extry")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "extract_source", "dst": "destination"}
    mode = expr.ParamDef(
        kind="attribute",
        annotation=ExtractMode,
        pattern=pattern.Wildcard("mode"),
        default=ExtractMode.LEGACY,
    )
    src_dtype = expr.ParamDef(
        kind="attribute", annotation=expr.DType, pattern=EXTRACT_SRC_DTYPE, default=expr.DType.f32
    )
    dst_dtype = expr.ParamDef(
        kind="attribute", annotation=expr.DType, pattern=EXTRACT_DST_DTYPE, default=expr.DType.f32
    )
    destination = expr.ParamDef(
        kind="attribute",
        annotation=Bank,
        pattern=pattern.Wildcard(
            "destination",
            predicates=(
                pattern.Or(Bank.X, Bank.Y),
                pattern.In(pattern.variable("mode"), (ExtractMode.INTEGER, ExtractMode.FLOAT))
                | (pattern.variable("destination") == Bank.Y),
            ),
        ),
        default=Bank.Y,
    )
    z_selector = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "z_selector",
            predicates=(
                pattern.Range(lo=0, hi=63),
                (pattern.variable("mode") != ExtractMode.COPY) | (pattern.variable("z_selector") <= 7),
            ),
        ),
        default=0,
    )
    destination_register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "destination_register",
            predicates=(
                pattern.Range(lo=0, hi=7),
                (pattern.variable("mode") == ExtractMode.COPY) | (pattern.variable("destination_register") == 0),
            ),
        ),
        default=0,
    )
    destination_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "destination_offset",
            predicates=(
                pattern.Range(lo=0, hi=511),
                (pattern.variable("mode") != ExtractMode.COPY) | (pattern.variable("destination_offset") == 0),
            ),
        ),
        default=0,
    )
    repeat = expr.ParamDef(
        kind="attribute",
        annotation=Repeat,
        pattern=pattern.Wildcard(
            "repeat",
            predicates=(
                (pattern.variable("repeat") == Repeat.ONE)
                | (
                    (pattern.variable("amx_generation") >= 2)
                    & pattern.In(pattern.variable("mode"), (ExtractMode.INTEGER, ExtractMode.FLOAT))
                ),
            ),
        ),
        default=Repeat.ONE,
    )
    alternating = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "alternating",
            predicates=(
                (pattern.In(pattern.variable("alternating"), (False,)))
                | (
                    pattern.In(pattern.variable("mode"), (ExtractMode.INTEGER, ExtractMode.FLOAT))
                    & (pattern.Bits("src_dtype") == 32)
                ),
            ),
        ),
        default=False,
    )
    low_byte_only = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "low_byte_only",
            predicates=(
                (pattern.In(pattern.variable("low_byte_only"), (False,)))
                | ((pattern.variable("mode") == ExtractMode.LEGACY) & (pattern.Bits("src_dtype") == 16)),
            ),
        ),
        default=False,
    )
    shift = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "shift",
            predicates=(
                pattern.Range(lo=0, hi=31),
                (pattern.variable("mode") == ExtractMode.INTEGER) | (pattern.variable("shift") == 0),
            ),
        ),
        default=0,
    )
    rounding = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "rounding",
            predicates=(
                (pattern.In(pattern.variable("rounding"), (False,))) | (pattern.variable("mode") == ExtractMode.INTEGER),
            ),
        ),
        default=False,
    )
    saturate = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "saturate",
            predicates=(
                (pattern.In(pattern.variable("saturate"), (False,))) | (pattern.variable("mode") == ExtractMode.INTEGER),
            ),
        ),
        default=False,
    )
    mask = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "mask",
            predicates=(
                pattern.Switch(
                    "mode",
                    {
                        ExtractMode.COPY: 0,
                        ExtractMode.LEGACY: pattern.Range(lo=0, hi=127),
                        ExtractMode.INTEGER: pattern.Range(lo=0, hi=511),
                        ExtractMode.FLOAT: pattern.Range(lo=0, hi=511),
                    },
                ),
                pattern.variable("src_lanes") == 64 // (pattern.Bits("src_dtype") // 8),
                pattern.variable("dst_lanes") == 64 // (pattern.Bits("dst_dtype") // 8),
                pattern.variable("rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("repeat")],
                pattern.variable("extract_ratio") == pattern.Bits("src_dtype") // pattern.Bits("dst_dtype"),
                pattern.variable("extract_inner") == 512 // pattern.Bits("src_dtype"),
                pattern.variable("source_row")
                == 64 // pattern.Table((0, 1, 2, 0, 4))[pattern.variable("repeat")] // (pattern.Bits("src_dtype") // 8),
                pattern.Switch(
                    "alternating",
                    {
                        False: pattern.Wildcard(
                            predicates=(pattern.variable("extract_step") == 64 // (pattern.Bits("src_dtype") // 8),)
                        ),
                        True: pattern.Switch(
                            "dst_dtype",
                            {
                                d: pattern.Wildcard(
                                    predicates=(
                                        pattern.variable("extract_step")
                                        == (2 if d.bit_width == 16 else 1)
                                        * 64
                                        // (pattern.Bits("src_dtype") // 8),
                                    )
                                )
                                for d in TYPES
                            },
                        ),
                    },
                ),
            ),
        ),
        default=0,
    )
    src = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Switch(
            "mode",
            {
                ExtractMode.COPY: pattern.Tensor(
                    shape=(1, pattern.Wildcard("src_lanes")),
                    dtype=pattern.Wildcard("src_dtype"),
                    storage=expr.StorageKind.RMEM,
                    layout=pattern.Layout((1, pattern.Wildcard("src_lanes")), (0, 1)),
                ),
                ExtractMode.LEGACY: pattern.Tensor(
                    shape=(pattern.Wildcard("rows"), 64),
                    dtype=pattern.Wildcard("src_dtype"),
                    storage=expr.StorageKind.RMEM,
                    layout=pattern.Layout(
                        (pattern.Wildcard("rows"), 64),
                        (pattern.Wildcard("source_row"), pattern.Wildcard("src_lanes")),
                    ),
                ),
                **{
                    m: pattern.Tensor(
                        shape=(pattern.Wildcard("rows"), pattern.Wildcard("dst_lanes")),
                        dtype=pattern.Wildcard("src_dtype"),
                        storage=expr.StorageKind.RMEM,
                        layout=pattern.Switch(
                            "extract_ratio",
                            {
                                1: pattern.Layout(
                                    (pattern.Wildcard("rows"), pattern.Wildcard("dst_lanes")),
                                    (pattern.Wildcard("source_row"), 64),
                                ),
                                **{
                                    ratio: pattern.Layout(
                                        (
                                            pattern.Wildcard("rows"),
                                            (
                                                pattern.Wildcard("extract_inner"),
                                                pattern.Wildcard("extract_ratio"),
                                            ),
                                        ),
                                        (
                                            pattern.Wildcard("source_row"),
                                            (64, pattern.Wildcard("extract_step")),
                                        ),
                                    )
                                    for ratio in (2, 4)
                                },
                            },
                        ),
                    )
                    for m in (ExtractMode.INTEGER, ExtractMode.FLOAT)
                },
            },
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=R | W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("rows"), pattern.Wildcard("dst_lanes")),
            dtype=pattern.Wildcard("dst_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("rows"), pattern.Wildcard("dst_lanes")),
                (pattern.Wildcard("dst_lanes"), 1),
            ),
        ),
    )


ExtrxV1, ExtrxV2, ExtrxV3 = instruction_versions(Extrx, (1, 2, 4), name="extrx", category="memory")

ExtryV1, ExtryV2, ExtryV3 = instruction_versions(Extry, (1, 2, 4), name="extry", category="memory")

register_semantics((ExtrxV1, ExtrxV2, ExtrxV3, ExtryV1, ExtryV2, ExtryV3,))


__all__ = ["ExtrxV1", "ExtrxV2", "ExtrxV3", "ExtryV1", "ExtryV2", "ExtryV3"]
