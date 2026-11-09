"""AMX VECFP Ops."""

from tilefoundry.interfaces import schedule
from tilefoundry.ir import expr, pattern
from tilefoundry.ir.hir import math

from .common import (
    FPALU,
    AmxOp,
    FPMode,
    IndexMode,
    Repeat,
    Shuffle,
    instruction_versions,
)
from .semantics import register_semantics

R = expr.MemoryEffect.READ
W = expr.MemoryEffect.WRITE


class Vecfp(AmxOp):
    hir_matches = tuple(
        schedule.HirMatch(
            math.Binary,
            when={"kind": expr.BinaryKind.MUL},
            attributes={"fp_mode": mode, "alu": FPALU.PRODUCT},
            operands={"acc": "result", "lhs": "lhs", "rhs": "rhs"},
        )
        for mode in FPMode
    ) + tuple(
        schedule.HirMatch(
            math.Binary,
            when={"kind": kind},
            attributes={"fp_mode": mode, "alu": alu},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        )
        for mode in FPMode
        for kind, alu in (
            (expr.BinaryKind.ADD, FPALU.ADD_X),
            (expr.BinaryKind.MIN, FPALU.MIN),
            (expr.BinaryKind.MAX, FPALU.MAX),
        )
    )
    mnemonic = "VECFP"
    capability = expr.OpCapability("amx_vecfp")
    resource = "amx_engine"
    roles = ("acc", "lhs", "rhs", "table")
    writes = ("acc",)
    register_roles = {"acc": "z", "lhs": "y", "rhs": "x", "table": "indexed_bank"}
    fp_mode = expr.ParamDef(
        kind="attribute",
        annotation=FPMode,
        pattern=pattern.Wildcard(
            "fp_mode",
            predicates=(
                (pattern.variable("amx_generation") >= 2)
                | ~pattern.In(pattern.variable("fp_mode"), (FPMode.BF16, FPMode.BF16_F32)),
            ),
        ),
        default=FPMode.F32,
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard(
            "dtype",
            predicates=(
                (
                    pattern.variable("dtype")
                    == pattern.Table(
                        (
                            expr.DType.bf16,
                            expr.DType.bf16,
                            expr.DType.f16,
                            expr.DType.f16,
                            expr.DType.f32,
                            expr.DType.f64,
                            expr.DType.f64,
                            expr.DType.f64,
                        )
                    )[pattern.variable("fp_mode")]
                ),
                (pattern.variable("x_lanes") == 512 // pattern.Bits("dtype")),
                (pattern.variable("y_lanes") == 512 // pattern.Bits("dtype")),
            ),
        ),
        default=expr.DType.f32,
    )
    acc_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard(
            "acc_dtype",
            predicates=(
                (
                    pattern.variable("acc_dtype")
                    == pattern.Table(
                        (
                            expr.DType.bf16,
                            expr.DType.f32,
                            expr.DType.f16,
                            expr.DType.f32,
                            expr.DType.f32,
                            expr.DType.f64,
                            expr.DType.f64,
                            expr.DType.f64,
                        )
                    )[pattern.variable("fp_mode")]
                ),
            ),
        ),
        default=expr.DType.f32,
    )
    alu = expr.ParamDef(
        kind="attribute",
        annotation=FPALU,
        pattern=pattern.Wildcard(
            "alu",
            predicates=(
                pattern.In(pattern.variable("alu"), tuple(FPALU)),
                (pattern.variable("amx_generation") >= 2)
                | pattern.In(
                    pattern.variable("alu"),
                    (
                        FPALU.ADD_PRODUCT,
                        FPALU.SUB_PRODUCT,
                        FPALU.POSITIVE_SELECT,
                        FPALU.MIN,
                        FPALU.MAX,
                    ),
                ),
            ),
        ),
        default=FPALU.ADD_PRODUCT,
    )
    index_mode = expr.ParamDef(
        kind="attribute",
        annotation=IndexMode,
        pattern=pattern.Wildcard(
            "index_mode",
            predicates=(
                (pattern.variable("alu") == FPALU.ADD_PRODUCT) | (pattern.variable("index_mode") == IndexMode.NONE),
            ),
        ),
        default=IndexMode.NONE,
    )
    index_register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "index_register",
            predicates=(
                pattern.Range(lo=0, hi=7),
                (pattern.variable("index_mode") != IndexMode.NONE) | (pattern.variable("index_register") == 0),
            ),
        ),
        default=0,
    )
    x_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("x_offset", predicates=(pattern.Range(lo=0, hi=511),)),
        default=0,
    )
    y_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("y_offset", predicates=(pattern.Range(lo=0, hi=511),)),
        default=64,
    )
    z_row = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("z_row", predicates=(pattern.Range(lo=0, hi=63),)),
        default=0,
    )
    mask = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("mask", predicates=(pattern.Range(lo=0, hi=511),)),
        default=0,
    )
    y_mask = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("y_mask", predicates=(pattern.Range(lo=0, hi=511),)),
        default=0,
    )
    shuffle_x = expr.ParamDef(
        kind="attribute",
        annotation=Shuffle,
        pattern=pattern.Wildcard("shuffle_x"),
        default=Shuffle.NONE,
    )
    shuffle_y = expr.ParamDef(
        kind="attribute",
        annotation=Shuffle,
        pattern=pattern.Wildcard("shuffle_y"),
        default=Shuffle.NONE,
    )
    repeat = expr.ParamDef(
        kind="attribute",
        annotation=Repeat,
        pattern=pattern.Wildcard(
            "repeat",
            predicates=((pattern.variable("amx_generation") >= 2) | (pattern.variable("repeat") == Repeat.ONE),),
        ),
        default=Repeat.ONE,
    )
    broadcast_mode = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "broadcast_mode",
            predicates=(
                pattern.Range(lo=0, hi=7),
                (pattern.variable("repeat") != Repeat.ONE) | (pattern.variable("broadcast_mode") == 0),
                (pattern.variable("rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("repeat")]),
                (pattern.variable("z_step") == 512 // pattern.Bits("acc_dtype")),
                (
                    (
                        (pattern.variable("acc_rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("repeat")])
                        & (pattern.variable("acc_lanes") == 512 // pattern.Bits("dtype"))
                        & (pattern.variable("acc_ratio") == pattern.Bits("acc_dtype") // pattern.Bits("dtype"))
                        & (pattern.variable("acc_inner") == 512 // pattern.Bits("acc_dtype"))
                        & (
                            pattern.variable("acc_wide")
                            == pattern.Table((0, 0, 1, 1, 1))[pattern.Bits("acc_dtype") // pattern.Bits("dtype")]
                        )
                    )
                    & (
                        pattern.variable("acc_step")
                        == (64 // pattern.Table((0, 1, 2, 0, 4))[pattern.variable("repeat")])
                        * 512
                        // pattern.Bits("acc_dtype")
                    )
                ),
                (pattern.variable("table_side") == pattern.Table((0, 1, 1, 1, 1))[pattern.variable("index_mode")]),
                (
                    pattern.variable("lhs_count")
                    == ((512 // pattern.Bits("dtype")) * pattern.Table((8, 8, 2, 8, 4))[pattern.variable("index_mode")] + 7)
                    // 8
                ),
                (pattern.variable("lhs_packed") == pattern.Table((0, 0, 1, 0, 1))[pattern.variable("index_mode")]),
                (
                    pattern.variable("lhs_row")
                    == (
                        (
                            (512 // pattern.Bits("dtype")) * pattern.Table((8, 8, 2, 8, 4))[pattern.variable("index_mode")]
                            + 7
                        )
                        // 8
                    )
                    * pattern.Table((1, 1, 1, 0, 1, 1, 1, 0))[pattern.variable("broadcast_mode")]
                ),
                (
                    pattern.variable("rhs_count")
                    == ((512 // pattern.Bits("dtype")) * pattern.Table((8, 2, 8, 4, 8))[pattern.variable("index_mode")] + 7)
                    // 8
                ),
                (pattern.variable("rhs_packed") == pattern.Table((0, 1, 0, 1, 0))[pattern.variable("index_mode")]),
                (
                    pattern.variable("rhs_row")
                    == (
                        (
                            (512 // pattern.Bits("dtype")) * pattern.Table((8, 2, 8, 4, 8))[pattern.variable("index_mode")]
                            + 7
                        )
                        // 8
                    )
                    * pattern.Table((1, 1, 0, 1, 1, 1, 0, 1))[pattern.variable("broadcast_mode")]
                ),
            ),
        ),
        default=0,
    )
    acc = expr.ParamDef(
        kind="input",
        effect=R | W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("acc_rows"), pattern.Wildcard("acc_lanes")),
            dtype=pattern.Wildcard("acc_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "acc_wide",
                {
                    0: pattern.Layout(
                        (pattern.Wildcard("acc_rows"), pattern.Wildcard("acc_lanes")),
                        (pattern.Wildcard("acc_step"), 1),
                    ),
                    1: pattern.Layout(
                        (
                            pattern.Wildcard("acc_rows"),
                            (pattern.Wildcard("acc_ratio"), pattern.Wildcard("acc_inner")),
                        ),
                        (pattern.Wildcard("acc_step"), (pattern.Wildcard("z_step"), 1)),
                    ),
                },
            ),
        ),
    )
    lhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("rows"), pattern.Wildcard("lhs_count")),
            dtype=pattern.Switch("lhs_packed", {0: pattern.Wildcard("dtype"), 1: expr.DType.u8}),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("rows"), pattern.Wildcard("lhs_count")),
                (pattern.Wildcard("lhs_row"), 1),
            ),
        ),
    )
    rhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("rows"), pattern.Wildcard("rhs_count")),
            dtype=pattern.Switch("rhs_packed", {0: pattern.Wildcard("dtype"), 1: expr.DType.u8}),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("rows"), pattern.Wildcard("rhs_count")),
                (pattern.Wildcard("rhs_row"), 1),
            ),
        ),
    )
    table = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Switch(
            "table_side",
            {
                0: None,
                1: pattern.Tensor(
                    shape=(pattern.Wildcard("x_lanes"),),
                    dtype=pattern.Wildcard("dtype"),
                    storage=expr.StorageKind.RMEM,
                    layout=pattern.Layout((pattern.Wildcard("x_lanes"),), (1,)),
                ),
            },
        ),
        optional=True,
        default=None,
    )


VecfpV1, VecfpV2, VecfpV3 = instruction_versions(Vecfp, (1, 2, 4), name="vecfp", category="arith")

register_semantics((VecfpV1, VecfpV2, VecfpV3,))


__all__ = ["VecfpV1", "VecfpV2", "VecfpV3"]
