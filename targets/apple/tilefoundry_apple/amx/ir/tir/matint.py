"""AMX MATINT Ops."""

from tilefoundry.interfaces import schedule
from tilefoundry.ir import expr, pattern
from tilefoundry.ir.hir import nn

from .common import (
    I8,
    I16,
    I32,
    AmxOp,
    IndexMode,
    IntALU,
    IntLane,
    Repeat,
    Shuffle,
    instruction_versions,
)
from .semantics import register_semantics

R = expr.MemoryEffect.READ
W = expr.MemoryEffect.WRITE


class Matint(AmxOp):
    hir_matches = tuple(
        schedule.HirMatch(
            nn.MatMul,
            when={"a_layout": "MK", "b_layout": "KN"},
            attributes={"lane_mode": lane, "alu": IntALU.ADD_PRODUCT},
            operands={"acc": "result", "lhs": "lhs", "rhs": "rhs"},
        )
        for lane in IntLane
    )
    mnemonic = "MATINT"
    capability = expr.OpCapability("amx_matint")
    resource = "amx_engine"
    roles = ("acc", "lhs", "rhs", "table")
    writes = ("acc",)
    register_roles = {"acc": "z", "lhs": "y", "rhs": "x", "table": "indexed_bank"}
    alu = expr.ParamDef(
        kind="attribute",
        annotation=IntALU,
        pattern=pattern.Wildcard(
            "alu",
            predicates=(
                pattern.In(
                    pattern.variable("alu"),
                    (
                        IntALU.ADD_PRODUCT,
                        IntALU.SUB_PRODUCT,
                        IntALU.ADD_SUM,
                        IntALU.SUB_SUM,
                        IntALU.SHIFT,
                        IntALU.SQRDMLAH,
                        IntALU.SQRDMLSH,
                        IntALU.WIDE_PRODUCT,
                        IntALU.POPCOUNT,
                    ),
                ),
                (pattern.variable("amx_generation") >= 2) | (pattern.variable("alu") < 10),
            ),
        ),
        default=IntALU.ADD_PRODUCT,
    )
    lane_mode = expr.ParamDef(
        kind="attribute",
        annotation=IntLane,
        pattern=pattern.Wildcard(
            "lane_mode",
            predicates=(
                (
                    (
                        (pattern.variable("alu") == IntALU.SHIFT)
                        & (
                            pattern.In(
                                pattern.variable("lane_mode"),
                                (
                                    IntLane.NARROW16,
                                    IntLane.WIDE16,
                                    IntLane.I32,
                                    IntLane.I8_I8_I32,
                                    IntLane.I8_I8_I16,
                                ),
                            )
                        )
                    )
                    | (
                        (pattern.In(pattern.variable("alu"), (IntALU.SQRDMLAH, IntALU.SQRDMLSH)))
                        & (pattern.variable("lane_mode") == IntLane.NARROW16)
                    )
                    | (
                        (pattern.variable("alu") == IntALU.WIDE_PRODUCT)
                        & (
                            pattern.In(
                                pattern.variable("lane_mode"),
                                (
                                    IntLane.NARROW16,
                                    IntLane.I8_I8_I32,
                                    IntLane.I8_I8_I16,
                                    IntLane.I8_I16_I32,
                                ),
                            )
                        )
                    )
                    | (
                        (pattern.variable("alu") == IntALU.POPCOUNT)
                        & (
                            pattern.In(
                                pattern.variable("lane_mode"),
                                (IntLane.NARROW16, IntLane.WIDE16, IntLane.I32),
                            )
                        )
                    )
                    | (
                        (
                            ~pattern.In(
                                pattern.variable("alu"),
                                (
                                    IntALU.SHIFT,
                                    IntALU.SQRDMLAH,
                                    IntALU.SQRDMLSH,
                                    IntALU.WIDE_PRODUCT,
                                    IntALU.POPCOUNT,
                                ),
                            )
                        )
                        & (pattern.In(pattern.variable("lane_mode"), (IntLane.NARROW16, IntLane.WIDE16)))
                    )
                ),
            ),
        ),
        default=IntLane.NARROW16,
    )
    x_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard(
            "x_dtype",
            predicates=(
                (
                    ((pattern.variable("alu") == IntALU.SHIFT) & (pattern.variable("x_dtype") == expr.DType.u8))
                    | (
                        (pattern.variable("alu") != IntALU.SHIFT)
                        & (
                            (pattern.In(pattern.variable("x_dtype"), (*I8, *I16, *I32)))
                            & (
                                ((pattern.variable("alu") == IntALU.WIDE_PRODUCT) & (pattern.Bits("x_dtype") == 8))
                                | (
                                    (pattern.variable("alu") != IntALU.WIDE_PRODUCT)
                                    & (
                                        pattern.Bits("x_dtype")
                                        == pattern.Table((0, 0, 16, 16, 32, 0, 0, 0, 0, 8, 8, 8, 8, 16))[
                                            pattern.variable("lane_mode")
                                        ]
                                    )
                                )
                            )
                        )
                    )
                ),
            ),
        ),
        default=expr.DType.i16,
    )
    y_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard(
            "y_dtype",
            predicates=(
                ((pattern.variable("alu") == IntALU.SHIFT) & (pattern.variable("y_dtype") == expr.DType.u8))
                | (
                    (pattern.variable("alu") != IntALU.SHIFT)
                    & (
                        (pattern.In(pattern.variable("y_dtype"), (*I8, *I16, *I32)))
                        & (
                            (
                                (pattern.variable("lane_mode") == IntLane.I8_I16_I32)
                                & (
                                    ((pattern.variable("amx_generation") < 3) & (pattern.Bits("y_dtype") == 8))
                                    | ((pattern.variable("amx_generation") >= 3) & (pattern.Bits("y_dtype") == 16))
                                )
                            )
                            | (
                                (pattern.variable("lane_mode") != IntLane.I8_I16_I32)
                                & (
                                    (
                                        (pattern.variable("alu") == IntALU.WIDE_PRODUCT)
                                        & (pattern.Bits("y_dtype") == 8)
                                    )
                                    | (
                                        (pattern.variable("alu") != IntALU.WIDE_PRODUCT)
                                        & (
                                            pattern.Bits("y_dtype")
                                            == pattern.Table(
                                                (0, 0, 16, 16, 32, 0, 0, 0, 0, 8, 8, 8, 16, 8)
                                            )[pattern.variable("lane_mode")]
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                pattern.variable("x_lanes") == 512 // pattern.Bits("x_dtype"),
                pattern.variable("y_lanes") == 512 // pattern.Bits("y_dtype"),
            ),
        ),
        default=expr.DType.i16,
    )
    acc_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard(
            "acc_dtype",
            predicates=(
                (
                    (pattern.In(pattern.variable("acc_dtype"), (*I8, *I16, *I32)))
                    & (
                        (
                            (pattern.variable("lane_mode") == IntLane.I8_I16_I32)
                            & (
                                ((pattern.variable("amx_generation") < 3) & (pattern.Bits("acc_dtype") == 16))
                                | ((pattern.variable("amx_generation") >= 3) & (pattern.Bits("acc_dtype") == 32))
                            )
                        )
                        | (
                            (pattern.variable("lane_mode") != IntLane.I8_I16_I32)
                            & (
                                pattern.Bits("acc_dtype")
                                == pattern.Table((0, 0, 16, 32, 32, 0, 0, 0, 0, 8, 32, 16, 32, 32))[
                                    pattern.variable("lane_mode")
                                ]
                            )
                        )
                    )
                ),
            ),
        ),
        default=expr.DType.i16,
    )
    index_mode = expr.ParamDef(
        kind="attribute",
        annotation=IndexMode,
        pattern=pattern.Wildcard(
            "index_mode",
            predicates=(
                pattern.In(pattern.variable("alu"), (IntALU.ADD_PRODUCT, IntALU.WIDE_PRODUCT))
                | (pattern.variable("index_mode") == IndexMode.NONE),
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
        pattern=pattern.Wildcard("z_row", predicates=(pattern.Range(lo=0, hi=3),)),
        default=0,
    )
    shift = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("shift", predicates=(pattern.Range(lo=0, hi=31),)),
        default=0,
    )
    mask = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("mask", predicates=(pattern.Range(lo=0, hi=511),)),
        default=0,
    )
    mask_y = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard("mask_y", predicates=(pattern.Or(False, True),)),
        default=False,
    )
    saturate = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "saturate",
            predicates=(
                pattern.In(pattern.variable("saturate"), (False, True)),
                (pattern.variable("alu") == IntALU.SHIFT) | (pattern.In(pattern.variable("saturate"), (False,))),
            ),
        ),
        default=False,
    )
    rounding = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "rounding",
            predicates=(
                pattern.In(pattern.variable("rounding"), (False, True)),
                (pattern.variable("alu") == IntALU.SHIFT) | (pattern.In(pattern.variable("rounding"), (False,))),
            ),
        ),
        default=False,
    )
    signed_output = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "signed_output",
            predicates=(
                pattern.In(pattern.variable("signed_output"), (False, True)),
                (pattern.variable("alu") == IntALU.SHIFT) | (pattern.In(pattern.variable("signed_output"), (False,))),
            ),
        ),
        default=False,
    )
    shuffle_x = expr.ParamDef(
        kind="attribute",
        annotation=Shuffle,
        pattern=pattern.Wildcard(
            "shuffle_x",
            predicates=(
                (pattern.variable("alu") != IntALU.SHIFT) | (pattern.variable("shuffle_x") == Shuffle.NONE),
            ),
        ),
        default=Shuffle.NONE,
    )
    shuffle_y = expr.ParamDef(
        kind="attribute",
        annotation=Shuffle,
        pattern=pattern.Wildcard(
            "shuffle_y",
            predicates=(
                (pattern.variable("alu") != IntALU.SHIFT) | (pattern.variable("shuffle_y") == Shuffle.NONE),
            ),
        ),
        default=Shuffle.NONE,
    )
    repeat = expr.ParamDef(
        kind="attribute",
        annotation=Repeat,
        pattern=pattern.Wildcard("repeat", predicates=(Repeat.ONE,)),
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
                (pattern.variable("rows") == 1),
                (pattern.variable("z_step") == 512 // pattern.Bits("acc_dtype")),
                (
                    (
                        (pattern.variable("alu") == IntALU.SHIFT)
                        & (
                            (pattern.variable("acc_rows") == 512 // pattern.Bits("acc_dtype"))
                            & (pattern.variable("acc_lanes") == 512 // pattern.Bits("acc_dtype"))
                            & (pattern.variable("acc_step") == 64)
                            & (pattern.variable("acc_ratio") == 1)
                            & (pattern.variable("acc_inner") == 512 // pattern.Bits("acc_dtype"))
                            & (pattern.variable("acc_wide") == 0)
                        )
                    )
                    | (
                        (pattern.variable("alu") != IntALU.SHIFT)
                        & (
                            (
                                (pattern.Bits("x_dtype") == 8)
                                & (pattern.variable("acc_rows") == 512 // pattern.Bits("acc_dtype"))
                                & (pattern.variable("acc_lanes") == 512 // pattern.Bits("x_dtype"))
                                & (pattern.variable("acc_step") == 64)
                                & (pattern.variable("acc_ratio") == pattern.Bits("acc_dtype") // pattern.Bits("x_dtype"))
                                & (pattern.variable("acc_inner") == 512 // pattern.Bits("acc_dtype"))
                                & (
                                    pattern.variable("acc_wide")
                                    == pattern.Table((0, 0, 1, 1, 1))[pattern.Bits("acc_dtype") // pattern.Bits("x_dtype")]
                                )
                            )
                            | (
                                (pattern.Bits("x_dtype") != 8)
                                & (
                                    (pattern.variable("acc_rows") == 512 // pattern.Bits("x_dtype"))
                                    & (pattern.variable("acc_lanes") == 512 // pattern.Bits("x_dtype"))
                                    & (
                                        pattern.variable("acc_ratio")
                                        == pattern.Bits("acc_dtype") // pattern.Bits("x_dtype")
                                    )
                                    & (pattern.variable("acc_inner") == 512 // pattern.Bits("acc_dtype"))
                                    & (
                                        pattern.variable("acc_wide")
                                        == pattern.Table((0, 0, 1, 1, 1))[
                                            pattern.Bits("acc_dtype") // pattern.Bits("x_dtype")
                                        ]
                                    )
                                )
                                & (
                                    pattern.variable("acc_step")
                                    == (pattern.Bits("x_dtype") // 8) * 512 // pattern.Bits("acc_dtype")
                                )
                            )
                        )
                    )
                ),
                (
                    pattern.variable("has_inputs")
                    == pattern.Table((1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1))[pattern.variable("alu")]
                ),
                (pattern.variable("table_side") == pattern.Table((0, 1, 2, 1, 2))[pattern.variable("index_mode")]),
                (
                    pattern.variable("lhs_count")
                    == (
                        (512 // pattern.Bits("y_dtype")) * pattern.Table((8, 8, 2, 8, 4))[pattern.variable("index_mode")]
                        + 7
                    )
                    // 8
                ),
                (pattern.variable("lhs_packed") == pattern.Table((0, 0, 1, 0, 1))[pattern.variable("index_mode")]),
                (
                    pattern.variable("rhs_count")
                    == (
                        (512 // pattern.Bits("x_dtype")) * pattern.Table((8, 2, 8, 4, 8))[pattern.variable("index_mode")]
                        + 7
                    )
                    // 8
                ),
                (pattern.variable("rhs_packed") == pattern.Table((0, 1, 0, 1, 0))[pattern.variable("index_mode")]),
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
        pattern=pattern.Switch(
            "has_inputs",
            {
                0: None,
                1: pattern.Switch(
                    "lhs_packed",
                    {
                        0: pattern.Tensor(
                            shape=(pattern.Wildcard("y_lanes"), 1),
                            dtype=pattern.Wildcard("y_dtype"),
                            storage=expr.StorageKind.RMEM,
                            layout=pattern.Layout((pattern.Wildcard("y_lanes"), 1), (1, 0)),
                        ),
                        1: pattern.Tensor(
                            shape=(pattern.Wildcard("lhs_count"),),
                            dtype=expr.DType.u8,
                            storage=expr.StorageKind.RMEM,
                            layout=pattern.Layout((pattern.Wildcard("lhs_count"),), (1,)),
                        ),
                    },
                ),
            },
        ),
        optional=True,
        default=None,
    )
    rhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Switch(
            "has_inputs",
            {
                0: None,
                1: pattern.Switch(
                    "rhs_packed",
                    {
                        0: pattern.Tensor(
                            shape=(1, pattern.Wildcard("x_lanes")),
                            dtype=pattern.Wildcard("x_dtype"),
                            storage=expr.StorageKind.RMEM,
                            layout=pattern.Layout((1, pattern.Wildcard("x_lanes")), (0, 1)),
                        ),
                        1: pattern.Tensor(
                            shape=(pattern.Wildcard("rhs_count"),),
                            dtype=expr.DType.u8,
                            storage=expr.StorageKind.RMEM,
                            layout=pattern.Layout((pattern.Wildcard("rhs_count"),), (1,)),
                        ),
                    },
                ),
            },
        ),
        optional=True,
        default=None,
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
                    dtype=pattern.Wildcard("x_dtype"),
                    storage=expr.StorageKind.RMEM,
                    layout=pattern.Layout((pattern.Wildcard("x_lanes"),), (1,)),
                ),
                2: pattern.Tensor(
                    shape=(pattern.Wildcard("y_lanes"),),
                    dtype=pattern.Wildcard("y_dtype"),
                    storage=expr.StorageKind.RMEM,
                    layout=pattern.Layout((pattern.Wildcard("y_lanes"),), (1,)),
                ),
            },
        ),
        optional=True,
        default=None,
    )


MatintV1, MatintV2 = instruction_versions(Matint, (1, 3), name="matint", category="nn")

register_semantics((MatintV1, MatintV2,))


__all__ = ["MatintV1", "MatintV2"]
