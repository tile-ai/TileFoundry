"""AMX MAC16 Op."""

from tilefoundry.interfaces import schedule
from tilefoundry.ir import expr, pattern
from tilefoundry.ir.hir import math, nn

from .common import (
    AmxOp,
    ProductMode,
)
from .semantics import register_semantics

R = expr.MemoryEffect.READ
W = expr.MemoryEffect.WRITE


@expr.register_op(dialect="T", namespace="amx", category="nn", name="mac16")
class Mac16(AmxOp):
    hir_matches = (
        schedule.HirMatch(
            nn.MatMul,
            when={"a_layout": "MK", "b_layout": "KN"},
            attributes={"mode": ProductMode.MATRIX},
            operands={"acc": "result", "lhs": "lhs", "rhs": "rhs"},
        ),
        schedule.HirMatch(
            math.Binary,
            when={"kind": expr.BinaryKind.MUL},
            attributes={"mode": ProductMode.VECTOR, "skip_z": True},
            operands={"acc": "result", "lhs": "lhs", "rhs": "rhs"},
        ),
        schedule.HirMatch(
            math.Binary,
            when={"kind": expr.BinaryKind.ADD},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": False},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        ),
    )
    """One 16-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "MAC16"
    capability = expr.OpCapability("amx_mac16")
    resource = "amx_engine"
    roles = ("acc", "lhs", "rhs")
    writes = ("acc",)
    register_roles = {"acc": "z", "lhs": "y", "rhs": "x"}
    mode = expr.ParamDef(
        kind="attribute",
        annotation=ProductMode,
        pattern=pattern.Wildcard("mode"),
        default=ProductMode.MATRIX,
    )
    acc_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard(
            "acc_dtype",
            predicates=(
                pattern.Switch(
                    "mode",
                    {
                        ProductMode.MATRIX: pattern.Or(expr.DType.i16, expr.DType.u16, expr.DType.i32, expr.DType.u32),
                        ProductMode.VECTOR: pattern.Or(expr.DType.i16, expr.DType.u16),
                    },
                ),
            ),
        ),
        default=expr.DType.i32,
    )
    x_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("x_dtype", predicates=(pattern.Or(expr.DType.i8, expr.DType.i16),)),
        default=expr.DType.i16,
    )
    y_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("y_dtype", predicates=(pattern.Or(expr.DType.i8, expr.DType.i16),)),
        default=expr.DType.i16,
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
        default=0,
    )
    z_row = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("z_row", predicates=(pattern.Range(lo=0, hi=63),)),
        default=0,
    )
    skip_x = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard("skip_x", predicates=(pattern.Or(False, True),)),
        default=False,
    )
    skip_y = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard("skip_y", predicates=(pattern.Or(False, True),)),
        default=False,
    )
    skip_z = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard("skip_z", predicates=(pattern.Or(False, True),)),
        default=False,
    )
    x_mask = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("x_mask", predicates=(pattern.Range(lo=0, hi=127),)),
        default=0,
    )
    y_mask = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "y_mask",
            predicates=(
                pattern.Switch(
                    "mode", {ProductMode.MATRIX: pattern.Range(lo=0, hi=127), ProductMode.VECTOR: 0}
                ),
            ),
        ),
        default=0,
    )
    acc = expr.ParamDef(
        kind="input",
        effect=R | W,
        pattern=pattern.Tensor(
            shape=(pattern.Switch("mode", {ProductMode.MATRIX: 32, ProductMode.VECTOR: 1}), 32),
            dtype=pattern.Wildcard("acc_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "mode",
                {
                    ProductMode.MATRIX: pattern.Switch(
                        "acc_dtype",
                        {
                            expr.DType.i16: pattern.Layout((32, 32), (64, 1)),
                            expr.DType.u16: pattern.Layout((32, 32), (64, 1)),
                            expr.DType.i32: pattern.Layout((32, (2, 16)), (32, (16, 1))),
                            expr.DType.u32: pattern.Layout((32, (2, 16)), (32, (16, 1))),
                        },
                    ),
                    ProductMode.VECTOR: pattern.Layout((1, 32), (0, 1)),
                },
            ),
        ),
    )
    lhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(
                pattern.Switch("mode", {ProductMode.MATRIX: 32, ProductMode.VECTOR: 1}),
                pattern.Switch("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 32}),
            ),
            dtype=pattern.Wildcard("y_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "mode",
                {
                    ProductMode.MATRIX: pattern.Layout(
                        (32, 1), (pattern.Switch("y_dtype", {expr.DType.i8: 2, expr.DType.i16: 1}), 0)
                    ),
                    ProductMode.VECTOR: pattern.Layout(
                        (1, 32), (0, pattern.Switch("y_dtype", {expr.DType.i8: 2, expr.DType.i16: 1}))
                    ),
                },
            ),
        ),
    )
    rhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(1, 32),
            dtype=pattern.Wildcard("x_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (1, 32), (0, pattern.Switch("x_dtype", {expr.DType.i8: 2, expr.DType.i16: 1}))
            ),
        ),
    )
    shift = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("shift", predicates=(pattern.Range(lo=0, hi=31),)),
        default=0,
    )

register_semantics((Mac16,))


__all__ = ["Mac16"]
