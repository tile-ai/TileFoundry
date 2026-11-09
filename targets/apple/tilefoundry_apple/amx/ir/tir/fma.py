"""AMX FMA and FMS Ops."""

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


@expr.register_op(dialect="T", namespace="amx", category="nn", name="fma16")
class Fma16(AmxOp):
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
    mnemonic = "FMA16"
    capability = expr.OpCapability("amx_fma16")
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
                        ProductMode.MATRIX: pattern.Or(expr.DType.f16, expr.DType.f32),
                        ProductMode.VECTOR: expr.DType.f16,
                    },
                ),
            ),
        ),
        default=expr.DType.f32,
    )
    x_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("x_dtype", predicates=(expr.DType.f16,)),
        default=expr.DType.f16,
    )
    y_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("y_dtype", predicates=(expr.DType.f16,)),
        default=expr.DType.f16,
    )
    out_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        optional=True,
        default=None,
        pattern=pattern.Wildcard("out_dtype", predicates=(pattern.Wildcard("acc_dtype"),)),
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
                            expr.DType.f16: pattern.Layout((32, 32), (64, 1)),
                            expr.DType.f32: pattern.Layout((32, (2, 16)), (32, (16, 1))),
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
                    ProductMode.MATRIX: pattern.Layout((32, 1), (1, 0)),
                    ProductMode.VECTOR: pattern.Layout((1, 32), (0, 1)),
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
            layout=pattern.Layout((1, 32), (0, 1)),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="nn", name="fma32")
class Fma32(AmxOp):
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
    """One 32-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMA32"
    capability = expr.OpCapability("amx_fma32")
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
        pattern=pattern.Wildcard("acc_dtype", predicates=(expr.DType.f32,)),
        default=expr.DType.f32,
    )
    x_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("x_dtype", predicates=(pattern.Or(expr.DType.f16, expr.DType.f32),)),
        default=expr.DType.f32,
    )
    y_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("y_dtype", predicates=(pattern.Or(expr.DType.f16, expr.DType.f32),)),
        default=expr.DType.f32,
    )
    out_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        optional=True,
        default=None,
        pattern=pattern.Wildcard("out_dtype", predicates=(pattern.Wildcard("acc_dtype"),)),
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
            shape=(pattern.Switch("mode", {ProductMode.MATRIX: 16, ProductMode.VECTOR: 1}), 16),
            dtype=pattern.Wildcard("acc_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "mode",
                {
                    ProductMode.MATRIX: pattern.Layout((16, 16), (64, 1)),
                    ProductMode.VECTOR: pattern.Layout((1, 16), (0, 1)),
                },
            ),
        ),
    )
    lhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(
                pattern.Switch("mode", {ProductMode.MATRIX: 16, ProductMode.VECTOR: 1}),
                pattern.Switch("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 16}),
            ),
            dtype=pattern.Wildcard("y_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "mode",
                {
                    ProductMode.MATRIX: pattern.Layout(
                        (16, 1), (pattern.Switch("y_dtype", {expr.DType.f16: 2, expr.DType.f32: 1}), 0)
                    ),
                    ProductMode.VECTOR: pattern.Layout(
                        (1, 16), (0, pattern.Switch("y_dtype", {expr.DType.f16: 2, expr.DType.f32: 1}))
                    ),
                },
            ),
        ),
    )
    rhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(1, 16),
            dtype=pattern.Wildcard("x_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (1, 16), (0, pattern.Switch("x_dtype", {expr.DType.f16: 2, expr.DType.f32: 1}))
            ),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="nn", name="fma64")
class Fma64(AmxOp):
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
    """One 64-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMA64"
    capability = expr.OpCapability("amx_fma64")
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
        pattern=pattern.Wildcard("acc_dtype", predicates=(expr.DType.f64,)),
        default=expr.DType.f64,
    )
    x_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("x_dtype", predicates=(expr.DType.f64,)),
        default=expr.DType.f64,
    )
    y_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("y_dtype", predicates=(expr.DType.f64,)),
        default=expr.DType.f64,
    )
    out_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        optional=True,
        default=None,
        pattern=pattern.Wildcard("out_dtype", predicates=(pattern.Wildcard("acc_dtype"),)),
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
            shape=(pattern.Switch("mode", {ProductMode.MATRIX: 8, ProductMode.VECTOR: 1}), 8),
            dtype=pattern.Wildcard("acc_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "mode",
                {
                    ProductMode.MATRIX: pattern.Layout((8, 8), (64, 1)),
                    ProductMode.VECTOR: pattern.Layout((1, 8), (0, 1)),
                },
            ),
        ),
    )
    lhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(
                pattern.Switch("mode", {ProductMode.MATRIX: 8, ProductMode.VECTOR: 1}),
                pattern.Switch("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 8}),
            ),
            dtype=pattern.Wildcard("y_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "mode",
                {
                    ProductMode.MATRIX: pattern.Layout((8, 1), (1, 0)),
                    ProductMode.VECTOR: pattern.Layout((1, 8), (0, 1)),
                },
            ),
        ),
    )
    rhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(1, 8),
            dtype=pattern.Wildcard("x_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout((1, 8), (0, 1)),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="nn", name="fms16")
class Fms16(AmxOp):
    hir_matches = (
        schedule.HirMatch(
            math.Unary,
            when={"kind": expr.UnaryKind.NEG},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": True},
            operands={"acc": "result", "lhs": "x", "rhs": "x"},
        ),
        schedule.HirMatch(
            math.Binary,
            when={"kind": expr.BinaryKind.SUB},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": False},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        ),
    )
    """One 16-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMS16"
    capability = expr.OpCapability("amx_fms16")
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
                        ProductMode.MATRIX: pattern.Or(expr.DType.f16, expr.DType.f32),
                        ProductMode.VECTOR: expr.DType.f16,
                    },
                ),
            ),
        ),
        default=expr.DType.f32,
    )
    x_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("x_dtype", predicates=(expr.DType.f16,)),
        default=expr.DType.f16,
    )
    y_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("y_dtype", predicates=(expr.DType.f16,)),
        default=expr.DType.f16,
    )
    out_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        optional=True,
        default=None,
        pattern=pattern.Wildcard("out_dtype", predicates=(pattern.Wildcard("acc_dtype"),)),
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
                            expr.DType.f16: pattern.Layout((32, 32), (64, 1)),
                            expr.DType.f32: pattern.Layout((32, (2, 16)), (32, (16, 1))),
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
                    ProductMode.MATRIX: pattern.Layout((32, 1), (1, 0)),
                    ProductMode.VECTOR: pattern.Layout((1, 32), (0, 1)),
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
            layout=pattern.Layout((1, 32), (0, 1)),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="nn", name="fms32")
class Fms32(AmxOp):
    hir_matches = (
        schedule.HirMatch(
            math.Unary,
            when={"kind": expr.UnaryKind.NEG},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": True},
            operands={"acc": "result", "lhs": "x", "rhs": "x"},
        ),
        schedule.HirMatch(
            math.Binary,
            when={"kind": expr.BinaryKind.SUB},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": False},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        ),
    )
    """One 32-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMS32"
    capability = expr.OpCapability("amx_fms32")
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
        pattern=pattern.Wildcard("acc_dtype", predicates=(expr.DType.f32,)),
        default=expr.DType.f32,
    )
    x_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("x_dtype", predicates=(pattern.Or(expr.DType.f16, expr.DType.f32),)),
        default=expr.DType.f32,
    )
    y_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("y_dtype", predicates=(pattern.Or(expr.DType.f16, expr.DType.f32),)),
        default=expr.DType.f32,
    )
    out_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        optional=True,
        default=None,
        pattern=pattern.Wildcard("out_dtype", predicates=(pattern.Wildcard("acc_dtype"),)),
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
            shape=(pattern.Switch("mode", {ProductMode.MATRIX: 16, ProductMode.VECTOR: 1}), 16),
            dtype=pattern.Wildcard("acc_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "mode",
                {
                    ProductMode.MATRIX: pattern.Layout((16, 16), (64, 1)),
                    ProductMode.VECTOR: pattern.Layout((1, 16), (0, 1)),
                },
            ),
        ),
    )
    lhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(
                pattern.Switch("mode", {ProductMode.MATRIX: 16, ProductMode.VECTOR: 1}),
                pattern.Switch("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 16}),
            ),
            dtype=pattern.Wildcard("y_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "mode",
                {
                    ProductMode.MATRIX: pattern.Layout(
                        (16, 1), (pattern.Switch("y_dtype", {expr.DType.f16: 2, expr.DType.f32: 1}), 0)
                    ),
                    ProductMode.VECTOR: pattern.Layout(
                        (1, 16), (0, pattern.Switch("y_dtype", {expr.DType.f16: 2, expr.DType.f32: 1}))
                    ),
                },
            ),
        ),
    )
    rhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(1, 16),
            dtype=pattern.Wildcard("x_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (1, 16), (0, pattern.Switch("x_dtype", {expr.DType.f16: 2, expr.DType.f32: 1}))
            ),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="nn", name="fms64")
class Fms64(AmxOp):
    hir_matches = (
        schedule.HirMatch(
            math.Unary,
            when={"kind": expr.UnaryKind.NEG},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": True},
            operands={"acc": "result", "lhs": "x", "rhs": "x"},
        ),
        schedule.HirMatch(
            math.Binary,
            when={"kind": expr.BinaryKind.SUB},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": False},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        ),
    )
    """One 64-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMS64"
    capability = expr.OpCapability("amx_fms64")
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
        pattern=pattern.Wildcard("acc_dtype", predicates=(expr.DType.f64,)),
        default=expr.DType.f64,
    )
    x_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("x_dtype", predicates=(expr.DType.f64,)),
        default=expr.DType.f64,
    )
    y_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("y_dtype", predicates=(expr.DType.f64,)),
        default=expr.DType.f64,
    )
    out_dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        optional=True,
        default=None,
        pattern=pattern.Wildcard("out_dtype", predicates=(pattern.Wildcard("acc_dtype"),)),
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
            shape=(pattern.Switch("mode", {ProductMode.MATRIX: 8, ProductMode.VECTOR: 1}), 8),
            dtype=pattern.Wildcard("acc_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "mode",
                {
                    ProductMode.MATRIX: pattern.Layout((8, 8), (64, 1)),
                    ProductMode.VECTOR: pattern.Layout((1, 8), (0, 1)),
                },
            ),
        ),
    )
    lhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(
                pattern.Switch("mode", {ProductMode.MATRIX: 8, ProductMode.VECTOR: 1}),
                pattern.Switch("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 8}),
            ),
            dtype=pattern.Wildcard("y_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Switch(
                "mode",
                {
                    ProductMode.MATRIX: pattern.Layout((8, 1), (1, 0)),
                    ProductMode.VECTOR: pattern.Layout((1, 8), (0, 1)),
                },
            ),
        ),
    )
    rhs = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(1, 8),
            dtype=pattern.Wildcard("x_dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout((1, 8), (0, 1)),
        ),
    )

register_semantics((Fma16, Fma32, Fma64, Fms16, Fms32, Fms64,))


__all__ = ["Fma16", "Fma32", "Fma64", "Fms16", "Fms32", "Fms64"]
