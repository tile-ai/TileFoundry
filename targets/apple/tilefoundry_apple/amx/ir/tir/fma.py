from tilefoundry.ir.core import OpCapability
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.pattern import (
    LayoutPattern,
    OrPattern,
    RangePattern,
    SwitchPattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.types import DType, StorageKind
from tilefoundry.visitor_registry.candidates import HirMatch

from .common import (
    AmxOp,
    ProductMode,
    bind,
)

R = MemoryEffect.READ
W = MemoryEffect.WRITE


@register_op(dialect="T", namespace="amx", category="nn", name="fma16")
class Fma16(AmxOp):
    hir_matches = (
        HirMatch(
            MatMul,
            when={"a_layout": "MK", "b_layout": "KN"},
            attributes={"mode": ProductMode.MATRIX},
            operands={"acc": "result", "lhs": "lhs", "rhs": "rhs"},
        ),
        HirMatch(
            Binary,
            when={"kind": BinaryKind.MUL},
            attributes={"mode": ProductMode.VECTOR, "skip_z": True},
            operands={"acc": "result", "lhs": "lhs", "rhs": "rhs"},
        ),
        HirMatch(
            Binary,
            when={"kind": BinaryKind.ADD},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": False},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        ),
    )
    """One 16-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMA16"
    capability = OpCapability("amx_fma16")
    resource = "amx_engine"
    roles = ("acc", "lhs", "rhs")
    writes = ("acc",)
    register_roles = {"acc": "z", "lhs": "y", "rhs": "x"}
    mode = ParamDef(
        kind="attribute",
        annotation=ProductMode,
        pattern=WildcardPattern("mode"),
        default=ProductMode.MATRIX,
    )
    acc_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern(
            "acc_dtype",
            predicates=(
                SwitchPattern(
                    "mode",
                    {
                        ProductMode.MATRIX: OrPattern(DType.f16, DType.f32),
                        ProductMode.VECTOR: DType.f16,
                    },
                ),
            ),
        ),
        default=DType.f32,
    )
    x_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("x_dtype", predicates=(DType.f16,)),
        default=DType.f16,
    )
    y_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("y_dtype", predicates=(DType.f16,)),
        default=DType.f16,
    )
    out_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        optional=True,
        default=None,
        pattern=WildcardPattern("out_dtype", predicates=(WildcardPattern("acc_dtype"),)),
    )
    x_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    y_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("y_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    z_row = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("z_row", predicates=(RangePattern(lo=0, hi=63),)),
        default=0,
    )
    skip_x = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_x", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_y = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_y", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_z = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_z", predicates=(OrPattern(False, True),)),
        default=False,
    )
    x_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_mask", predicates=(RangePattern(lo=0, hi=127),)),
        default=0,
    )
    y_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "y_mask",
            predicates=(
                SwitchPattern(
                    "mode", {ProductMode.MATRIX: RangePattern(lo=0, hi=127), ProductMode.VECTOR: 0}
                ),
            ),
        ),
        default=0,
    )
    acc = ParamDef(
        kind="input",
        effect=R | W,
        pattern=TensorPattern(
            shape=(SwitchPattern("mode", {ProductMode.MATRIX: 32, ProductMode.VECTOR: 1}), 32),
            dtype=WildcardPattern("acc_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: SwitchPattern(
                        "acc_dtype",
                        {
                            DType.f16: LayoutPattern((32, 32), (64, 1)),
                            DType.f32: LayoutPattern((32, (2, 16)), (32, (16, 1))),
                        },
                    ),
                    ProductMode.VECTOR: LayoutPattern((1, 32), (0, 1)),
                },
            ),
        ),
    )
    lhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(
                SwitchPattern("mode", {ProductMode.MATRIX: 32, ProductMode.VECTOR: 1}),
                SwitchPattern("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 32}),
            ),
            dtype=WildcardPattern("y_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: LayoutPattern((32, 1), (1, 0)),
                    ProductMode.VECTOR: LayoutPattern((1, 32), (0, 1)),
                },
            ),
        ),
    )
    rhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(1, 32),
            dtype=WildcardPattern("x_dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((1, 32), (0, 1)),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="nn", name="fma32")
class Fma32(AmxOp):
    hir_matches = (
        HirMatch(
            MatMul,
            when={"a_layout": "MK", "b_layout": "KN"},
            attributes={"mode": ProductMode.MATRIX},
            operands={"acc": "result", "lhs": "lhs", "rhs": "rhs"},
        ),
        HirMatch(
            Binary,
            when={"kind": BinaryKind.MUL},
            attributes={"mode": ProductMode.VECTOR, "skip_z": True},
            operands={"acc": "result", "lhs": "lhs", "rhs": "rhs"},
        ),
        HirMatch(
            Binary,
            when={"kind": BinaryKind.ADD},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": False},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        ),
    )
    """One 32-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMA32"
    capability = OpCapability("amx_fma32")
    resource = "amx_engine"
    roles = ("acc", "lhs", "rhs")
    writes = ("acc",)
    register_roles = {"acc": "z", "lhs": "y", "rhs": "x"}
    mode = ParamDef(
        kind="attribute",
        annotation=ProductMode,
        pattern=WildcardPattern("mode"),
        default=ProductMode.MATRIX,
    )
    acc_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("acc_dtype", predicates=(DType.f32,)),
        default=DType.f32,
    )
    x_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("x_dtype", predicates=(OrPattern(DType.f16, DType.f32),)),
        default=DType.f32,
    )
    y_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("y_dtype", predicates=(OrPattern(DType.f16, DType.f32),)),
        default=DType.f32,
    )
    out_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        optional=True,
        default=None,
        pattern=WildcardPattern("out_dtype", predicates=(WildcardPattern("acc_dtype"),)),
    )
    x_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    y_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("y_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    z_row = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("z_row", predicates=(RangePattern(lo=0, hi=63),)),
        default=0,
    )
    skip_x = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_x", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_y = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_y", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_z = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_z", predicates=(OrPattern(False, True),)),
        default=False,
    )
    x_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_mask", predicates=(RangePattern(lo=0, hi=127),)),
        default=0,
    )
    y_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "y_mask",
            predicates=(
                SwitchPattern(
                    "mode", {ProductMode.MATRIX: RangePattern(lo=0, hi=127), ProductMode.VECTOR: 0}
                ),
            ),
        ),
        default=0,
    )
    acc = ParamDef(
        kind="input",
        effect=R | W,
        pattern=TensorPattern(
            shape=(SwitchPattern("mode", {ProductMode.MATRIX: 16, ProductMode.VECTOR: 1}), 16),
            dtype=WildcardPattern("acc_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: LayoutPattern((16, 16), (64, 1)),
                    ProductMode.VECTOR: LayoutPattern((1, 16), (0, 1)),
                },
            ),
        ),
    )
    lhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(
                SwitchPattern("mode", {ProductMode.MATRIX: 16, ProductMode.VECTOR: 1}),
                SwitchPattern("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 16}),
            ),
            dtype=WildcardPattern("y_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: LayoutPattern(
                        (16, 1), (SwitchPattern("y_dtype", {DType.f16: 2, DType.f32: 1}), 0)
                    ),
                    ProductMode.VECTOR: LayoutPattern(
                        (1, 16), (0, SwitchPattern("y_dtype", {DType.f16: 2, DType.f32: 1}))
                    ),
                },
            ),
        ),
    )
    rhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(1, 16),
            dtype=WildcardPattern("x_dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern(
                (1, 16), (0, SwitchPattern("x_dtype", {DType.f16: 2, DType.f32: 1}))
            ),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="nn", name="fma64")
class Fma64(AmxOp):
    hir_matches = (
        HirMatch(
            MatMul,
            when={"a_layout": "MK", "b_layout": "KN"},
            attributes={"mode": ProductMode.MATRIX},
            operands={"acc": "result", "lhs": "lhs", "rhs": "rhs"},
        ),
        HirMatch(
            Binary,
            when={"kind": BinaryKind.MUL},
            attributes={"mode": ProductMode.VECTOR, "skip_z": True},
            operands={"acc": "result", "lhs": "lhs", "rhs": "rhs"},
        ),
        HirMatch(
            Binary,
            when={"kind": BinaryKind.ADD},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": False},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        ),
    )
    """One 64-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMA64"
    capability = OpCapability("amx_fma64")
    resource = "amx_engine"
    roles = ("acc", "lhs", "rhs")
    writes = ("acc",)
    register_roles = {"acc": "z", "lhs": "y", "rhs": "x"}
    mode = ParamDef(
        kind="attribute",
        annotation=ProductMode,
        pattern=WildcardPattern("mode"),
        default=ProductMode.MATRIX,
    )
    acc_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("acc_dtype", predicates=(DType.f64,)),
        default=DType.f64,
    )
    x_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("x_dtype", predicates=(DType.f64,)),
        default=DType.f64,
    )
    y_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("y_dtype", predicates=(DType.f64,)),
        default=DType.f64,
    )
    out_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        optional=True,
        default=None,
        pattern=WildcardPattern("out_dtype", predicates=(WildcardPattern("acc_dtype"),)),
    )
    x_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    y_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("y_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    z_row = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("z_row", predicates=(RangePattern(lo=0, hi=63),)),
        default=0,
    )
    skip_x = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_x", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_y = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_y", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_z = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_z", predicates=(OrPattern(False, True),)),
        default=False,
    )
    x_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_mask", predicates=(RangePattern(lo=0, hi=127),)),
        default=0,
    )
    y_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "y_mask",
            predicates=(
                SwitchPattern(
                    "mode", {ProductMode.MATRIX: RangePattern(lo=0, hi=127), ProductMode.VECTOR: 0}
                ),
            ),
        ),
        default=0,
    )
    acc = ParamDef(
        kind="input",
        effect=R | W,
        pattern=TensorPattern(
            shape=(SwitchPattern("mode", {ProductMode.MATRIX: 8, ProductMode.VECTOR: 1}), 8),
            dtype=WildcardPattern("acc_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: LayoutPattern((8, 8), (64, 1)),
                    ProductMode.VECTOR: LayoutPattern((1, 8), (0, 1)),
                },
            ),
        ),
    )
    lhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(
                SwitchPattern("mode", {ProductMode.MATRIX: 8, ProductMode.VECTOR: 1}),
                SwitchPattern("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 8}),
            ),
            dtype=WildcardPattern("y_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: LayoutPattern((8, 1), (1, 0)),
                    ProductMode.VECTOR: LayoutPattern((1, 8), (0, 1)),
                },
            ),
        ),
    )
    rhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(1, 8),
            dtype=WildcardPattern("x_dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((1, 8), (0, 1)),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="nn", name="fms16")
class Fms16(AmxOp):
    hir_matches = (
        HirMatch(
            Unary,
            when={"kind": UnaryKind.NEG},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": True},
            operands={"acc": "result", "lhs": "x", "rhs": "x"},
        ),
        HirMatch(
            Binary,
            when={"kind": BinaryKind.SUB},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": False},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        ),
    )
    """One 16-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMS16"
    capability = OpCapability("amx_fms16")
    resource = "amx_engine"
    roles = ("acc", "lhs", "rhs")
    writes = ("acc",)
    register_roles = {"acc": "z", "lhs": "y", "rhs": "x"}
    mode = ParamDef(
        kind="attribute",
        annotation=ProductMode,
        pattern=WildcardPattern("mode"),
        default=ProductMode.MATRIX,
    )
    acc_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern(
            "acc_dtype",
            predicates=(
                SwitchPattern(
                    "mode",
                    {
                        ProductMode.MATRIX: OrPattern(DType.f16, DType.f32),
                        ProductMode.VECTOR: DType.f16,
                    },
                ),
            ),
        ),
        default=DType.f32,
    )
    x_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("x_dtype", predicates=(DType.f16,)),
        default=DType.f16,
    )
    y_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("y_dtype", predicates=(DType.f16,)),
        default=DType.f16,
    )
    out_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        optional=True,
        default=None,
        pattern=WildcardPattern("out_dtype", predicates=(WildcardPattern("acc_dtype"),)),
    )
    x_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    y_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("y_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    z_row = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("z_row", predicates=(RangePattern(lo=0, hi=63),)),
        default=0,
    )
    skip_x = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_x", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_y = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_y", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_z = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_z", predicates=(OrPattern(False, True),)),
        default=False,
    )
    x_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_mask", predicates=(RangePattern(lo=0, hi=127),)),
        default=0,
    )
    y_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "y_mask",
            predicates=(
                SwitchPattern(
                    "mode", {ProductMode.MATRIX: RangePattern(lo=0, hi=127), ProductMode.VECTOR: 0}
                ),
            ),
        ),
        default=0,
    )
    acc = ParamDef(
        kind="input",
        effect=R | W,
        pattern=TensorPattern(
            shape=(SwitchPattern("mode", {ProductMode.MATRIX: 32, ProductMode.VECTOR: 1}), 32),
            dtype=WildcardPattern("acc_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: SwitchPattern(
                        "acc_dtype",
                        {
                            DType.f16: LayoutPattern((32, 32), (64, 1)),
                            DType.f32: LayoutPattern((32, (2, 16)), (32, (16, 1))),
                        },
                    ),
                    ProductMode.VECTOR: LayoutPattern((1, 32), (0, 1)),
                },
            ),
        ),
    )
    lhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(
                SwitchPattern("mode", {ProductMode.MATRIX: 32, ProductMode.VECTOR: 1}),
                SwitchPattern("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 32}),
            ),
            dtype=WildcardPattern("y_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: LayoutPattern((32, 1), (1, 0)),
                    ProductMode.VECTOR: LayoutPattern((1, 32), (0, 1)),
                },
            ),
        ),
    )
    rhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(1, 32),
            dtype=WildcardPattern("x_dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((1, 32), (0, 1)),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="nn", name="fms32")
class Fms32(AmxOp):
    hir_matches = (
        HirMatch(
            Unary,
            when={"kind": UnaryKind.NEG},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": True},
            operands={"acc": "result", "lhs": "x", "rhs": "x"},
        ),
        HirMatch(
            Binary,
            when={"kind": BinaryKind.SUB},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": False},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        ),
    )
    """One 32-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMS32"
    capability = OpCapability("amx_fms32")
    resource = "amx_engine"
    roles = ("acc", "lhs", "rhs")
    writes = ("acc",)
    register_roles = {"acc": "z", "lhs": "y", "rhs": "x"}
    mode = ParamDef(
        kind="attribute",
        annotation=ProductMode,
        pattern=WildcardPattern("mode"),
        default=ProductMode.MATRIX,
    )
    acc_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("acc_dtype", predicates=(DType.f32,)),
        default=DType.f32,
    )
    x_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("x_dtype", predicates=(OrPattern(DType.f16, DType.f32),)),
        default=DType.f32,
    )
    y_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("y_dtype", predicates=(OrPattern(DType.f16, DType.f32),)),
        default=DType.f32,
    )
    out_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        optional=True,
        default=None,
        pattern=WildcardPattern("out_dtype", predicates=(WildcardPattern("acc_dtype"),)),
    )
    x_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    y_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("y_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    z_row = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("z_row", predicates=(RangePattern(lo=0, hi=63),)),
        default=0,
    )
    skip_x = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_x", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_y = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_y", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_z = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_z", predicates=(OrPattern(False, True),)),
        default=False,
    )
    x_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_mask", predicates=(RangePattern(lo=0, hi=127),)),
        default=0,
    )
    y_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "y_mask",
            predicates=(
                SwitchPattern(
                    "mode", {ProductMode.MATRIX: RangePattern(lo=0, hi=127), ProductMode.VECTOR: 0}
                ),
            ),
        ),
        default=0,
    )
    acc = ParamDef(
        kind="input",
        effect=R | W,
        pattern=TensorPattern(
            shape=(SwitchPattern("mode", {ProductMode.MATRIX: 16, ProductMode.VECTOR: 1}), 16),
            dtype=WildcardPattern("acc_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: LayoutPattern((16, 16), (64, 1)),
                    ProductMode.VECTOR: LayoutPattern((1, 16), (0, 1)),
                },
            ),
        ),
    )
    lhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(
                SwitchPattern("mode", {ProductMode.MATRIX: 16, ProductMode.VECTOR: 1}),
                SwitchPattern("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 16}),
            ),
            dtype=WildcardPattern("y_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: LayoutPattern(
                        (16, 1), (SwitchPattern("y_dtype", {DType.f16: 2, DType.f32: 1}), 0)
                    ),
                    ProductMode.VECTOR: LayoutPattern(
                        (1, 16), (0, SwitchPattern("y_dtype", {DType.f16: 2, DType.f32: 1}))
                    ),
                },
            ),
        ),
    )
    rhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(1, 16),
            dtype=WildcardPattern("x_dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern(
                (1, 16), (0, SwitchPattern("x_dtype", {DType.f16: 2, DType.f32: 1}))
            ),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="nn", name="fms64")
class Fms64(AmxOp):
    hir_matches = (
        HirMatch(
            Unary,
            when={"kind": UnaryKind.NEG},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": True},
            operands={"acc": "result", "lhs": "x", "rhs": "x"},
        ),
        HirMatch(
            Binary,
            when={"kind": BinaryKind.SUB},
            attributes={"mode": ProductMode.VECTOR, "skip_y": True, "skip_z": False},
            operands={"acc": "lhs", "lhs": "rhs", "rhs": "rhs"},
        ),
    )
    """One 64-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic = "FMS64"
    capability = OpCapability("amx_fms64")
    resource = "amx_engine"
    roles = ("acc", "lhs", "rhs")
    writes = ("acc",)
    register_roles = {"acc": "z", "lhs": "y", "rhs": "x"}
    mode = ParamDef(
        kind="attribute",
        annotation=ProductMode,
        pattern=WildcardPattern("mode"),
        default=ProductMode.MATRIX,
    )
    acc_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("acc_dtype", predicates=(DType.f64,)),
        default=DType.f64,
    )
    x_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("x_dtype", predicates=(DType.f64,)),
        default=DType.f64,
    )
    y_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("y_dtype", predicates=(DType.f64,)),
        default=DType.f64,
    )
    out_dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        optional=True,
        default=None,
        pattern=WildcardPattern("out_dtype", predicates=(WildcardPattern("acc_dtype"),)),
    )
    x_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    y_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("y_offset", predicates=(RangePattern(lo=0, hi=511),)),
        default=0,
    )
    z_row = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("z_row", predicates=(RangePattern(lo=0, hi=63),)),
        default=0,
    )
    skip_x = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_x", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_y = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_y", predicates=(OrPattern(False, True),)),
        default=False,
    )
    skip_z = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("skip_z", predicates=(OrPattern(False, True),)),
        default=False,
    )
    x_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("x_mask", predicates=(RangePattern(lo=0, hi=127),)),
        default=0,
    )
    y_mask = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "y_mask",
            predicates=(
                SwitchPattern(
                    "mode", {ProductMode.MATRIX: RangePattern(lo=0, hi=127), ProductMode.VECTOR: 0}
                ),
            ),
        ),
        default=0,
    )
    acc = ParamDef(
        kind="input",
        effect=R | W,
        pattern=TensorPattern(
            shape=(SwitchPattern("mode", {ProductMode.MATRIX: 8, ProductMode.VECTOR: 1}), 8),
            dtype=WildcardPattern("acc_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: LayoutPattern((8, 8), (64, 1)),
                    ProductMode.VECTOR: LayoutPattern((1, 8), (0, 1)),
                },
            ),
        ),
    )
    lhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(
                SwitchPattern("mode", {ProductMode.MATRIX: 8, ProductMode.VECTOR: 1}),
                SwitchPattern("mode", {ProductMode.MATRIX: 1, ProductMode.VECTOR: 8}),
            ),
            dtype=WildcardPattern("y_dtype"),
            storage=StorageKind.RMEM,
            layout=SwitchPattern(
                "mode",
                {
                    ProductMode.MATRIX: LayoutPattern((8, 1), (1, 0)),
                    ProductMode.VECTOR: LayoutPattern((1, 8), (0, 1)),
                },
            ),
        ),
    )
    rhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(1, 8),
            dtype=WildcardPattern("x_dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((1, 8), (0, 1)),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)
