from tilefoundry.ir.core import OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.pattern import (
    LayoutPattern,
    OrPattern,
    RangePattern,
    SwitchPattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.pattern.predicates import Bits, Table, variable
from tilefoundry.ir.types import DType, StorageKind
from tilefoundry.visitor_registry.candidates import HirMatch

from .common import (
    TYPES,
    AmxOp,
    TransferCount,
    bind,
    instruction_versions,
)

R = MemoryEffect.READ
W = MemoryEffect.WRITE


class Ldx(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "LDX"
    capability = OpCapability("amx_ldx")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "memory", "dst": "x"}
    count = ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=WildcardPattern(
            "count",
            predicates=(
                SwitchPattern(
                    "amx_generation",
                    {
                        1: OrPattern(TransferCount.ONE, TransferCount.TWO),
                        2: OrPattern(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                        3: OrPattern(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                        4: OrPattern(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                    },
                ),
            ),
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern(
            "nonconsecutive",
            predicates=(
                SwitchPattern(
                    "count",
                    {
                        TransferCount.ONE: False,
                        TransferCount.TWO: SwitchPattern(
                            "amx_generation",
                            {
                                1: False,
                                2: False,
                                3: OrPattern(False, True),
                                4: OrPattern(False, True),
                            },
                        ),
                        TransferCount.FOUR: SwitchPattern(
                            "amx_generation",
                            {
                                1: False,
                                2: False,
                                3: OrPattern(False, True),
                                4: OrPattern(False, True),
                            },
                        ),
                    },
                ),
            ),
        ),
        default=False,
    )
    dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("dtype", predicates=(OrPattern(*TYPES),)),
        default=DType.u8,
    )
    register = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("register", predicates=(RangePattern(lo=0, hi=7),)),
        default=0,
    )
    memory_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "memory_offset",
            predicates=(
                RangePattern(lo=0, hi=511),
                variable("transfer_rows") == Table((0, 1, 2, 0, 4))[variable("count")],
                variable("transfer_lanes") == 512 // Bits("dtype"),
                variable("transfer_stride")
                == (512 // Bits("dtype"))
                * (
                    1
                    + Table((0, 1))[variable("nonconsecutive")]
                    * (Table((0, 1, 4, 0, 2))[variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.GMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=W,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_stride"), 1),
            ),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


class Ldy(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "LDY"
    capability = OpCapability("amx_ldy")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "memory", "dst": "y"}
    count = ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=WildcardPattern(
            "count",
            predicates=(
                SwitchPattern(
                    "amx_generation",
                    {
                        1: OrPattern(TransferCount.ONE, TransferCount.TWO),
                        2: OrPattern(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                        3: OrPattern(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                        4: OrPattern(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                    },
                ),
            ),
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern(
            "nonconsecutive",
            predicates=(
                SwitchPattern(
                    "count",
                    {
                        TransferCount.ONE: False,
                        TransferCount.TWO: SwitchPattern(
                            "amx_generation",
                            {
                                1: False,
                                2: False,
                                3: OrPattern(False, True),
                                4: OrPattern(False, True),
                            },
                        ),
                        TransferCount.FOUR: SwitchPattern(
                            "amx_generation",
                            {
                                1: False,
                                2: False,
                                3: OrPattern(False, True),
                                4: OrPattern(False, True),
                            },
                        ),
                    },
                ),
            ),
        ),
        default=False,
    )
    dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("dtype", predicates=(OrPattern(*TYPES),)),
        default=DType.u8,
    )
    register = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("register", predicates=(RangePattern(lo=0, hi=7),)),
        default=0,
    )
    memory_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "memory_offset",
            predicates=(
                RangePattern(lo=0, hi=511),
                variable("transfer_rows") == Table((0, 1, 2, 0, 4))[variable("count")],
                variable("transfer_lanes") == 512 // Bits("dtype"),
                variable("transfer_stride")
                == (512 // Bits("dtype"))
                * (
                    1
                    + Table((0, 1))[variable("nonconsecutive")]
                    * (Table((0, 1, 4, 0, 2))[variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.GMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=W,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_stride"), 1),
            ),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="memory", name="ldz")
class Ldz(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "LDZ"
    capability = OpCapability("amx_ldz")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "memory", "dst": "z"}
    count = ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=WildcardPattern(
            "count", predicates=(OrPattern(TransferCount.ONE, TransferCount.TWO),)
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("dtype", predicates=(OrPattern(*TYPES),)),
        default=DType.u8,
    )
    register = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("register", predicates=(RangePattern(lo=0, hi=63),)),
        default=0,
    )
    memory_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "memory_offset",
            predicates=(
                RangePattern(lo=0, hi=511),
                variable("transfer_rows") == Table((0, 1, 2, 0, 4))[variable("count")],
                variable("transfer_lanes") == 512 // Bits("dtype"),
                variable("transfer_stride")
                == (512 // Bits("dtype"))
                * (
                    1
                    + Table((0, 1))[variable("nonconsecutive")]
                    * (Table((0, 1, 4, 0, 2))[variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.GMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=W,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="memory", name="ldzi")
class Ldzi(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "LDZI"
    capability = OpCapability("amx_ldzi")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "memory", "dst": "z"}
    count = ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=WildcardPattern("count", predicates=(TransferCount.ONE,)),
        default=TransferCount.ONE,
    )
    nonconsecutive = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("dtype", predicates=(OrPattern(DType.u32, DType.i32, DType.f32),)),
        default=DType.u32,
    )
    register = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("register", predicates=(RangePattern(lo=0, hi=63),)),
        default=0,
    )
    memory_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "memory_offset",
            predicates=(
                RangePattern(lo=0, hi=511),
                variable("transfer_rows") == Table((0, 1, 2, 0, 4))[variable("count")],
                variable("transfer_lanes") == 512 // Bits("dtype"),
                variable("transfer_stride")
                == (512 // Bits("dtype"))
                * (
                    1
                    + Table((0, 1))[variable("nonconsecutive")]
                    * (Table((0, 1, 4, 0, 2))[variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.GMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=W,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((1, (2, 8)), (0, (16, 1))),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="memory", name="stx")
class Stx(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "STX"
    capability = OpCapability("amx_stx")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "x", "dst": "memory"}
    count = ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=WildcardPattern(
            "count", predicates=(OrPattern(TransferCount.ONE, TransferCount.TWO),)
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("dtype", predicates=(OrPattern(*TYPES),)),
        default=DType.u8,
    )
    register = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("register", predicates=(RangePattern(lo=0, hi=7),)),
        default=0,
    )
    memory_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "memory_offset",
            predicates=(
                RangePattern(lo=0, hi=511),
                variable("transfer_rows") == Table((0, 1, 2, 0, 4))[variable("count")],
                variable("transfer_lanes") == 512 // Bits("dtype"),
                variable("transfer_stride")
                == (512 // Bits("dtype"))
                * (
                    1
                    + Table((0, 1))[variable("nonconsecutive")]
                    * (Table((0, 1, 4, 0, 2))[variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=W,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.GMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="memory", name="sty")
class Sty(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "STY"
    capability = OpCapability("amx_sty")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "y", "dst": "memory"}
    count = ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=WildcardPattern(
            "count", predicates=(OrPattern(TransferCount.ONE, TransferCount.TWO),)
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("dtype", predicates=(OrPattern(*TYPES),)),
        default=DType.u8,
    )
    register = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("register", predicates=(RangePattern(lo=0, hi=7),)),
        default=0,
    )
    memory_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "memory_offset",
            predicates=(
                RangePattern(lo=0, hi=511),
                variable("transfer_rows") == Table((0, 1, 2, 0, 4))[variable("count")],
                variable("transfer_lanes") == 512 // Bits("dtype"),
                variable("transfer_stride")
                == (512 // Bits("dtype"))
                * (
                    1
                    + Table((0, 1))[variable("nonconsecutive")]
                    * (Table((0, 1, 4, 0, 2))[variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=W,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.GMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="memory", name="stz")
class Stz(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "STZ"
    capability = OpCapability("amx_stz")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "z", "dst": "memory"}
    count = ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=WildcardPattern(
            "count", predicates=(OrPattern(TransferCount.ONE, TransferCount.TWO),)
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("dtype", predicates=(OrPattern(*TYPES),)),
        default=DType.u8,
    )
    register = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("register", predicates=(RangePattern(lo=0, hi=63),)),
        default=0,
    )
    memory_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "memory_offset",
            predicates=(
                RangePattern(lo=0, hi=511),
                variable("transfer_rows") == Table((0, 1, 2, 0, 4))[variable("count")],
                variable("transfer_lanes") == 512 // Bits("dtype"),
                variable("transfer_stride")
                == (512 // Bits("dtype"))
                * (
                    1
                    + Table((0, 1))[variable("nonconsecutive")]
                    * (Table((0, 1, 4, 0, 2))[variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=W,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.GMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


@register_op(dialect="T", namespace="amx", category="memory", name="stzi")
class Stzi(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "STZI"
    capability = OpCapability("amx_stzi")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "z", "dst": "memory"}
    count = ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=WildcardPattern("count", predicates=(TransferCount.ONE,)),
        default=TransferCount.ONE,
    )
    nonconsecutive = ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=WildcardPattern("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = ParamDef(
        kind="attribute",
        annotation=DType,
        pattern=WildcardPattern("dtype", predicates=(OrPattern(DType.u32, DType.i32, DType.f32),)),
        default=DType.u32,
    )
    register = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("register", predicates=(RangePattern(lo=0, hi=63),)),
        default=0,
    )
    memory_offset = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern(
            "memory_offset",
            predicates=(
                RangePattern(lo=0, hi=511),
                variable("transfer_rows") == Table((0, 1, 2, 0, 4))[variable("count")],
                variable("transfer_lanes") == 512 // Bits("dtype"),
                variable("transfer_stride")
                == (512 // Bits("dtype"))
                * (
                    1
                    + Table((0, 1))[variable("nonconsecutive")]
                    * (Table((0, 1, 4, 0, 2))[variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((1, (2, 8)), (0, (16, 1))),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=W,
        pattern=TensorPattern(
            shape=(WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
            dtype=WildcardPattern("dtype"),
            storage=StorageKind.GMEM,
            layout=LayoutPattern(
                (WildcardPattern("transfer_rows"), WildcardPattern("transfer_lanes")),
                (WildcardPattern("transfer_lanes"), 1),
            ),
        ),
    )

    def __init__(self, **attrs):
        bind(self, attrs)


LdxV1, LdxV2, LdxV3 = instruction_versions(Ldx, (1, 2, 3), name="ldx", category="memory")

LdyV1, LdyV2, LdyV3 = instruction_versions(Ldy, (1, 2, 3), name="ldy", category="memory")
