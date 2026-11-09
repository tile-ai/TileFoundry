"""AMX register load and store Ops."""

from tilefoundry.interfaces import schedule
from tilefoundry.ir import expr, pattern
from tilefoundry.ir.hir import sharding

from .common import (
    TYPES,
    AmxOp,
    TransferCount,
    instruction_versions,
)
from .semantics import register_semantics

R = expr.MemoryEffect.READ
W = expr.MemoryEffect.WRITE


class Ldx(AmxOp):
    hir_matches = (schedule.HirMatch(sharding.Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "LDX"
    capability = expr.OpCapability("amx_ldx")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "memory", "dst": "x"}
    count = expr.ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=pattern.Wildcard(
            "count",
            predicates=(
                pattern.Switch(
                    "amx_generation",
                    {
                        1: pattern.Or(TransferCount.ONE, TransferCount.TWO),
                        2: pattern.Or(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                        3: pattern.Or(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                        4: pattern.Or(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                    },
                ),
            ),
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "nonconsecutive",
            predicates=(
                pattern.Switch(
                    "count",
                    {
                        TransferCount.ONE: False,
                        TransferCount.TWO: pattern.Switch(
                            "amx_generation",
                            {
                                1: False,
                                2: False,
                                3: pattern.Or(False, True),
                                4: pattern.Or(False, True),
                            },
                        ),
                        TransferCount.FOUR: pattern.Switch(
                            "amx_generation",
                            {
                                1: False,
                                2: False,
                                3: pattern.Or(False, True),
                                4: pattern.Or(False, True),
                            },
                        ),
                    },
                ),
            ),
        ),
        default=False,
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("dtype", predicates=(pattern.Or(*TYPES),)),
        default=expr.DType.u8,
    )
    register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("register", predicates=(pattern.Range(lo=0, hi=7),)),
        default=0,
    )
    memory_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "memory_offset",
            predicates=(
                pattern.Range(lo=0, hi=511),
                pattern.variable("transfer_rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("count")],
                pattern.variable("transfer_lanes") == 512 // pattern.Bits("dtype"),
                pattern.variable("transfer_stride")
                == (512 // pattern.Bits("dtype"))
                * (
                    1
                    + pattern.Table((0, 1))[pattern.variable("nonconsecutive")]
                    * (pattern.Table((0, 1, 4, 0, 2))[pattern.variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.GMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_stride"), 1),
            ),
        ),
    )


class Ldy(AmxOp):
    hir_matches = (schedule.HirMatch(sharding.Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "LDY"
    capability = expr.OpCapability("amx_ldy")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "memory", "dst": "y"}
    count = expr.ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=pattern.Wildcard(
            "count",
            predicates=(
                pattern.Switch(
                    "amx_generation",
                    {
                        1: pattern.Or(TransferCount.ONE, TransferCount.TWO),
                        2: pattern.Or(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                        3: pattern.Or(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                        4: pattern.Or(TransferCount.ONE, TransferCount.TWO, TransferCount.FOUR),
                    },
                ),
            ),
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard(
            "nonconsecutive",
            predicates=(
                pattern.Switch(
                    "count",
                    {
                        TransferCount.ONE: False,
                        TransferCount.TWO: pattern.Switch(
                            "amx_generation",
                            {
                                1: False,
                                2: False,
                                3: pattern.Or(False, True),
                                4: pattern.Or(False, True),
                            },
                        ),
                        TransferCount.FOUR: pattern.Switch(
                            "amx_generation",
                            {
                                1: False,
                                2: False,
                                3: pattern.Or(False, True),
                                4: pattern.Or(False, True),
                            },
                        ),
                    },
                ),
            ),
        ),
        default=False,
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("dtype", predicates=(pattern.Or(*TYPES),)),
        default=expr.DType.u8,
    )
    register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("register", predicates=(pattern.Range(lo=0, hi=7),)),
        default=0,
    )
    memory_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "memory_offset",
            predicates=(
                pattern.Range(lo=0, hi=511),
                pattern.variable("transfer_rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("count")],
                pattern.variable("transfer_lanes") == 512 // pattern.Bits("dtype"),
                pattern.variable("transfer_stride")
                == (512 // pattern.Bits("dtype"))
                * (
                    1
                    + pattern.Table((0, 1))[pattern.variable("nonconsecutive")]
                    * (pattern.Table((0, 1, 4, 0, 2))[pattern.variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.GMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_stride"), 1),
            ),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="memory", name="ldz")
class Ldz(AmxOp):
    hir_matches = (schedule.HirMatch(sharding.Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "LDZ"
    capability = expr.OpCapability("amx_ldz")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "memory", "dst": "z"}
    count = expr.ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=pattern.Wildcard(
            "count", predicates=(pattern.Or(TransferCount.ONE, TransferCount.TWO),)
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("dtype", predicates=(pattern.Or(*TYPES),)),
        default=expr.DType.u8,
    )
    register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("register", predicates=(pattern.Range(lo=0, hi=63),)),
        default=0,
    )
    memory_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "memory_offset",
            predicates=(
                pattern.Range(lo=0, hi=511),
                pattern.variable("transfer_rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("count")],
                pattern.variable("transfer_lanes") == 512 // pattern.Bits("dtype"),
                pattern.variable("transfer_stride")
                == (512 // pattern.Bits("dtype"))
                * (
                    1
                    + pattern.Table((0, 1))[pattern.variable("nonconsecutive")]
                    * (pattern.Table((0, 1, 4, 0, 2))[pattern.variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.GMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="memory", name="ldzi")
class Ldzi(AmxOp):
    hir_matches = (schedule.HirMatch(sharding.Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "LDZI"
    capability = expr.OpCapability("amx_ldzi")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "memory", "dst": "z"}
    count = expr.ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=pattern.Wildcard("count", predicates=(TransferCount.ONE,)),
        default=TransferCount.ONE,
    )
    nonconsecutive = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("dtype", predicates=(pattern.Or(expr.DType.u32, expr.DType.i32, expr.DType.f32),)),
        default=expr.DType.u32,
    )
    register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("register", predicates=(pattern.Range(lo=0, hi=63),)),
        default=0,
    )
    memory_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "memory_offset",
            predicates=(
                pattern.Range(lo=0, hi=511),
                pattern.variable("transfer_rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("count")],
                pattern.variable("transfer_lanes") == 512 // pattern.Bits("dtype"),
                pattern.variable("transfer_stride")
                == (512 // pattern.Bits("dtype"))
                * (
                    1
                    + pattern.Table((0, 1))[pattern.variable("nonconsecutive")]
                    * (pattern.Table((0, 1, 4, 0, 2))[pattern.variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.GMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout((1, (2, 8)), (0, (16, 1))),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="memory", name="stx")
class Stx(AmxOp):
    hir_matches = (schedule.HirMatch(sharding.Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "STX"
    capability = expr.OpCapability("amx_stx")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "x", "dst": "memory"}
    count = expr.ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=pattern.Wildcard(
            "count", predicates=(pattern.Or(TransferCount.ONE, TransferCount.TWO),)
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("dtype", predicates=(pattern.Or(*TYPES),)),
        default=expr.DType.u8,
    )
    register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("register", predicates=(pattern.Range(lo=0, hi=7),)),
        default=0,
    )
    memory_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "memory_offset",
            predicates=(
                pattern.Range(lo=0, hi=511),
                pattern.variable("transfer_rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("count")],
                pattern.variable("transfer_lanes") == 512 // pattern.Bits("dtype"),
                pattern.variable("transfer_stride")
                == (512 // pattern.Bits("dtype"))
                * (
                    1
                    + pattern.Table((0, 1))[pattern.variable("nonconsecutive")]
                    * (pattern.Table((0, 1, 4, 0, 2))[pattern.variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.GMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="memory", name="sty")
class Sty(AmxOp):
    hir_matches = (schedule.HirMatch(sharding.Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "STY"
    capability = expr.OpCapability("amx_sty")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "y", "dst": "memory"}
    count = expr.ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=pattern.Wildcard(
            "count", predicates=(pattern.Or(TransferCount.ONE, TransferCount.TWO),)
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("dtype", predicates=(pattern.Or(*TYPES),)),
        default=expr.DType.u8,
    )
    register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("register", predicates=(pattern.Range(lo=0, hi=7),)),
        default=0,
    )
    memory_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "memory_offset",
            predicates=(
                pattern.Range(lo=0, hi=511),
                pattern.variable("transfer_rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("count")],
                pattern.variable("transfer_lanes") == 512 // pattern.Bits("dtype"),
                pattern.variable("transfer_stride")
                == (512 // pattern.Bits("dtype"))
                * (
                    1
                    + pattern.Table((0, 1))[pattern.variable("nonconsecutive")]
                    * (pattern.Table((0, 1, 4, 0, 2))[pattern.variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.GMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="memory", name="stz")
class Stz(AmxOp):
    hir_matches = (schedule.HirMatch(sharding.Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "STZ"
    capability = expr.OpCapability("amx_stz")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "z", "dst": "memory"}
    count = expr.ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=pattern.Wildcard(
            "count", predicates=(pattern.Or(TransferCount.ONE, TransferCount.TWO),)
        ),
        default=TransferCount.ONE,
    )
    nonconsecutive = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("dtype", predicates=(pattern.Or(*TYPES),)),
        default=expr.DType.u8,
    )
    register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("register", predicates=(pattern.Range(lo=0, hi=63),)),
        default=0,
    )
    memory_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "memory_offset",
            predicates=(
                pattern.Range(lo=0, hi=511),
                pattern.variable("transfer_rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("count")],
                pattern.variable("transfer_lanes") == 512 // pattern.Bits("dtype"),
                pattern.variable("transfer_stride")
                == (512 // pattern.Bits("dtype"))
                * (
                    1
                    + pattern.Table((0, 1))[pattern.variable("nonconsecutive")]
                    * (pattern.Table((0, 1, 4, 0, 2))[pattern.variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.GMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )


@expr.register_op(dialect="T", namespace="amx", category="memory", name="stzi")
class Stzi(AmxOp):
    hir_matches = (schedule.HirMatch(sharding.Reshard, operands={"src": "x", "dst": "result"}),)
    mnemonic = "STZI"
    capability = expr.OpCapability("amx_stzi")
    resource = "amx_engine"
    roles = ("src", "dst")
    writes = ("dst",)
    register_roles = {"src": "z", "dst": "memory"}
    count = expr.ParamDef(
        kind="attribute",
        annotation=TransferCount,
        pattern=pattern.Wildcard("count", predicates=(TransferCount.ONE,)),
        default=TransferCount.ONE,
    )
    nonconsecutive = expr.ParamDef(
        kind="attribute",
        annotation=bool,
        pattern=pattern.Wildcard("nonconsecutive", predicates=(False,)),
        default=False,
    )
    dtype = expr.ParamDef(
        kind="attribute",
        annotation=expr.DType,
        pattern=pattern.Wildcard("dtype", predicates=(pattern.Or(expr.DType.u32, expr.DType.i32, expr.DType.f32),)),
        default=expr.DType.u32,
    )
    register = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("register", predicates=(pattern.Range(lo=0, hi=63),)),
        default=0,
    )
    memory_offset = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard(
            "memory_offset",
            predicates=(
                pattern.Range(lo=0, hi=511),
                pattern.variable("transfer_rows") == pattern.Table((0, 1, 2, 0, 4))[pattern.variable("count")],
                pattern.variable("transfer_lanes") == 512 // pattern.Bits("dtype"),
                pattern.variable("transfer_stride")
                == (512 // pattern.Bits("dtype"))
                * (
                    1
                    + pattern.Table((0, 1))[pattern.variable("nonconsecutive")]
                    * (pattern.Table((0, 1, 4, 0, 2))[pattern.variable("count")] - 1)
                ),
            ),
        ),
        default=0,
    )
    src = expr.ParamDef(
        kind="input",
        effect=R,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.RMEM,
            layout=pattern.Layout((1, (2, 8)), (0, (16, 1))),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=W,
        pattern=pattern.Tensor(
            shape=(pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
            dtype=pattern.Wildcard("dtype"),
            storage=expr.StorageKind.GMEM,
            layout=pattern.Layout(
                (pattern.Wildcard("transfer_rows"), pattern.Wildcard("transfer_lanes")),
                (pattern.Wildcard("transfer_lanes"), 1),
            ),
        ),
    )


LdxV1, LdxV2, LdxV3 = instruction_versions(Ldx, (1, 2, 3), name="ldx", category="memory")

LdyV1, LdyV2, LdyV3 = instruction_versions(Ldy, (1, 2, 3), name="ldy", category="memory")

register_semantics((Ldz, Ldzi, Stx, Sty, Stz, Stzi, LdxV1, LdxV2, LdxV3, LdyV1, LdyV2, LdyV3,))


__all__ = ["Ldz", "Ldzi", "Stx", "Sty", "Stz", "Stzi", "LdxV1", "LdxV2", "LdxV3", "LdyV1", "LdyV2", "LdyV3"]
