"""Apple AMX FP16 outer-product and memory-transfer contracts.

X holds 32 weight columns, Y holds 32 activation rows, and Z accumulates
32x32 FP32 results. The native module implements the same explicit contract.
"""

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    StorageKind,
    Topology,
    UnitType,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    matmul_relations,
    register_access_relation,
)

Z_LAYOUT = Layout((32, (16, 2)), (32, (1, 16)))
AMX = Mesh((Topology("amx", 1),), Layout((1,), (1,)), ("unit",))


def mesh_pattern(name):
    return MeshPattern(
        (name,),
        ComposedLayoutPattern(
            inner=None,
            offset=WildcardPattern(name + "_offset"),
            outer=LayoutPattern(((1,),), ((1,),)),
        ),
    )


AMX_PATTERN = mesh_pattern("amx")
R = MemoryEffect.READ
W = MemoryEffect.WRITE


@register_op(dialect="T", category="amx", name="amx_fma16")
class Fma16(Op):
    capability = OpCapability("amx_outer_product")
    resource = "amx_engine"
    execution_mesh = AMX_PATTERN
    acc = ParamDef(
        kind="input",
        effect=R | W,
        pattern=TensorPattern(
            shape=(32, 32),
            dtype=DType.f32,
            storage=StorageKind.RMEM,
            layout=LayoutPattern.from_layout(Z_LAYOUT),
        ),
    )
    lhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(shape=(32, 1), dtype=DType.f16, storage=StorageKind.RMEM),
    )
    rhs = ParamDef(
        kind="input",
        effect=R,
        pattern=TensorPattern(shape=(1, 32), dtype=DType.f16, storage=StorageKind.RMEM),
    )


@register_access_relation(Fma16)
def relation(call, ctx):
    _, a, b = (ctx.type_of(v) for v in call.args)
    *operands, out = matmul_relations(a.shape, b.shape, (-2, -1, -1, -2))
    return (out, *operands, out)


@register_schedule_eval(Fma16)
def evaluate(ctx):
    c, a, b = (v.data for v in ctx.args)
    return TensorValue(data=c + a.float() @ b.float(), type=ctx.result_type)


@register_typeinfer(Fma16)
def infer(call, ctx):
    return UnitType()


def transfer(name, shape, dtype, storage, execution):
    cls = register_op(dialect="T", category="amx", name=name)(
        type(
            name,
            (Op,),
            dict(
                __module__=__name__,
                capability=OpCapability("amx_outer_product"),
                resource="amx_engine",
                execution_mesh=execution,
                src=ParamDef(
                    kind="input",
                    effect=R,
                    pattern=TensorPattern(
                        shape=shape,
                        dtype=dtype,
                        storage=StorageKind.RMEM
                        if storage is StorageKind.GMEM
                        else StorageKind.GMEM,
                        layout=LayoutPattern.from_layout(Z_LAYOUT)
                        if storage is StorageKind.GMEM
                        else None,
                    ),
                ),
                dst=ParamDef(
                    kind="input",
                    effect=W,
                    pattern=TensorPattern(shape=shape, dtype=dtype, storage=storage),
                ),
            ),
        )
    )
    register_typeinfer(cls)(lambda call, ctx: UnitType())
    register_access_relation(cls)(identity_relations)
    register_schedule_eval(cls)(
        lambda ctx: TensorValue(data=ctx.args[0].data.clone(), type=ctx.result_type)
    )
    globals()[name] = cls
    return cls


LoadX = transfer("amx_load_x", (1, 32), DType.f16, StorageKind.RMEM, AMX_PATTERN)
LoadY = transfer("amx_load_y", (32, 1), DType.f16, StorageKind.RMEM, AMX_PATTERN)
StoreZ = transfer("amx_store", (32, 32), DType.f32, StorageKind.GMEM, AMX_PATTERN)
