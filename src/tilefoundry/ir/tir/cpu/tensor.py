"""Explicit CPU tensor-kernel contracts for hybrid CPU/AMX schedules."""

from dataclasses import replace
from enum import Enum

import isl

from tilefoundry.evaluator.registry import eval_registry, register_schedule_eval
from tilefoundry.ir.core import Call, Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef, collect_param_defs
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.softplus import Softplus
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.hir.nn.sigmoid import Sigmoid
from tilefoundry.ir.hir.nn.silu import Silu
from tilefoundry.ir.hir.nn.softmax import SoftMax
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.hir.tensor.full_like import FullLike
from tilefoundry.ir.hir.tensor.index_select import IndexSelect
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.hir.tensor.repeat_interleave import RepeatInterleave
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    TensorPattern,
    WildcardPattern,
)
from tilefoundry.ir.types import DType, StorageKind, UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
    register_access_relation,
)
from tilefoundry.visitor_registry.contexts import TrafficBytes, TypeInferContext
from tilefoundry.visitor_registry.registries import (
    cost_evaluator_registry,
    register_cost_evaluator,
    typeinfer_registry,
)

CORE = MeshPattern(
    ("core",),
    ComposedLayoutPattern(
        inner=None, offset=WildcardPattern("core_offset"), outer=LayoutPattern(((1,),), ((1,),))
    ),
)
INSTRUCTIONS = {}


def _semantic_type(ty):
    """Treat scalar-by-value inputs as uniform HIR operands."""
    return replace(ty, storage=StorageKind.UMAT) if getattr(ty, "shape", None) == () else ty


def _register(hir, arity):
    original_attrs = tuple(p for p in collect_param_defs(hir) if p.kind == "attribute")
    attributes = {p.name: replace(p, effect=None) for p in original_attrs}
    if "dtype" not in attributes:
        attributes["dtype"] = ParamDef(kind="attribute", annotation=DType, default=DType.f32)

    def constructor(self, **bound):
        for name, value in tuple(bound.items()):
            if isinstance(value, str) and name in ("dtype", "out_dtype"):
                bound[name] = DType.from_name(value)
        for parameter in original_attrs:
            value = bound.get(parameter.name)
            annotation = parameter.annotation
            if (
                isinstance(value, str)
                and isinstance(annotation, type)
                and issubclass(annotation, Enum)
            ):
                bound[parameter.name] = annotation(value)
        Op.__init__(self, **bound)

    attrs = {
        "__init__": constructor,
        "__module__": __name__,
        "capability": OpCapability("cpu_tensor"),
        "resource": "cpu_engine",
        "execution_mesh": CORE,
        **attributes,
    }
    for index in range(arity):
        attrs[f"arg{index}"] = ParamDef(
            kind="input", effect=MemoryEffect.READ, pattern=TensorPattern()
        )
    attrs["dst"] = ParamDef(
        kind="input",
        effect=MemoryEffect.WRITE,
        pattern=TensorPattern(storage=StorageKind.GMEM),
    )
    name = "cpu_" + hir._op_schema.name + (f"_{arity}" if hir is Concat else "")
    cls = register_op(dialect="T", category="cpu", name=name)(
        type("Cpu" + hir.__name__ + str(arity), (Op,), attrs)
    )

    def reconstruct(op):
        return hir(**{p.name: getattr(op, p.name) for p in original_attrs})

    @register_typeinfer(cls)
    def infer(call, ctx):
        return UnitType()

    @register_access_relation(cls)
    def relation(call, ctx):
        original = Call(reconstruct(call.target), call.args[:arity], type=UnitType())
        local = TypeInferContext(
            memo={id(arg): (arg, _semantic_type(ctx.type_of(arg))) for arg in original.args}
        )
        original.type = typeinfer_registry.lookup(hir)(original, local).type
        local.memo[id(original)] = (original, original.type)
        boundaries = access_relation_registry.lookup(hir)(original, local)
        output = boundaries[-1]
        if not output.relation.is_single_valued():
            relation = output.relation
            n = relation.dim(isl.dim_type.IN)
            rank = relation.dim(isl.dim_type.OUT)
            write = relation.move_dims(isl.dim_type.IN, n, isl.dim_type.OUT, 0, rank).add_dims(
                isl.dim_type.OUT, rank
            )
            for axis in range(rank):
                write = write.equate(isl.dim_type.IN, n + axis, isl.dim_type.OUT, axis)
            domain = write.domain()
            boundaries = tuple(
                replace(
                    item,
                    relation=item.relation.add_dims(isl.dim_type.IN, rank).intersect_domain(domain),
                )
                for item in boundaries[:-1]
            ) + (replace(output, relation=write),)
        return (*boundaries, boundaries[-1])

    @register_verify_stmt(cls)
    def verify(call, ctx):
        target = ctx.resolve_target()
        if target is not None and target.name not in ("cpu", "amx"):
            ctx.error(call, "Host tensor kernels require a CPU/AMX target")
        original = Call(reconstruct(call.target), call.args[:arity], type=UnitType())
        local = TypeInferContext(
            memo={id(arg): (arg, _semantic_type(ctx.type_of(arg))) for arg in original.args}
        )
        expected = typeinfer_registry.lookup(hir)(original, local).type
        actual = ctx.type_of(call.args[-1])
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            ctx.error(call, f"CPU tensor kernel writes {actual}, expected {expected}")

    @register_cost_evaluator(cls)
    def cost(call, ctx):
        original = Call(reconstruct(call.target), call.args[:arity], type=call.type)
        result = cost_evaluator_registry.lookup(hir)(original, ctx)
        return replace(result, traffic=(*result.traffic, TrafficBytes()))

    @register_schedule_eval(cls)
    def evaluate(ctx):
        op = reconstruct(ctx.op)
        result = eval_registry.lookup(hir)(ctx.for_op(op, ctx.args[:arity], ctx.result_type))
        return result

    INSTRUCTIONS[hir, arity] = cls
    globals()[cls.__name__] = cls


for hir, arity in (
    (Reshape, 1),
    (Transpose, 1),
    (Softplus, 1),
    (Binary, 2),
    (Unary, 1),
    (MatMul, 2),
    (RMSNorm, 2),
    (SoftMax, 1),
    (Silu, 1),
    (Sigmoid, 1),
    (Cast, 1),
    (Reduce, 1),
    (IndexSelect, 2),
    (RepeatInterleave, 1),
    (FullLike, 1),
    (Arange, 0),
    (Concat, 2),
):
    _register(hir, arity)


def instruction_for(operation, arity, dtype):
    cls = INSTRUCTIONS[type(operation), arity]
    attrs = {
        p.name: getattr(operation, p.name)
        for p in collect_param_defs(type(operation))
        if p.kind == "attribute"
    }
    if "dtype" in attrs:
        attrs["dtype"] = dtype
    return cls(**attrs)
