"""Lower scheduled HIR for CUDA with the ``sync`` steps at each node that has them.

See [passes §7.1](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/passes.md#71-converthirtotir).
"""

from __future__ import annotations

from dataclasses import replace
from math import prod

from tilefoundry.analysis.emission import (
    copies_carry_backedge,
    copies_carry_entry,
    copies_insert_slice_base,
    copies_insert_slice_update,
    issue_mesh,
    operand_roles,
    required_execution_mesh,
    result_param,
    staged_scalar,
)
from tilefoundry.analysis.sync import SyncMetadata, SyncPlanMetadata
from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var, get_metadata
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.tir import CpAsyncCommit, CpAsyncWait
from tilefoundry.ir.tir.arith import Binary as TirBinary
from tilefoundry.ir.tir.memory import AllocTensor, Fill, PtrOf, TensorView
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.ir.tir.sync import Sync
from tilefoundry.ir.types import (
    Broadcast,
    ComposedLayout,
    PointerType,
    ShardLayout,
    Split,
    StorageKind,
    TensorType,
    TupleType,
)
from tilefoundry.ir.types.int_tuple import flatten as flatten_tuple
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.mesh import axis_keys
from tilefoundry.ir.types.shard_layout import local_layout, shard_layout_of
from tilefoundry.ir.types.utils import _participant_frame, i64_const, static_dim_value
from tilefoundry.passes.transforms.convert_hir_to_tir import (
    _BINDING,
    _INDEX,
    ConvertHIRToTIR,
    Lowering,
    LoweringError,
    View,
    _label,
)
from tilefoundry.target import bind
from tilefoundry_nvidia.cuda import CudaTarget
from tilefoundry_nvidia.cuda.ir.tir.sync.wgmma import (
    FenceProxyAsync,
    WgmmaCommitGroup,
    WgmmaFence,
    WgmmaWaitGroup,
)

_STEPS = {
    ("commit", "cp_async"): CpAsyncCommit,
    ("commit", "wgmma"): WgmmaCommitGroup,
    ("wait", "cp_async"): lambda: CpAsyncWait(n=0),
    ("wait", "wgmma"): lambda: WgmmaWaitGroup(n=0),
    ("fence", "wgmma"): WgmmaFence,
    ("proxy_fence", None): FenceProxyAsync,
}


class CudaLowering(Lowering):
    """The core lowering, emitting the steps the ``sync`` analysis attached to each node."""

    def run(self) -> PrimFunction:
        function = self.context.function
        if get_metadata(function, SyncPlanMetadata) is None:
            raise LoweringError(
                f"{function.name!r}: needs the result of analyze(..., analysis=('sync',))"
            )
        return super().run()

    def _emit_steps(self, owner: Expr, tag, ctx) -> None:
        """Emit the steps *owner*'s ``SyncMetadata`` holds for *tag*, in order."""
        record = get_metadata(owner, SyncMetadata)
        for step in () if record is None else record.at(tag):
            if step.kind == "barrier":
                ctx.builder.add(Evaluate(Sync(mesh=step.mesh), ()))
            else:
                ctx.builder.add(Evaluate(_STEPS[step.kind, step.group](), ()))

    def visit_MeshRegion(self, region: MeshRegion, ctx):
        self._emit_steps(region, None, ctx)
        return super().visit_MeshRegion(region, ctx)

    def visit_callee(self, call: Call, operands, ctx) -> Expr | View:
        self._emit_steps(call, None, ctx)
        return super().visit_callee(call, operands, ctx)

    def visit_T(self, call: Call, operands, ctx) -> Expr | View:
        self._emit_steps(call, None, ctx)
        return super().visit_T(call, operands, ctx)

    def visit_LoopRegion(self, loop: LoopRegion, ctx) -> Expr:
        carried_values = []
        for slot, param in enumerate(loop.params[: len(loop.yield_values)]):
            initial = self.visit(loop.args[slot], ctx=ctx)
            carried = ctx.result(param)
            if copies_carry_entry(loop, slot, ctx.storage.aliases_of):
                self._emit_steps(loop, ("entry", slot), ctx)
                ctx.builder.transfer(initial, carried, ctx=ctx, source_type=loop.args[slot].type)
            self._memo[id(param)] = (param, carried)
            carried_values.append(carried)
        init = tuple(carried_values)
        induction = Var(loop.induction_var.name, type=_INDEX)
        self._memo[id(loop.induction_var)] = (loop.induction_var, induction)
        self._emit_steps(loop, None, ctx)
        with ctx.builder.region(
            loop=(
                induction,
                self.visit(loop.start, ctx=ctx, dimension=True),
                self.visit(loop.extent, ctx=ctx, dimension=True),
                self.visit(loop.step, ctx=ctx, dimension=True),
            ),
            owner=loop,
        ) as scope:
            inner = self.on_region_start(loop, ctx)
            self.visit(loop.body, inner)
            yielded = tuple(self.visit(value, inner) for value in loop.yield_values)
            for slot, (source, carried, value) in enumerate(
                zip(loop.yield_values, init, yielded, strict=False)
            ):
                if copies_carry_backedge(loop, slot, ctx.storage.aliases_of):
                    self._emit_steps(loop, ("backedge", slot), ctx)
                    ctx.builder.transfer(value, carried, ctx=ctx, source_type=source.type)
            for counter, one in (
                values for key, values in ctx.builder.stage_counters.items() if key[-1] == id(loop)
            ):
                ctx.builder.add(Evaluate(TirBinary(kind=BinaryKind.ADD), (counter, one, counter)))
            scope.result = (
                init[0]
                if len(init) == 1
                else Tuple(init, type=TupleType(tuple(value.type for value in init)))
            )
        return scope.result

    def visit_InsertSlice(self, call: Call, operands, ctx) -> Expr:
        target, update = operands[:2]
        result = ctx.result(call)
        if copies_insert_slice_base(call, ctx.storage.aliases_of):
            self._emit_steps(call, "base", ctx)
            ctx.builder.transfer(target, result, ctx=ctx, source_type=call.args[0].type)
        if not copies_insert_slice_update(call, ctx.storage.aliases_of):
            return result
        starts, desired = ctx.storage.window(
            result,
            update.type,
            tuple(
                self.visit(value, ctx=ctx, dimension=True)
                for value in (
                    call.args[2].elements
                    if isinstance(call.args[2], Tuple)
                    else (call.args[2], *(i64_const(0) for _ in result.type.shape[1:]))
                )
            ),
        )
        self._emit_steps(call, "update", ctx)
        ctx.builder.transfer(
            update,
            result,
            starts=starts,
            sizes=tuple(update.type.shape),
            desired=desired,
            ctx=ctx,
            source_type=call.args[1].type,
        )
        return result

    def visit_Zeros(self, call: Call, operands, ctx) -> Expr | View:
        result = ctx.result(call)
        scalar = Constant(
            getattr(call.target, "value", 0.0), type=TensorType.scalar(call.type.dtype)
        )
        self._emit_steps(call, None, ctx)
        ctx.builder.transfer(scalar, result, op=Fill(), ctx=ctx)
        return result

    visit_FullLike = visit_Zeros

    def default_visit_T(self, call: Call, op, operands, ctx) -> Expr:
        mesh = ctx.enclosing_mesh()
        plan, types, owner = ctx.issues.plan(call, op, mesh)
        params = plan.params
        supplied, produced = operand_roles(call, params)
        destination_operand = (
            call.args[params.index(produced[0])]
            if isinstance(call.target, ScheduleOp) and produced
            else None
        )

        written = ctx.result(call) if produced else None
        read_values = {}
        for index, (param, operand) in enumerate(zip(supplied, call.args, strict=True)):
            value = written if operand is destination_operand else operands[index]
            if (
                not isinstance(call.target, ScheduleOp)
                and operand.type != types[params.index(param)]
            ):
                value = ctx.storage.operand(value, types[params.index(param)], whole_tensor=True)
            if staged_scalar(value.type):
                scalar_type = replace(value.type, storage=StorageKind.RMEM)
                scalar = Var(ctx.names.fresh("scalar"), type=scalar_type)
                with ctx.builder.region():
                    ctx.builder.bind(
                        scalar, Call(AllocTensor(tensor_type=scalar_type), (), type=scalar_type)
                    )
                ctx.builder.add(Evaluate(Fill(), (scalar, ctx.builder.value(value, ctx=ctx))))
                value = scalar
            read_values[param.name] = value
        issued_operands: list[tuple[str, Expr | None]] = []
        for param in params:
            value = (
                read_values[param.name]
                if isinstance(call.target, ScheduleOp) or param.effect & MemoryEffect.READ
                else written
            )
            if value is None:
                raise LoweringError(f"{_label(call)} has no storage for {param.name}")
            issued_operands.append((param.name, value))

        types = tuple(
            replace(type_, storage=StorageKind.RMEM)
            if type_.shape == () and type_.storage is StorageKind.UMAT
            else type_
            for type_ in types
        )
        required = required_execution_mesh(op)
        issuers = issue_mesh(op, owner, mesh)
        if required is None:
            if mesh is None:
                raise LoweringError(f"{_label(call)} instruction issue has no lexical mesh")
            with ctx.builder.region(
                mesh=issuers, binding=Var(ctx.names.fresh("threads"), type=_BINDING)
            ):
                issued = []
                for param, (role, value) in zip(params, issued_operands, strict=True):
                    desired = types[params.index(param)]
                    issued.append(
                        ctx.builder.value(
                            ctx.storage.operand(value, desired, stem=f"{role}_view"), ctx=ctx
                        )
                    )
                attributes = {
                    param.name: getattr(op, param.name)
                    for param in type(op)._op_schema.signature
                    if param.kind == "attribute"
                    and (
                        not param.has_default
                        or getattr(op, param.name) != param.default
                        or param.name == "fill"
                    )
                }
                self._emit_steps(call, "issue", ctx)
                ctx.builder.add(Evaluate(type(op)(**attributes), tuple(issued)))
                self._emit_steps(call, "after", ctx)
            result = result_param(params)
            return read_values[result.name] if result.effect & MemoryEffect.READ else written
        for region in ctx.builder.blocks[:-1]:
            for key, value in region.views.items():
                ctx.builder.blocks[-1].views.setdefault(key, value)
        locals_ = []
        unit_coordinates = {}
        for (role, value), type_, rest in zip(issued_operands, types, plan.rests, strict=True):
            base = ctx.builder.value(ctx.storage.operand(value, type_, whole_tensor=True), ctx=ctx)
            binding_mesh = rest.mesh if isinstance(rest, ShardLayout) else issuers
            _frame, dropped = _participant_frame(binding_mesh, required)
            source = shard_layout_of(base.type.layout)
            attrs = (
                list(source.attrs)
                if source is not None
                else [Broadcast()] * len(axis_keys(binding_mesh))
            )
            coords = [None] * len(attrs)
            for mesh_axis, (topology, _name) in enumerate(axis_keys(binding_mesh)[:dropped]):
                attr = rest.attrs[mesh_axis]
                attrs[mesh_axis] = attr
                coordinate = 0
                if isinstance(attr, Split) and not ctx.target.storage_owners.private(
                    type_.storage, topology
                ):
                    key = id(binding_mesh), mesh_axis
                    if key not in unit_coordinates:
                        coord = Call(
                            MeshCoord(mesh=binding_mesh), (i64_const(mesh_axis),), type=_BINDING
                        )
                        unit_coordinates[key] = ctx.builder.bind(
                            Var(ctx.names.fresh("unit"), type=_BINDING), coord
                        )
                    coordinate = unit_coordinates[key]
                coords[mesh_axis] = coordinate
            address = source.layout if source is not None else base.type.layout
            local, offset = local_layout(
                ShardLayout(address, tuple(attrs), binding_mesh), base.type.shape, tuple(coords)
            )
            if any(isinstance(attr, Split) for attr in attrs[:dropped]):
                primitive = local.layout if isinstance(local, ShardLayout) else local
                outer = primitive.outer if isinstance(primitive, ComposedLayout) else primitive
                local_shape = tuple(prod(flatten_tuple(mode)) for mode in outer.shape)
                original = address.outer if isinstance(address, ComposedLayout) else address
                if not any(isinstance(mode, tuple) for mode in original.shape):
                    flat = flatten(outer)
                    primitive = (
                        replace(primitive, outer=flat)
                        if isinstance(primitive, ComposedLayout)
                        else flat
                    )
                if static_dim_value(offset) != 0:
                    primitive = ComposedLayout(None, offset, primitive)
                local = (
                    replace(local, layout=primitive)
                    if source is not None and isinstance(local, ShardLayout)
                    else primitive
                )
                local_type = replace(base.type, shape=local_shape, layout=local)
                pointer_type = PointerType(base.type.dtype, base.type.storage)
                pointer = ctx.builder.bind(
                    Var(ctx.names.fresh(f"{role}_pointer"), type=pointer_type),
                    Call(PtrOf(), (base,), type=pointer_type),
                )
                base = Call(TensorView(layout=local, shape=local_shape), (pointer,), type=local_type)
            locals_.append(base)
        with ctx.builder.region(mesh=issuers, binding=Var(ctx.names.fresh("threads"), type=_BINDING)):
            locals_ = [
                ctx.builder.bind(Var(ctx.names.fresh(f"{role}_local"), type=base.type), base)
                if isinstance(base, Call) and isinstance(base.target, TensorView)
                else base
                for (role, _value), base in zip(issued_operands, locals_, strict=True)
            ]
            for windows in ctx.issues.points(plan):
                issued = []
                for (role, _value), base, desired, starts in zip(
                    issued_operands, locals_, plan.tiles, windows, strict=True
                ):
                    item = ctx.storage.issue_window(base, desired, starts, ctx.builder.loops)
                    issued.append(
                        ctx.builder.bind(Var(ctx.names.fresh(f"{role}_view"), type=item.type), item)
                        if isinstance(item, Call)
                        else item
                    )
                self._emit_steps(call, "issue", ctx)
                ctx.builder.add(Evaluate(op, tuple(issued)))
            self._emit_steps(call, "after", ctx)
        result = result_param(params)
        if result.effect & MemoryEffect.READ:
            return read_values[result.name]
        if written is None:
            raise LoweringError(f"{_label(call)} produced no addressable result")
        return written


@bind(CudaTarget, ConvertHIRToTIR)
class CudaConvertHIRToTIR(ConvertHIRToTIR):
    """Lower scheduled HIR with the visitor that emits the sync steps."""

    visitor = CudaLowering


__all__ = ["CudaConvertHIRToTIR", "CudaLowering"]
