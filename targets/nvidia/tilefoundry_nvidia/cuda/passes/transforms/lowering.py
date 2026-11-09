"""Lower scheduled HIR for CUDA with the ``sync`` steps at each node that has them.

See [passes §7.1](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/passes.md#71-converthirtotir).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from types import MappingProxyType

from tilefoundry import target
from tilefoundry.interfaces import schedule
from tilefoundry.ir import core, dim, expr, hir, tir, types, utils
from tilefoundry.ir.tir import memory
from tilefoundry.passes import transforms
from tilefoundry_nvidia.cuda import analysis
from tilefoundry_nvidia.cuda import target as cuda_target
from tilefoundry_nvidia.cuda.ir.tir import memory as cuda_memory
from tilefoundry_nvidia.cuda.ir.tir import sync

STEPS = MappingProxyType({
    (analysis.StepKind.COMMIT, analysis.Group.CP_ASYNC): tir.async_copy.CpAsyncCommit,
    (analysis.StepKind.COMMIT, analysis.Group.WGMMA): sync.WgmmaCommitGroup,
    (analysis.StepKind.WAIT, analysis.Group.CP_ASYNC): lambda: tir.async_copy.CpAsyncWait(n=0),
    (analysis.StepKind.WAIT, analysis.Group.WGMMA): lambda: sync.WgmmaWaitGroup(n=0),
    (analysis.StepKind.FENCE, analysis.Group.WGMMA): sync.WgmmaFence,
})


def staged_scalar(type_) -> bool:
    """Whether lowering first fills an operand of *type_* into a register scalar."""
    return (
        isinstance(type_, types.TensorType) and type_.shape == () and type_.storage is expr.StorageKind.UMAT
    )


@dataclass(frozen=True)
class Issue:
    """What ``default_visit_T`` issues an instruction with, before it issues it.

    ``operands`` pairs each parameter's name with its value, in parameter
    order; ``types`` are the types the instruction is issued with.
    """

    plan: schedule.Projection
    types: tuple
    operands: tuple[tuple[str, expr.Expr], ...]
    read_values: dict[str, expr.Expr]
    written: expr.Expr | None
    enclosing: types.Mesh | None
    issuers: types.Mesh | None


class CudaLowering(transforms.Lowering):
    """The core lowering, emitting the steps the ``sync`` analysis attached to each node."""

    def run(self) -> core.PrimFunction:
        function = self.context.function
        if expr.get_metadata(function, analysis.SyncPlanMetadata) is None:
            raise transforms.LoweringError(
                f"{function.name!r}: needs the result of analyze(..., analysis=('sync',))"
            )
        return super().run()

    def emit_steps(self, owner: expr.Expr, tag, ctx) -> None:
        """Emit the steps *owner*'s ``SyncMetadata`` holds for *tag*, in order."""
        record = expr.get_metadata(owner, analysis.SyncMetadata)
        for step in () if record is None else record.at(tag):
            if step.kind is analysis.StepKind.BARRIER:
                ctx.builder.add(core.Evaluate(tir.sync.Sync(mesh=step.mesh), ()))
            elif step.kind is analysis.StepKind.PROXY_FENCE:
                ctx.builder.add(core.Evaluate(sync.FenceProxyAsync(storage=step.storage), ()))
            else:
                ctx.builder.add(core.Evaluate(STEPS[step.kind, step.group](), ()))

    def visit_MeshRegion(self, region: core.MeshRegion, ctx):
        self.emit_steps(region, None, ctx)
        return super().visit_MeshRegion(region, ctx)

    def visit_callee(self, call: expr.Call, operands, ctx) -> expr.Expr | transforms.View:
        self.emit_steps(call, None, ctx)
        return super().visit_callee(call, operands, ctx)

    def visit_T(self, call: expr.Call, operands, ctx) -> expr.Expr | transforms.View:
        self.emit_steps(call, None, ctx)
        return super().visit_T(call, operands, ctx)

    def visit_LoopRegion(self, loop: core.LoopRegion, ctx) -> expr.Expr:
        carried_values = []
        for slot, param in enumerate(loop.params[: len(loop.yield_values)]):
            initial = self.visit(loop.args[slot], ctx=ctx)
            carried = ctx.result(param)
            if analysis.copies_carry_entry(loop, slot, ctx.storage.aliases_of):
                self.emit_steps(loop, ("entry", slot), ctx)
                ctx.builder.transfer(initial, carried, ctx=ctx, source_type=loop.args[slot].type)
            self.memo[id(param)] = (param, carried)
            carried_values.append(carried)
        init = tuple(carried_values)
        induction = expr.Var(loop.induction_var.name, type=transforms.INDEX)
        self.memo[id(loop.induction_var)] = (loop.induction_var, induction)
        self.emit_steps(loop, None, ctx)
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
                if analysis.copies_carry_backedge(loop, slot, ctx.storage.aliases_of):
                    self.emit_steps(loop, ("backedge", slot), ctx)
                    ctx.builder.transfer(value, carried, ctx=ctx, source_type=source.type)
            for counter, one in (
                values for key, values in ctx.builder.stage_counters.items() if key[-1] == id(loop)
            ):
                ctx.builder.add(core.Evaluate(tir.arith.Binary(kind=expr.BinaryKind.ADD), (counter, one, counter)))
            scope.yield_value(
                init[0]
                if len(init) == 1
                else expr.Tuple(init, type=types.TupleType(tuple(value.type for value in init)))
            )
        return scope.result

    def visit_InsertSlice(self, call: expr.Call, operands, ctx) -> expr.Expr:
        tgt, update = operands[:2]
        result = ctx.result(call)
        if analysis.copies_insert_slice_base(call, ctx.storage.aliases_of):
            self.emit_steps(call, "base", ctx)
            ctx.builder.transfer(tgt, result, ctx=ctx, source_type=call.args[0].type)
        if not analysis.copies_insert_slice_update(call, ctx.storage.aliases_of):
            return result
        starts, desired = ctx.storage.window(
            result,
            update.type,
            tuple(
                self.visit(value, ctx=ctx, dimension=True)
                for value in (
                    call.args[2].elements
                    if isinstance(call.args[2], expr.Tuple)
                    else (call.args[2], *(dim.i64_const(0) for _ in result.type.shape[1:]))
                )
            ),
        )
        self.emit_steps(call, "update", ctx)
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

    def fill(self, call: expr.Call, ctx) -> expr.Expr:
        """Declare *call*'s result and fill it with the call's value, else zero."""
        result = ctx.result(call)
        scalar = expr.Constant(
            getattr(call.target, "value", 0.0), type=types.TensorType.from_scalar(call.type.dtype)
        )
        self.emit_steps(call, None, ctx)
        ctx.builder.transfer(scalar, result, op=memory.Fill(), ctx=ctx)
        return result

    def visit_Zeros(self, call: expr.Call, operands, ctx) -> expr.Expr | transforms.View:
        return self.fill(call, ctx)

    def visit_FullLike(self, call: expr.Call, operands, ctx) -> expr.Expr | transforms.View:
        return self.fill(call, ctx)

    def default_visit_T(self, call: expr.Call, op, operands, ctx, emit=None) -> expr.Expr:
        """Issue *op* once in its owner region, or at each point of its execution mesh.

        *emit* adds one issue from the op and its operands in place of a bare
        ``Evaluate``.
        """
        emit = emit or (lambda issued_op, issued: ctx.builder.add(core.Evaluate(issued_op, issued)))
        issue = self.prepare_operands(call, op, operands, ctx)
        if analysis.required_execution_mesh(op) is None:
            self.issue_in_owner(call, op, issue, ctx, emit)
        else:
            locals_ = self.local_operands(op, issue, ctx)
            self.issue_at_points(call, op, issue, locals_, ctx, emit)
        return self.result_value(call, issue)

    def visit_T_CopyAsyncTensor(self, call: expr.Call, op, operands, ctx) -> expr.Expr:
        """Issue a tensor-map load as ``CopyAsyncTensorTx`` on the mbarrier words.

        The words are the ones ``SyncPlanMetadata.barriers`` places. A store
        keeps ``CopyAsyncTensor``. Either emitter completes the copy before the
        operation ends.
        """
        if not analysis.tensor_map_load(op, transforms.operand_types(call, op)):
            return self.default_visit_T(call, op, operands, ctx)
        plan = expr.get_metadata(self.context.function, analysis.SyncPlanMetadata)
        if plan is None or plan.barriers is None:
            raise transforms.LoweringError(f"{transforms.label(call)} has no mbarrier words to complete on")
        offset, count = plan.barriers
        type_ = sync.mbarrier_words(count)
        words = ctx.builder.bind(
            expr.Var(ctx.names.fresh("mbarriers"), type=type_), ctx.storage.view(offset, None, type_)
        )

        def emit(load, issued) -> None:
            tx = cuda_memory.CopyAsyncTensorTx(execution_mesh=load.execution_mesh)
            ctx.builder.add(core.Evaluate(tx, (*issued, words)))

        return self.default_visit_T(call, op, operands, ctx, emit)

    def result_value(self, call: expr.Call, issue: Issue) -> expr.Expr:
        """The value lowering gives *call*: the operand written in place, else its result."""
        result = analysis.result_param(issue.plan.params)
        if result.effect & expr.MemoryEffect.READ:
            return issue.read_values[result.name]
        if issue.written is None:
            raise transforms.LoweringError(f"{transforms.label(call)} produced no addressable result")
        return issue.written

    def prepare_operands(self, call: expr.Call, op, operands, ctx) -> Issue:
        """Project *op*, lower the operands it takes and stage each scalar in a register."""
        mesh = ctx.enclosing_mesh()
        plan, type_list, owner = ctx.issues.plan(call, op, mesh)
        params = plan.params
        supplied, produced = analysis.operand_roles(call, params)
        destination_operand = (
            call.args[params.index(produced[0])]
            if isinstance(call.target, hir.schedule.ScheduleOp) and produced
            else None
        )

        written = ctx.result(call) if produced else None
        read_values = {}
        for index, (param, operand) in enumerate(zip(supplied, call.args, strict=True)):
            value = written if operand is destination_operand else operands[index]
            if (
                not isinstance(call.target, hir.schedule.ScheduleOp)
                and operand.type != type_list[params.index(param)]
            ):
                value = ctx.storage.operand(value, type_list[params.index(param)], whole_tensor=True)
            if staged_scalar(value.type):
                scalar_type = replace(value.type, storage=expr.StorageKind.RMEM)
                scalar = expr.Var(ctx.names.fresh("scalar"), type=scalar_type)
                with ctx.builder.region():
                    ctx.builder.bind(
                        scalar, expr.Call(memory.AllocTensor(tensor_type=scalar_type), (), type=scalar_type)
                    )
                ctx.builder.add(core.Evaluate(memory.Fill(), (scalar, ctx.builder.value(value, ctx=ctx))))
                value = scalar
            read_values[param.name] = value
        issued_operands: list[tuple[str, expr.Expr | None]] = []
        for param in params:
            value = (
                read_values[param.name]
                if isinstance(call.target, hir.schedule.ScheduleOp) or param.effect & expr.MemoryEffect.READ
                else written
            )
            if value is None:
                raise transforms.LoweringError(f"{transforms.label(call)} has no storage for {param.name}")
            issued_operands.append((param.name, value))

        type_list = tuple(
            replace(type_, storage=expr.StorageKind.RMEM)
            if type_.shape == () and type_.storage is expr.StorageKind.UMAT
            else type_
            for type_ in type_list
        )
        issuers = analysis.issue_mesh(op, owner, mesh)
        return Issue(plan, type_list, tuple(issued_operands), read_values, written, mesh, issuers)

    def issue_in_owner(self, call: expr.Call, op, issue: Issue, ctx, emit) -> None:
        """Issue *op* once with its attributes through *emit*, in the region of its owner mesh."""
        if issue.enclosing is None:
            raise transforms.LoweringError(f"{transforms.label(call)} instruction issue has no lexical mesh")
        params, type_list = issue.plan.params, issue.types
        with ctx.builder.region(
            mesh=issue.issuers, binding=expr.Var(ctx.names.fresh("threads"), type=transforms.BINDING)
        ):
            issued = []
            for param, (role, value) in zip(params, issue.operands, strict=True):
                desired = type_list[params.index(param)]
                issued.append(
                    ctx.builder.value(
                        ctx.storage.operand(value, desired, stem=f"{role}_view"), ctx=ctx
                    )
                )
            attributes = {
                param.name: getattr(op, param.name)
                for param in expr.schema_of(type(op)).signature
                if param.kind == "attribute"
                and (
                    not param.has_default
                    or getattr(op, param.name) != param.default
                    or param.name == "fill"
                )
            }
            self.emit_steps(call, "issue", ctx)
            emit(type(op)(**attributes), tuple(issued))
            self.emit_steps(call, "after", ctx)

    def local_operands(self, op, issue: Issue, ctx) -> list:
        """Each operand as one point of *op*'s execution mesh sees it, in parameter order."""
        plan, type_list, issuers = issue.plan, issue.types, issue.issuers
        required = analysis.required_execution_mesh(op)
        for region in ctx.builder.blocks[:-1]:
            for key, value in region.views.items():
                ctx.builder.blocks[-1].views.setdefault(key, value)
        locals_ = []
        unit_coordinates = {}
        for (role, value), type_, rest in zip(issue.operands, type_list, plan.rests, strict=True):
            base = ctx.builder.value(ctx.storage.operand(value, type_, whole_tensor=True), ctx=ctx)
            binding_mesh = rest.mesh if isinstance(rest, types.ShardLayout) else issuers
            frame, dropped = utils.participant_frame(binding_mesh, required)
            source = types.shard_layout_of(base.type.layout)
            attrs = (
                list(source.attrs)
                if source is not None
                else [types.Broadcast()] * len(types.axis_keys(binding_mesh))
            )
            coords = [None] * len(attrs)
            for mesh_axis, (topology, name) in enumerate(types.axis_keys(binding_mesh)[:dropped]):
                attr = rest.attrs[mesh_axis]
                attrs[mesh_axis] = attr
                coordinate = 0
                if isinstance(attr, types.Split) and not ctx.target.storage_owners.private(
                    type_.storage, topology
                ):
                    key = id(binding_mesh), mesh_axis
                    if key not in unit_coordinates:
                        coord = expr.Call(
                            dim.MeshCoord(mesh=binding_mesh), (dim.i64_const(mesh_axis),), type=transforms.BINDING
                        )
                        unit_coordinates[key] = ctx.builder.bind(
                            expr.Var(ctx.names.fresh("unit"), type=transforms.BINDING), coord
                        )
                    coordinate = unit_coordinates[key]
                coords[mesh_axis] = coordinate
            address = source.layout if source is not None else base.type.layout
            local, offset = types.local_layout(
                types.ShardLayout(address, tuple(attrs), binding_mesh), base.type.shape, tuple(coords)
            )
            if any(isinstance(attr, types.Split) for attr in attrs[:dropped]):
                primitive = local.layout if isinstance(local, types.ShardLayout) else local
                outer = primitive.outer if isinstance(primitive, types.ComposedLayout) else primitive
                local_shape = tuple(prod(types.flatten(mode)) for mode in outer.shape)
                original = address.outer if isinstance(address, types.ComposedLayout) else address
                if not any(isinstance(mode, tuple) for mode in original.shape):
                    flat = types.flatten(outer)
                    primitive = (
                        replace(primitive, outer=flat)
                        if isinstance(primitive, types.ComposedLayout)
                        else flat
                    )
                if expr.static_dim_value(offset) != 0:
                    primitive = types.ComposedLayout(None, offset, primitive)
                local = (
                    replace(local, layout=primitive)
                    if source is not None and isinstance(local, types.ShardLayout)
                    else primitive
                )
                local_type = replace(base.type, shape=local_shape, layout=local)
                pointer_type = types.PointerType(base.type.dtype, base.type.storage)
                pointer = ctx.builder.bind(
                    expr.Var(ctx.names.fresh(f"{role}_pointer"), type=pointer_type),
                    expr.Call(memory.PtrOf(), (base,), type=pointer_type),
                )
                base = expr.Call(memory.TensorView(layout=local, shape=local_shape), (pointer,), type=local_type)
            locals_.append(base)
        return locals_

    def issue_at_points(self, call: expr.Call, op, issue: Issue, locals_, ctx, emit) -> None:
        """Issue *op* through *emit* at each ``IssuePlan.points`` point, in the issuing region."""
        with ctx.builder.region(
            mesh=issue.issuers, binding=expr.Var(ctx.names.fresh("threads"), type=transforms.BINDING)
        ):
            locals_ = [
                ctx.builder.bind(expr.Var(ctx.names.fresh(f"{role}_local"), type=base.type), base)
                if isinstance(base, expr.Call) and isinstance(base.target, memory.TensorView)
                else base
                for (role, _), base in zip(issue.operands, locals_, strict=True)
            ]
            for windows in ctx.issues.points(issue.plan):
                issued = []
                for (role, _), base, desired, starts in zip(
                    issue.operands, locals_, issue.plan.tiles, windows, strict=True
                ):
                    item = ctx.storage.issue_window(base, desired, starts, ctx.builder.loops)
                    issued.append(
                        ctx.builder.bind(expr.Var(ctx.names.fresh(f"{role}_view"), type=item.type), item)
                        if isinstance(item, expr.Call)
                        else item
                    )
                self.emit_steps(call, "issue", ctx)
                emit(op, tuple(issued))
            self.emit_steps(call, "after", ctx)


@target.bind(cuda_target.Target, transforms.ConvertHIRToTIR)
class CudaConvertHIRToTIR(transforms.ConvertHIRToTIR):
    """Lower scheduled HIR with the visitor that emits the sync steps."""

    visitor = CudaLowering
    analyses = ("memory", "sync")


__all__ = ["CudaConvertHIRToTIR", "CudaLowering"]
