"""Lower scheduled HIR to effect-form TIR.

The pass consumes the checked, inlined view produced by ``analyze(memory)``.
Memory metadata owns storage addresses; ``IssuePlan`` owns instruction issue
geometry.  This file only turns those two public facts into statements.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from tilefoundry.analysis import MemoryMetadata, analyze
from tilefoundry.analysis.iteration_scope import build_scopes, walk_scopes
from tilefoundry.ir.core import (
    BindingMetadata,
    Call,
    Constant,
    Expr,
    SourceSpanMetadata,
    Tuple,
    Var,
    get_metadata,
)
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import IssuePlan, ScheduleOp, issue_plan
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.zeros import Zeros
from tilefoundry.ir.tir.cast import Cast as TirCast
from tilefoundry.ir.tir.cuda.nn.mma import TiledMma
from tilefoundry.ir.tir.memory import AllocTensor, Copy, Fill, PtrOf, TensorView
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, For, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    PointerType,
    ShardLayout,
    Split,
    StorageKind,
    TensorType,
    TupleType,
)
from tilefoundry.ir.types.dim import DimFloorDiv, DimMod, DimSub, is_dim_op_call, simplify_dim
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.ir.types.utils import i64_const, static_dim_value
from tilefoundry.ir.visitor import ExprVisitor, collect_exprs
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.visitor_registry.access_relation import access_relation_registry
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext
from tilefoundry.visitor_registry.registries import typeinfer_registry

_INDEX = TensorType.umat_scalar()
_BINDING = TensorType.scalar(DType.i64, storage=StorageKind.RMEM)
_VIEW_OPS = (Slice, Reshape, Transpose)


class LoweringError(ValueError):
    """A checked HIR construct has no sound TIR representation."""


class _Cursor:
    """Append statements while making a let binding contain later entries."""

    def __init__(self) -> None:
        self._root: list = []
        self._block = self._root

    def add(self, stmt) -> None:
        self._block.append(("stmt", stmt))

    def bind(self, var: Var, value: Expr) -> Var:
        inner: list = []
        self._block.append(("let", var, value, inner))
        self._block = inner
        return var

    def build(self) -> Sequential:
        def build(entries: list) -> Sequential:
            body = []
            for entry in entries:
                if entry[0] == "stmt":
                    body.append(entry[1])
                else:
                    _, var, value, inner = entry
                    body.append(LetStmt(var, value, build(inner)))
            return Sequential(tuple(body))

        return build(self._root)


class Names:
    """Stable authored names plus deterministic generated-name suffixes."""

    def __init__(self, authored: Function) -> None:
        self.used = {param.name for param in authored.params}
        self.authored: dict[SourceSpanMetadata, str] = {}
        self.scopes: dict[SourceSpanMetadata, str] = {}
        for expr in collect_exprs(authored.body):
            span = get_metadata(expr, SourceSpanMetadata)
            binding = get_metadata(expr, BindingMetadata)
            if span is not None and binding is not None:
                self.authored[span] = binding.name
            if (
                isinstance(expr, MeshRegion)
                and expr is not authored.body
                and not (isinstance(expr.body, Call) and isinstance(expr.body.target, Zeros))
                and not any(
                    getattr(topology, "name", topology) == "cta"
                    for topology in expr.mesh.topologies
                )
                and span is not None
            ):
                self.scopes[span] = self.fresh("scope")

    def fresh(self, stem: str) -> str:
        if stem not in self.used:
            self.used.add(stem)
            return stem
        index = 1
        while f"{stem}_{index}" in self.used:
            index += 1
        result = f"{stem}_{index}"
        self.used.add(result)
        return result

    def binding(self, expr: Expr, fallback: str = "value") -> str:
        span = get_metadata(expr, SourceSpanMetadata)
        authored = self.authored.get(span) if span is not None else None
        return self.fresh(authored or fallback)

    def scope(self, expr: MeshRegion) -> str:
        span = get_metadata(expr, SourceSpanMetadata)
        if span is not None and span in self.scopes:
            return self.scopes[span]
        return self.fresh("scope")


def _frame(mesh: Mesh) -> Mesh:
    rank = len(tuple(flatten(flatten(mesh.layout).shape)))
    return Mesh(mesh.topologies, mesh.layout, tuple(f"d{index}" for index in range(rank)))


def _plain_layout(type_: TensorType):
    layout = type_.layout
    if isinstance(layout, ShardLayout):
        return layout.layout
    return layout


def _storage_type(type_: TensorType) -> TensorType:
    return replace(type_, layout=_plain_layout(type_))


def _with_frame(type_: TensorType, frame: Mesh) -> TensorType:
    layout = type_.layout
    if isinstance(layout, ShardLayout):
        layout = replace(layout, mesh=frame)
    return replace(type_, layout=layout)


def _label(call: Call) -> str:
    binding = get_metadata(call, BindingMetadata)
    return binding.name if binding is not None else type(call.target).__name__


def _needs_m2(plan: IssuePlan) -> bool:
    """Whether an atom issue needs grouping, repetition, or row widening."""
    return any(axis.is_group or axis.extent // axis.atom != 1 for axis in plan.axes)


@dataclass
class Lowering(ExprVisitor[Expr]):
    """Mechanical lowering of one analyzed HIR function."""

    module: Module
    authored: Function
    function: Function

    def __post_init__(self) -> None:
        ExprVisitor.__init__(self)
        self.names = Names(self.authored)
        self.memo: dict[int, Expr] = {}
        self.emitted: set[int] = set()
        self.logical: dict[int, TensorType] = {}
        self.staged: dict[int, Var] = {}
        self.current_mesh: list[Mesh] = []
        self.current_loops: list[tuple[LoopRegion, Var]] = []
        self.output: Var | None = None
        self.scratch: list[Var] = []
        self.output_seeded = False
        self.scopes = build_scopes(self.module, self.function)
        self.scope_for_call = {
            expr_id: scope
            for scope in walk_scopes(self.scopes)
            for expr_id, (call, _accesses) in scope.accesses["narrow"].items()
            if id(call) == expr_id
        }
        self.type_ctx = TypeInferContext(scope=FunctionScope(self.module, self.function))
        for param in self.function.params:
            physical = Var(param.name, type=param.type, is_const=param.is_const)
            self.memo[id(param)] = physical
            if isinstance(param.type, TensorType):
                self.logical[id(physical)] = param.type

    def run(self) -> PrimFunction:
        if self.function.body is None:
            raise LoweringError(f"{self.function.name!r} has no body to lower")
        if not isinstance(self.function.return_type, TensorType):
            raise LoweringError("scheduled lowering currently requires one tensor result")

        result_type = self.function.return_type
        out_type = TensorType(tuple(result_type.shape), result_type.dtype, None, StorageKind.GMEM)
        self.output = Var(self.names.fresh("out"), type=out_type)
        self._preflight_m2()
        body = self.function.body
        if isinstance(body, MeshRegion):
            root_mesh = _frame(body.mesh)
            inner = _Cursor()
            self._prepare_buffers(inner)
            self.current_mesh.append(body.mesh)
            self._bind_region_args(body, inner)
            result = self.lower(body.body, inner)
            self.current_mesh.pop()
            self._finish(result, inner)
            statements = Sequential(
                (MeshScope(root_mesh, Var(self.names.fresh("cta"), type=_BINDING), inner.build()),)
            )
        else:
            inner = _Cursor()
            self._prepare_buffers(inner)
            result = self.lower(body, inner)
            self._finish(result, inner)
            statements = inner.build()

        params = tuple(self.memo[id(param)] for param in self.function.params)
        return PrimFunction(
            self.function.name,
            (*params, *self.scratch, self.output),
            statements,
            output_count=1,
            target=self.module.resolve_target(),
        )

    def _preflight_m2(self) -> None:
        """Refuse later issue shapes before unrelated HIR obscures that boundary."""
        for expr in collect_exprs(self.function.body):
            if not (
                isinstance(expr, Call)
                and isinstance(expr.target, ScheduleOp)
                and isinstance(expr.target.op, TiledMma)
            ):
                continue
            try:
                plan = issue_plan(expr, self.type_ctx)
            except (TypeError, ValueError):
                continue
            if _needs_m2(plan):
                raise LoweringError(
                    f"{_label(expr)} needs grouped, repeated, or row-wise atom issue; "
                    "M2 not implemented"
                )

    def _prepare_buffers(self, cursor: _Cursor) -> None:
        """Declare material buffers at their function-lifetime owning scope."""
        for expr in collect_exprs(self.function.body):
            if not isinstance(expr, Call):
                continue
            if isinstance(expr.target, (Zeros, HirCast)):
                self._declare(expr, expr.type, 1, cursor)
                continue
            if not isinstance(expr.target, ScheduleOp):
                continue
            params = self._instruction_params(expr.target.op)
            writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
            produced = tuple(param for param in writes if not param.effect & MemoryEffect.READ)
            if produced:
                self._declare(expr, expr.type, expr.target.buffers, cursor)

    def _declare(self, expr: Call, type_: object, buffers: int, cursor: _Cursor) -> Expr:
        known = self.memo.get(id(expr))
        if known is not None:
            return known
        if not isinstance(type_, TensorType):
            raise LoweringError(f"{_label(expr)} has non-tensor material result {type_!r}")
        storage_type = _storage_type(type_)
        stem = self.names.binding(expr)
        if storage_type.storage is StorageKind.RMEM:
            var = Var(stem, type=storage_type)
            self.memo[id(expr)] = cursor.bind(
                var,
                Call(AllocTensor(tensor_type=storage_type), (), type=storage_type),
            )
            self.logical[id(var)] = type_
            return var
        if storage_type.storage not in (StorageKind.SMEM, StorageKind.GMEM):
            raise LoweringError(
                f"{_label(expr)} has addressable result in unsupported storage "
                f"{storage_type.storage}"
            )
        memory = get_metadata(expr, MemoryMetadata)
        offsets = () if memory is None else memory.offsets
        if not offsets:
            raise LoweringError(
                f"{_label(expr)} has an addressable {storage_type.storage} result but no offsets"
            )
        if len(offsets) != buffers:
            raise LoweringError(
                f"{_label(expr)} declares {buffers} buffer(s), but memory analysis placed "
                f"{len(offsets)}"
            )
        if storage_type.storage is StorageKind.GMEM:
            if buffers != 1:
                raise LoweringError(f"{_label(expr)} uses staged internal gmem")
            elements = prod(tuple(storage_type.shape))
            scratch_type = TensorType((elements,), storage_type.dtype, None, StorageKind.GMEM)
            scratch = Var(self.names.fresh("scratch"), type=scratch_type)
            self.scratch.append(scratch)
            desired = replace(
                storage_type,
                layout=Layout(
                    tuple(storage_type.shape),
                    tuple(compact_row_major(tuple(storage_type.shape))),
                ),
            )
            var = self._window(
                scratch,
                (i64_const(0),),
                (elements,),
                desired,
                cursor,
                stem,
            )
            self.memo[id(expr)] = var
            self.logical[id(var)] = type_
            return var
        fields = tuple(self._literal_view(storage_type, offset) for offset in offsets)
        if len(fields) == 1:
            var = Var(stem, type=storage_type)
            self.memo[id(expr)] = cursor.bind(var, fields[0])
            self.logical[id(var)] = type_
            return var
        tuple_type = TupleType(tuple(field.type for field in fields))
        stages = Var(self.names.fresh(f"{stem}_stages"), type=tuple_type)
        cursor.bind(stages, Tuple(fields, type=tuple_type))
        self.logical[id(stages)] = type_
        self.staged[id(expr)] = stages
        selected = self._stage(expr, stages)
        self.memo[id(expr)] = selected
        return selected

    @staticmethod
    def _literal_view(type_: TensorType, offset: int) -> Call:
        pointer = Constant(offset, type=PointerType(type_.dtype, type_.storage))
        return Call(
            TensorView(
                dtype=type_.dtype.name,
                storage=type_.storage,
                layout=type_.layout,
                shape=tuple(type_.shape),
            ),
            (pointer,),
            type=type_,
        )

    def _stage(self, expr: Call, stages: Var) -> Expr:
        if not self.current_loops:
            index = i64_const(0)
        else:
            loop, induction = self.current_loops[-1]
            start = static_dim_value(loop.start)
            step = static_dim_value(loop.step)
            if step is None:
                raise LoweringError(
                    f"{_label(expr)} staged selection needs a literal enclosing loop step"
                )
            moved = induction if start == 0 else simplify_dim(DimSub, (induction, loop.start))
            walked = moved if step == 1 else simplify_dim(DimFloorDiv, (moved, step))
            walked = simplify_dim(DimMod, (walked, len(stages.type.fields)))
            index = walked
        field = stages.type.fields[0]
        return Call(TupleGetItem(), (stages, index), type=field)

    def _known(self, expr: Expr) -> Expr:
        known = self.memo.get(id(expr))
        return expr if known is None else known

    def _bind_region_args(self, region: MeshRegion, cursor: _Cursor) -> None:
        values = tuple(self.lower(arg, cursor) for arg in region.args)
        for param, value in zip(region.params, values, strict=True):
            self.memo[id(param)] = value

    def lower(self, expr: Expr, cursor: _Cursor) -> Expr:
        known = self.memo.get(id(expr))
        material_call = isinstance(expr, Call) and isinstance(
            expr.target, (Zeros, HirCast, ScheduleOp)
        )
        if known is not None and (not material_call or id(expr) in self.emitted):
            if id(expr) in self.staged:
                return self._stage(expr, self.staged[id(expr)])
            return known
        if isinstance(expr, Constant):
            return expr
        if isinstance(expr, Var):
            raise LoweringError(f"unbound HIR value {expr.name!r} reached lowering")
        if isinstance(expr, Tuple):
            elements = tuple(self.lower(element, cursor) for element in expr.elements)
            result = Tuple(elements, type=TupleType(tuple(element.type for element in elements)))
            self.memo[id(expr)] = result
            return result
        if isinstance(expr, MeshRegion):
            return self._lower_mesh(expr, cursor)
        if isinstance(expr, LoopRegion):
            return self._lower_loop(expr, cursor)
        if not isinstance(expr, Call):
            raise LoweringError(f"unknown HIR value {type(expr).__name__}")
        return self._lower_call(expr, cursor)

    def _lower_mesh(self, region: MeshRegion, cursor: _Cursor) -> Expr:
        self._bind_region_args(region, cursor)
        if isinstance(region.body, Call) and isinstance(region.body.target, Zeros):
            self.current_mesh.append(region.mesh)
            result = self.lower(region.body, cursor)
            self.current_mesh.pop()
            self.memo[id(region)] = result
            return result
        inner = _Cursor()
        self.current_mesh.append(region.mesh)
        result = self.lower(region.body, inner)
        self.current_mesh.pop()
        built = inner.build()
        if built.body:
            cursor.add(
                MeshScope(
                    _frame(region.mesh),
                    Var(self.names.scope(region), type=_BINDING),
                    built,
                )
            )
        self.memo[id(region)] = result
        return result

    def _lower_loop(self, loop: LoopRegion, cursor: _Cursor) -> Expr:
        init = tuple(self.lower(value, cursor) for value in loop.init_args)
        if len(init) != len(loop.carried_args):
            raise LoweringError("loop carry arity changed during lowering")
        for var, value in zip(loop.carried_args, init, strict=True):
            self.memo[id(var)] = value
        induction = Var(loop.induction_var.name, type=_INDEX)
        self.memo[id(loop.induction_var)] = induction
        inner = _Cursor()
        self.current_loops.append((loop, induction))
        self.lower(loop.body, inner)
        yielded = tuple(self.lower(value, inner) for value in loop.yield_values)
        self.current_loops.pop()
        for carried, value in zip(init, yielded, strict=True):
            if carried is not value:
                self._emit_copy(value, carried, inner)
        cursor.add(
            For(
                induction,
                self._dim(loop.start),
                self._dim(loop.extent),
                self._dim(loop.step),
                inner.build(),
            )
        )
        result: Expr
        if len(init) == 1:
            result = init[0]
        else:
            result = Tuple(init, type=TupleType(tuple(value.type for value in init)))
        self.memo[id(loop)] = result
        return result

    def _dim(self, value) -> Expr:
        number = static_dim_value(value)
        if number is not None:
            return i64_const(number)
        if isinstance(value, Var):
            known = self.memo.get(id(value))
            if known is None:
                raise LoweringError(f"dimension uses unbound value {value.name!r}")
            return known
        if isinstance(value, Call) and isinstance(value.target, MeshCoord):
            return Call(
                replace(value.target, mesh=_frame(value.target.mesh)), value.args, type=value.type
            )
        if isinstance(value, Call) and is_dim_op_call(value):
            return Call(value.target, tuple(self._dim(arg) for arg in value.args), type=value.type)
        if isinstance(value, Expr):
            return self._known(value)
        raise LoweringError(f"dimension {value!r} is not lowerable")

    def _lower_call(self, call: Call, cursor: _Cursor) -> Expr:
        target = call.target
        if is_dim_op_call(call) or isinstance(target, MeshCoord):
            result = self._dim(call)
        elif isinstance(target, TupleGetItem):
            source = self.lower(call.args[0], cursor)
            index = self._dim(call.args[1])
            if isinstance(source, Tuple) and isinstance(index, Constant):
                result = source.elements[index.value]
            else:
                result = Call(target, (source, index), type=call.type)
        elif isinstance(target, Slice):
            result = self._lower_slice(call, cursor)
        elif isinstance(target, (Reshape, Transpose)):
            result = self._lower_view(call, cursor)
        elif isinstance(target, Zeros):
            result = self.memo[id(call)]
            self._emit_fill(result, call.type, cursor)
        elif isinstance(target, HirCast):
            result = self.memo[id(call)]
            source = self.lower(call.args[0], cursor)
            self._emit_cast(call, source, result, call.type, cursor)
        elif isinstance(target, InsertSlice):
            result = self._lower_insert(call, cursor)
        elif isinstance(target, ScheduleOp):
            result = self._lower_schedule(call, cursor)
        else:
            raise LoweringError(f"{_label(call)} uses unknown HIR call {type(target).__name__}")
        self.memo[id(call)] = result
        self.emitted.add(id(call))
        return result

    def _lower_slice(self, call: Call, cursor: _Cursor) -> Expr:
        base = self.lower(call.args[0], cursor)
        starts_arg = call.args[1]
        starts = (
            tuple(self._dim(value) for value in starts_arg.elements)
            if isinstance(starts_arg, Tuple)
            else (self._dim(starts_arg),)
        )
        if any(stride != 1 for stride in call.target.strides):
            raise LoweringError(f"{_label(call)} has a strided Slice, which is not contiguous")
        layout = _plain_layout(call.type)
        if layout is None:
            source_layout = _plain_layout(call.args[0].type)
            strides = (
                tuple(source_layout.strides)
                if isinstance(source_layout, Layout) and source_layout.strides is not None
                else tuple(compact_row_major(tuple(call.args[0].type.shape)))
            )
            layout = Layout(tuple(call.type.shape), strides)
        return self._window(
            base,
            starts,
            tuple(call.target.sizes),
            replace(call.type, layout=layout),
            cursor,
            "tile",
        )

    def _lower_view(self, call: Call, cursor: _Cursor) -> Expr:
        source = self.lower(call.args[0], cursor)
        desired = call.type
        if isinstance(call.target, Transpose):
            layout = _plain_layout(desired)
        else:
            layout = _plain_layout(desired)
        desired = replace(desired, layout=layout)
        starts = tuple(i64_const(0) for _ in source.type.shape)
        return self._window(
            source,
            starts,
            tuple(source.type.shape),
            desired,
            cursor,
            "tile",
        )

    def _lower_insert(self, call: Call, cursor: _Cursor) -> Expr:
        assert self.output is not None
        destination = call.args[0]
        if isinstance(destination, Call) and isinstance(destination.target, Zeros):
            if not self.output_seeded:
                self._emit_fill(self.output, self.output.type, cursor)
                self.output_seeded = True
        else:
            self.lower(destination, cursor)
        update = self.lower(call.args[1], cursor)
        offsets = call.args[2]
        starts = (
            tuple(self._dim(value) for value in offsets.elements)
            if isinstance(offsets, Tuple)
            else (self._dim(offsets), *(i64_const(0) for _ in self.output.type.shape[1:]))
        )
        target = self._window(
            self.output,
            starts,
            tuple(update.type.shape),
            replace(update.type, storage=StorageKind.GMEM, layout=None),
            cursor,
            "window",
        )
        self._emit_copy(update, target, cursor)
        return self.output

    def _lower_schedule(self, call: Call, cursor: _Cursor) -> Expr:
        if access_relation_registry.lookup(type(call.target.op)) is None:
            raise LoweringError(
                f"{_label(call)} selects {type(call.target.op).__name__}, which has no "
                "registered access relation"
            )
        scope = self.scope_for_call.get(id(call))
        mesh = scope.enclosing_mesh() if scope is not None else None
        self.type_ctx.current_mesh = mesh
        try:
            plan = issue_plan(call, self.type_ctx)
        except (TypeError, ValueError) as error:
            raise LoweringError(f"{_label(call)} cannot derive an issue plan: {error}") from error

        params = self._instruction_params(call.target.op)
        reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
        read_values = {
            param.name: self.lower(value, cursor)
            for param, value in zip(reads, call.args, strict=True)
        }
        written = (
            self._stage(call, self.staged[id(call)])
            if id(call) in self.staged
            else self.memo.get(id(call))
        )
        operands = []
        for param, desired in zip(params, plan.operand_types, strict=True):
            if param.effect & MemoryEffect.READ:
                value = read_values[param.name]
            else:
                if written is None:
                    raise LoweringError(f"{_label(call)} has no storage for {param.name}")
                value = written
            operands.append((param.name, value, desired))

        if isinstance(call.target.op, TiledMma):
            if _needs_m2(plan):
                raise LoweringError(
                    f"{_label(call)} needs grouped, repeated, or row-wise atom issue; "
                    "M2 not implemented"
                )
            frame = self._issuer_frame(tuple(desired for _, _, desired in operands), mesh)
            counters = {
                axis.name: Var(self.names.fresh(f"o_{axis.name}"), type=_INDEX)
                for axis in plan.axes
            }
            issue = _Cursor()
            issued_args = []
            for (role, value, desired), axes in zip(operands, plan.operand_axes, strict=True):
                if any(axis is None for axis in axes):
                    raise LoweringError(
                        f"{_label(call)} operand {role!r} is not a projection of work axes"
                    )
                starts = tuple(counters[axis] for axis in axes)
                issued_args.append(
                    self._window(
                        value,
                        starts,
                        tuple(desired.shape),
                        _with_frame(desired, frame),
                        issue,
                        f"{role}_view",
                    )
                )
            atom = call.target.op.atom.on(frame)
            issue.add(Evaluate(type(call.target.op)(atom=atom), tuple(issued_args)))
            statement = issue.build()
            for axis_index in reversed(plan.order):
                axis = plan.axes[axis_index]
                statement = Sequential(
                    (
                        For(
                            counters[axis.name],
                            i64_const(0),
                            i64_const(axis.extent),
                            i64_const(axis.atom),
                            statement,
                        ),
                    )
                )
            cursor.add(MeshScope(frame, Var(self.names.fresh("threads"), type=_BINDING), statement))
        else:
            frame = self._issuer_frame(tuple(desired for _, _, desired in operands), mesh)
            issue = _Cursor()
            issued_args = tuple(
                self._whole_operand(value, desired, frame, issue, f"{role}_frame")
                for role, value, desired in operands
            )
            issue.add(Evaluate(call.target.op, issued_args))
            cursor.add(
                MeshScope(
                    frame,
                    Var(self.names.fresh("threads"), type=_BINDING),
                    issue.build(),
                )
            )
        result_param = next(param for param in params if param.effect & MemoryEffect.WRITE)
        if result_param.effect & MemoryEffect.READ:
            return read_values[result_param.name]
        if written is None:
            raise LoweringError(f"{_label(call)} produced no addressable result")
        return written

    @staticmethod
    def _instruction_params(op) -> tuple:
        schema = getattr(type(op), "_op_schema", None)
        if schema is None:
            raise LoweringError(f"{type(op).__name__} is not a registered operation")
        params = tuple(param for param in schema.signature if param.kind == "input")
        if any(param.effect is None for param in params):
            raise LoweringError(f"{type(op).__name__} does not declare every operand memory effect")
        return params

    def _issuer_frame(self, types: tuple[TensorType, ...], lexical: Mesh | None) -> Mesh:
        for type_ in types:
            layout = type_.layout
            if isinstance(layout, ShardLayout):
                return _frame(layout.mesh)
        if lexical is None:
            raise LoweringError("instruction issue has no lexical or sharded physical mesh")
        return _frame(lexical)

    def _whole_operand(
        self,
        value: Expr,
        desired: TensorType,
        frame: Mesh,
        cursor: _Cursor,
        stem: str,
    ) -> Expr:
        desired = _with_frame(desired, frame)
        if value.type == desired or (
            not isinstance(desired.layout, ShardLayout)
            and isinstance(value.type, TensorType)
            and tuple(value.type.shape) == tuple(desired.shape)
            and value.type.dtype == desired.dtype
            and value.type.storage is desired.storage
        ):
            return value
        return self._window(
            value,
            tuple(i64_const(0) for _ in value.type.shape),
            tuple(value.type.shape),
            desired,
            cursor,
            stem,
        )

    def _window(
        self,
        base: Expr,
        starts: tuple[Expr, ...],
        sizes: tuple,
        desired: TensorType,
        cursor: _Cursor,
        stem: str,
    ) -> Var:
        held = base.type
        if not isinstance(held, TensorType):
            raise LoweringError(f"a tensor window cannot use {held!r} as its base")
        if len(starts) != len(held.shape):
            raise LoweringError(
                f"a rank-{len(held.shape)} buffer is windowed by {len(starts)} starts"
            )
        if prod(tuple(sizes)) != prod(tuple(desired.shape)):
            raise LoweringError(
                f"a window of {tuple(sizes)} cannot be viewed as {tuple(desired.shape)}"
            )
        if desired.layout is None:
            desired = replace(
                desired,
                layout=Layout(
                    tuple(desired.shape),
                    tuple(compact_row_major(tuple(desired.shape))),
                ),
            )
        keys = Tuple(starts, type=TupleType(tuple(start.type for start in starts)))
        cut_type = TensorType(tuple(sizes), held.dtype, None, held.storage)
        cut = Call(
            Slice(sizes=tuple(sizes), strides=(1,) * len(sizes)),
            (base, keys),
            type=cut_type,
        )
        inferred = typeinfer_registry.lookup(Slice)(cut, TypeInferContext())
        cut.type = getattr(inferred, "type", inferred)
        pointer = Call(PtrOf(), (cut,), type=PointerType(held.dtype, held.storage))
        view = Call(
            TensorView(layout=desired.layout, shape=tuple(desired.shape)),
            (pointer,),
            type=desired,
        )
        return cursor.bind(Var(self.names.fresh(stem), type=desired), view)

    def _holder_mesh(self, type_: TensorType) -> Mesh | None:
        layout = type_.layout
        return layout.mesh if isinstance(layout, ShardLayout) else None

    def _emit_fill(self, target: Expr, logical_type: TensorType, cursor: _Cursor) -> None:
        mesh = self._holder_mesh(logical_type)
        zero = Constant(0.0, type=TensorType.scalar(logical_type.dtype))
        statement = Sequential((Evaluate(Fill(), (target, zero)),))
        if mesh is None:
            cursor.add(statement.body[0])
        else:
            cursor.add(
                MeshScope(
                    _frame(mesh),
                    Var(self.names.fresh("threads"), type=_BINDING),
                    statement,
                )
            )

    def _emit_cast(
        self,
        call: Call,
        source: Expr,
        target: Expr,
        logical_type: TensorType,
        cursor: _Cursor,
    ) -> None:
        if logical_type.storage is StorageKind.GMEM:
            self._emit_gmem_cast(call, source, target, logical_type, cursor)
            return
        mesh = self._holder_mesh(logical_type)
        if mesh is None:
            raise LoweringError("register cast result has no holder mesh")
        frame = _frame(mesh)
        issue = _Cursor()
        written = self._whole_operand(target, logical_type, frame, issue, "value_view")
        issue.add(Evaluate(TirCast(), (source, written)))
        cursor.add(
            MeshScope(
                frame,
                Var(self.names.fresh("threads"), type=_BINDING),
                issue.build(),
            )
        )

    def _emit_gmem_cast(
        self,
        call: Call,
        source: Expr,
        target: Expr,
        logical_type: TensorType,
        cursor: _Cursor,
    ) -> None:
        scope = self.scope_for_call.get(id(call))
        mesh = scope.enclosing_mesh() if scope is not None else None
        if mesh is None:
            raise LoweringError(f"{_label(call)} gmem cast has no issuing mesh")
        frame = _frame(mesh)
        mesh_shape = tuple(flatten(flatten(mesh.layout).shape))
        elements = prod(tuple(logical_type.shape))
        participants = prod(mesh_shape)
        if elements % participants:
            raise LoweringError(
                f"{_label(call)} casts {elements} elements over {participants} participants"
            )
        local = elements // participants
        layout_shape = (*mesh_shape, local)
        distributed = ShardLayout(
            Layout(layout_shape, tuple(compact_row_major(layout_shape))),
            tuple(Split(index) for index in range(len(mesh_shape))),
            frame,
        )
        src_logical = TensorType(
            tuple(source.type.shape), source.type.dtype, distributed, StorageKind.RMEM
        )
        dst_logical = TensorType(
            tuple(logical_type.shape), logical_type.dtype, distributed, StorageKind.RMEM
        )
        src_type = _storage_type(src_logical)
        dst_type = _storage_type(dst_logical)
        src_stage = cursor.bind(
            Var(self.names.fresh("stage_f32"), type=src_type),
            Call(AllocTensor(tensor_type=src_type), (), type=src_type),
        )
        dst_stage = cursor.bind(
            Var(self.names.fresh("stage_bf16"), type=dst_type),
            Call(AllocTensor(tensor_type=dst_type), (), type=dst_type),
        )
        self.logical[id(src_stage)] = src_logical
        self.logical[id(dst_stage)] = dst_logical
        issue = _Cursor()
        read = self._whole_operand(
            source,
            replace(src_logical, storage=StorageKind.GMEM),
            frame,
            issue,
            f"{getattr(source, 'name', 'source')}_view",
        )
        written = self._whole_operand(
            target,
            replace(dst_logical, storage=StorageKind.GMEM),
            frame,
            issue,
            "value_view",
        )
        issue.add(Evaluate(Copy(), (read, src_stage)))
        issue.add(Evaluate(TirCast(), (src_stage, dst_stage)))
        issue.add(Evaluate(Copy(), (dst_stage, written)))
        cursor.add(
            MeshScope(
                frame,
                Var(self.names.fresh("threads"), type=_BINDING),
                issue.build(),
            )
        )

    def _emit_copy(self, source: Expr, target: Expr, cursor: _Cursor) -> None:
        source_type = self.logical.get(id(source), source.type)
        if not isinstance(source_type, TensorType) or not isinstance(target.type, TensorType):
            raise LoweringError("copy lowering requires tensor operands")
        mesh = self._holder_mesh(source_type) or self._holder_mesh(target.type)
        if mesh is None:
            cursor.add(Evaluate(Copy(), (source, target)))
            return
        frame = _frame(mesh)
        issue = _Cursor()
        read = self._whole_operand(
            source,
            source_type,
            frame,
            issue,
            f"{getattr(source, 'name', 'value')}_view",
        )
        issue.add(Evaluate(Copy(), (read, target)))
        cursor.add(
            MeshScope(
                frame,
                Var(self.names.fresh("threads"), type=_BINDING),
                issue.build(),
            )
        )

    def _finish(self, result: Expr, cursor: _Cursor) -> None:
        assert self.output is not None
        if result is not self.output:
            self._emit_copy(result, self.output, cursor)


@dataclass
class ConvertHIRToTIR(ModulePass):
    """Replace one selected HIR entry with its scheduled TIR function."""

    entry: str | None = None

    name = "convert-hir-to-tir"
    requires = ()

    def run(self, module: Module) -> Module:
        authored = module.entry_function() if self.entry is None else module.lookup(self.entry)
        if not isinstance(authored, Function):
            raise LoweringError(f"{getattr(authored, 'name', self.entry)!r} is not an HIR function")
        result = analyze(module, authored, analysis=("memory",))
        lowered = Lowering(result.module, authored, result.function).run()
        functions = tuple(
            lowered if function is authored else function for function in module.functions
        )
        return replace(module, functions=functions)


__all__ = ["ConvertHIRToTIR", "LoweringError"]
