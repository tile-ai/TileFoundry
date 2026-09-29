"""Lower scheduled HIR to effect-form TIR.

The pass consumes the checked, inlined view produced by ``analyze(memory)``.
Memory metadata owns storage addresses; registered access relations own
instruction issue geometry. This file only turns those two facts into statements.
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
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.zeros import Zeros
from tilefoundry.ir.tir.cast import Cast as TirCast
from tilefoundry.ir.tir.cuda.nn.mma import operand_axes
from tilefoundry.ir.tir.memory import AllocTensor, Copy, Fill, PtrOf, TensorView
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, For, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import (
    ComposedLayout,
    DType,
    Layout,
    Mesh,
    PointerType,
    ShardLayout,
    StorageKind,
    Swizzle,
    TensorType,
    TupleType,
)
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMod,
    DimMul,
    DimSub,
    is_dim_op_call,
    simplify_dim,
)
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.ir.types.utils import i64_const, issue_frames, nonunit_mesh, static_dim_value
from tilefoundry.ir.visitor import ExprVisitor, StmtMutator, expr_children
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.visitor_registry.access_relation import access_relation_registry
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext
from tilefoundry.visitor_registry.registries import typeinfer_registry

_INDEX = TensorType.umat_scalar()
_BINDING = TensorType.scalar(DType.i64, storage=StorageKind.RMEM)
_VIEW_OPS = (Slice, Reshape, Transpose)
_DIM_BINARY = {
    BinaryKind.ADD: DimAdd,
    BinaryKind.SUB: DimSub,
    BinaryKind.MUL: DimMul,
    BinaryKind.FLOOR_DIV: DimFloorDiv,
    BinaryKind.MOD: DimMod,
}


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


class _MeshScopeCoalescer(StmtMutator):
    """Merge adjacent issues by the same physical participants."""

    def visit_Sequential(self, stmt: Sequential) -> Sequential:
        body = []
        for child in stmt.body:
            child = self.visit(child)
            if (
                body
                and isinstance(body[-1], MeshScope)
                and isinstance(child, MeshScope)
                and body[-1].mesh == child.mesh
            ):
                previous = body[-1]
                body[-1] = MeshScope(
                    previous.mesh,
                    previous.binding,
                    Sequential((*previous.body.body, *child.body.body)),
                )
            else:
                body.append(child)
        return Sequential(tuple(body))


class Names:
    """Stable authored names plus deterministic generated-name suffixes."""

    def __init__(self, authored: Function) -> None:
        self.used = {param.name for param in authored.params}

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

    def binding(self, authored: Expr | None, fallback: str = "value") -> str:
        binding = get_metadata(authored, BindingMetadata) if authored is not None else None
        return self.fresh(binding.name if binding is not None else fallback)

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
        self.staged: dict[int, tuple[Var, LoopRegion | None]] = {}
        self.frames: list[Mesh] = []
        self.output: Var | None = None
        self.scratch: list[Var] = []
        self.owner_cursors: dict[int, _Cursor] = {}
        self.bindings: dict[int, Expr] = {}
        self.output_windows: dict[int, Call] = {}
        self.output_seed: Call | None = None
        self.output_initialized = False
        self.authored_values: dict[int, Expr] = {}
        if self.function.body is not None and self.authored.body is not None:
            self.authored_values[id(self.function.body)] = self.authored.body
        self.scopes = build_scopes(self.module, self.function)
        self.scope_for_call = {
            expr_id: scope
            for scope in walk_scopes(self.scopes)
            for expr_id, (call, _relations) in scope.relations.items()
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
        body = self.function.body
        inner = _Cursor()
        self.owner_cursors[id(self.function)] = inner
        if isinstance(body, MeshRegion):
            root_mesh = self._physical_frame(body.mesh)
            self.owner_cursors[id(body)] = inner
            self._pair_authored(body)
            self._bind_region_args(body, inner)
            result = self.lower(body.body, inner)
        else:
            result = self.lower(body, inner)
        self._finish(result, inner)
        statements = inner.build()
        if isinstance(body, MeshRegion):
            statements = Sequential(
                (MeshScope(root_mesh, Var(self.names.fresh("cta"), type=_BINDING), statements),)
            )

        statements = _MeshScopeCoalescer().visit(statements)

        params = tuple(self.memo[id(param)] for param in self.function.params)
        return PrimFunction(
            self.function.name,
            (*params, *self.scratch, self.output),
            statements,
            output_count=1,
            target=self.module.resolve_target(),
        )

    def _staging_owner(self, expr: Call) -> Expr:
        loop = self._staging_loop(expr)
        if loop is None:
            return self.function
        scope = self.scope_for_call.get(id(expr))
        while scope is not None and scope.owner is not loop:
            scope = scope.parent
        return self.function if scope is None or scope.parent is None else scope.parent.owner

    def _staging_loop(self, expr: Call) -> LoopRegion | None:
        scope = self.scope_for_call.get(id(expr))
        while scope is not None:
            if isinstance(scope.owner, LoopRegion):
                return scope.owner
            scope = scope.parent
        return None

    def _ensure_output_seed(self, seed: Call) -> Expr:
        """Bind one reached gmem zero to the function output at first use."""
        assert self.output is not None
        if self.output_seed is not None and self.output_seed is not seed:
            raise LoweringError("scheduled lowering cannot choose among multiple gmem seeds")
        self.output_seed = seed
        self.memo[id(seed)] = self.output
        self.logical[id(self.output)] = seed.type
        self.emitted.add(id(seed))
        return self.output

    def _material_root(self, value: Expr) -> Expr:
        seen: set[int] = set()
        while id(value) not in seen:
            seen.add(id(value))
            bound = self.bindings.get(id(value))
            if bound is not None:
                value = bound
                continue
            if isinstance(value, Call) and isinstance(value.target, _VIEW_OPS):
                value = value.args[0]
                continue
            if isinstance(value, MeshRegion):
                value = value.body
                continue
            if isinstance(value, LoopRegion) and len(value.init_args) == 1:
                value = value.init_args[0]
                continue
            return value
        raise LoweringError("output seed resolution found a cyclic value")

    def _declare(
        self,
        expr: Call,
        type_: object,
        buffers: int,
        cursor: _Cursor,
    ) -> Expr:
        known = self.memo.get(id(expr))
        if known is not None:
            return known
        if not isinstance(type_, TensorType):
            raise LoweringError(f"{_label(expr)} has non-tensor material result {type_!r}")
        storage_type = _storage_type(type_)
        stem = (
            self.names.fresh("value")
            if isinstance(expr.target, HirCast)
            else self.names.binding(self.authored_values.get(id(expr)))
        )
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
        self.staged[id(expr)] = (stages, self._staging_loop(expr))
        selected = self._stage(expr)
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

    def _stage(self, expr: Call) -> Expr:
        stages, loop = self.staged[id(expr)]
        if loop is None:
            index = i64_const(0)
        else:
            induction = self.memo.get(id(loop.induction_var))
            if not isinstance(induction, Var):
                raise LoweringError(f"{_label(expr)} staged selection has no owning induction")
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
        self.bindings.update(zip(map(id, region.params), region.args, strict=True))
        values = tuple(self.lower(arg, cursor) for arg in region.args)
        for param, value in zip(region.params, values, strict=True):
            self.memo[id(param)] = value

    def _pair_authored(self, expr: Expr) -> None:
        authored = self.authored_values.get(id(expr))
        if authored is None:
            return
        actual_children = expr_children(expr)
        authored_children = expr_children(authored)
        if len(actual_children) != len(authored_children):
            return
        for actual, source in zip(actual_children, authored_children, strict=True):
            actual_span = get_metadata(actual, SourceSpanMetadata)
            source_span = get_metadata(source, SourceSpanMetadata)
            if type(actual) is type(source) and (
                actual_span is None or source_span is None or actual_span == source_span
            ):
                self.authored_values[id(actual)] = source

    def lower(self, expr: Expr, cursor: _Cursor) -> Expr:
        self._pair_authored(expr)
        known = self.memo.get(id(expr))
        material_call = isinstance(expr, Call) and isinstance(
            expr.target, (Zeros, HirCast, ScheduleOp)
        )
        if known is not None and (not material_call or id(expr) in self.emitted):
            if id(expr) in self.staged:
                return self._stage(expr)
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
            result = self.lower(region.body, cursor)
            self.memo[id(region)] = result
            return result
        inner = _Cursor()
        self.owner_cursors[id(region)] = inner
        result = self.lower(region.body, inner)
        built = inner.build()
        if built.body:
            cursor.add(
                MeshScope(
                    self._physical_frame(region.mesh),
                    Var(self.names.fresh("scope"), type=_BINDING),
                    built,
                )
            )
        self.memo[id(region)] = result
        return result

    def _lower_loop(self, loop: LoopRegion, cursor: _Cursor) -> Expr:
        self.bindings.update(zip(map(id, loop.carried_args), loop.init_args, strict=True))
        init = tuple(self.lower(value, cursor) for value in loop.init_args)
        if len(init) != len(loop.carried_args):
            raise LoweringError("loop carry arity changed during lowering")
        for var, value in zip(loop.carried_args, init, strict=True):
            self.memo[id(var)] = value
        induction = Var(loop.induction_var.name, type=_INDEX)
        self.memo[id(loop.induction_var)] = induction
        inner = _Cursor()
        self.owner_cursors[id(loop)] = inner
        self.lower(loop.body, inner)
        yielded = tuple(self.lower(value, inner) for value in loop.yield_values)
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
                MeshCoord(mesh=self._physical_frame(value.target.mesh)),
                value.args,
                type=value.type,
            )
        if isinstance(value, Call) and isinstance(value.target, HirBinary):
            dim_op = _DIM_BINARY.get(value.target.kind)
            if dim_op is None or not isinstance(value.type, TensorType) or value.type.shape != ():
                raise LoweringError(
                    f"dimension uses unsupported binary operation {value.target.kind}"
                )
            return simplify_dim(dim_op, tuple(self._dim(arg) for arg in value.args))
        if isinstance(value, Call) and is_dim_op_call(value):
            return Call(value.target, tuple(self._dim(arg) for arg in value.args), type=value.type)
        if isinstance(value, Expr):
            return self._known(value)
        raise LoweringError(f"dimension {value!r} is not lowerable")

    def _lower_call(self, call: Call, cursor: _Cursor) -> Expr:
        target = call.target
        if is_dim_op_call(call) or isinstance(target, (MeshCoord, HirBinary)):
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
            if call.type.storage is StorageKind.GMEM:
                result = self._ensure_output_seed(call)
            else:
                result = self._declare(
                    call,
                    call.type,
                    1,
                    self.owner_cursors[id(self.function)],
                )
                self._emit_fill(result, call.type, cursor)
        elif isinstance(target, HirCast):
            if call.type.storage is StorageKind.GMEM:
                raise LoweringError(
                    f"{_label(call)} is an unscheduled gmem cast; T.cast accepts only rmem "
                    "operands, so write an explicit tf.schedule for each storage transition"
                )
            result = self._declare(
                call,
                call.type,
                1,
                self.owner_cursors[id(self.function)],
            )
            source = self.lower(call.args[0], cursor)
            self._emit_cast(source, result, call.type, cursor)
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
        destination_root = self._material_root(destination)
        if (
            isinstance(destination_root, Call)
            and isinstance(destination_root.target, Zeros)
            and destination_root.type.storage is StorageKind.GMEM
        ):
            self._ensure_output_seed(destination_root)
            if not self.output_initialized:
                owner = self.owner_cursors[id(self.function)]
                self._emit_fill(self.output, destination_root.type, owner)
                self.output_initialized = True
        if not (isinstance(destination, Call) and isinstance(destination.target, Zeros)):
            self.lower(destination, cursor)
        update_root = self._material_root(call.args[1])
        if (
            isinstance(update_root, Call)
            and isinstance(update_root.target, ScheduleOp)
            and isinstance(update_root.type, TensorType)
            and update_root.type.storage is StorageKind.GMEM
        ):
            self.output_windows[id(update_root)] = call
        update = self.lower(call.args[1], cursor)
        if id(update_root) in self.output_windows:
            return self.output
        offsets = call.args[2]
        starts = (
            tuple(self._dim(value) for value in offsets.elements)
            if isinstance(offsets, Tuple)
            else (self._dim(offsets), *(i64_const(0) for _ in self.output.type.shape[1:]))
        )
        desired = replace(
            update.type,
            storage=StorageKind.GMEM,
            layout=Layout(
                tuple(update.type.shape),
                tuple(compact_row_major(tuple(self.output.type.shape))),
            ),
        )
        self._emit_copy(
            update,
            self.output,
            cursor,
            starts=starts,
            sizes=tuple(update.type.shape),
            desired=desired,
        )
        return self.output

    def _insert_target(
        self,
        target: Expr,
        starts: tuple[Expr, ...],
        sizes: tuple,
        desired: TensorType,
        cursor: _Cursor,
    ) -> Expr:
        return self._window(
            target,
            starts,
            sizes,
            desired,
            cursor,
            "window",
        )

    def _lower_schedule(self, call: Call, cursor: _Cursor) -> Expr:
        if access_relation_registry.lookup(type(call.target.op)) is None:
            raise LoweringError(
                f"{_label(call)} selects {type(call.target.op).__name__}, which has no "
                "registered access relation"
            )
        scope = self.scope_for_call.get(id(call))
        mesh = scope.enclosing_mesh() if scope is not None else None
        params = self._instruction_params(call.target.op)
        reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
        writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
        produced = tuple(param for param in writes if not param.effect & MemoryEffect.READ)
        output_window = self.output_windows.get(id(call))
        if produced and output_window is None and id(call) not in self.memo:
            staged = isinstance(call.type, TensorType) and call.type.storage in (
                StorageKind.SMEM,
                StorageKind.GMEM,
            )
            if (
                isinstance(call.type, TensorType)
                and call.type.storage is StorageKind.RMEM
                and call.target.buffers != 1
            ):
                raise LoweringError(f"{_label(call)} staged buffer has no owning scope")
            destination = (
                self.owner_cursors.get(id(self._staging_owner(call))) if staged else cursor
            )
            if destination is None:
                raise LoweringError(f"{_label(call)} staged buffer has no owning scope")
            self._declare(call, call.type, call.target.buffers, destination)
        read_values = {
            param.name: self.lower(value, cursor)
            for param, value in zip(reads, call.args, strict=True)
        }
        written = (
            self._stage(call)
            if id(call) in self.staged
            else self.memo.get(id(call))
        )
        operands: list[tuple[str, Expr | None]] = []
        for param in params:
            value = read_values[param.name] if param.effect & MemoryEffect.READ else written
            if value is None and output_window is None:
                raise LoweringError(f"{_label(call)} has no storage for {param.name}")
            operands.append((param.name, value))

        if (atom := getattr(call.target.op, "atom", None)) is not None:
            if any(value is None for _, value in operands):
                raise LoweringError(f"{_label(call)} cannot issue an atom into an output window")
            self._emit_atom(call, atom, tuple(operands), mesh, cursor)
        else:
            written = self._emit_transfer(
                call, tuple(operands), mesh, output_window, written, cursor
            )
        result_param = next(param for param in params if param.effect & MemoryEffect.WRITE)
        if result_param.effect & MemoryEffect.READ:
            return read_values[result_param.name]
        if written is None:
            raise LoweringError(f"{_label(call)} produced no addressable result")
        return written

    def _output_window(self, write: Call, desired: TensorType, cursor: _Cursor) -> Expr:
        assert self.output is not None
        offsets = write.args[2]
        starts_ = (
            tuple(self._dim(value) for value in offsets.elements)
            if isinstance(offsets, Tuple)
            else (
                self._dim(offsets),
                *(i64_const(0) for _ in self.output.type.shape[1:]),
            )
        )
        layout = Layout(
            tuple(desired.shape),
            tuple(compact_row_major(tuple(self.output.type.shape))),
        )
        return self._window(
            self.output,
            starts_,
            tuple(desired.shape),
            replace(desired, storage=StorageKind.GMEM, layout=layout),
            cursor,
            "window",
        )

    def _emit_transfer(self, call, operands, mesh, output_window, written, cursor):
        if mesh is None:
            raise LoweringError(f"{_label(call)} instruction issue has no lexical mesh")
        declared = next(
            (held for _role, value in operands if value is not None and (held := self._holder_mesh(self.logical.get(id(value), value.type))) is not None),
            None,
        )
        try:
            frame = self._physical_frame(declared or nonunit_mesh(mesh))
        except ValueError as error:
            raise LoweringError(f"{_label(call)} {error}") from error
        issue, issued = _Cursor(), []
        for role, value in operands:
            if value is None:
                value = self._output_window(output_window, call.type, issue)
                self.memo[id(call)], self.logical[id(value)], written = value, call.type, value
            desired = self.logical.get(id(value), value.type)
            issued.append(self._whole_operand(value, desired, frame, issue, f"{role}_frame"))
        issue.add(Evaluate(type(call.target.op)(), tuple(issued)))
        cursor.add(MeshScope(frame, Var(self.names.fresh("threads"), type=_BINDING), issue.build()))
        return written

    def _emit_atom(self, call, atom, operands, mesh, cursor) -> None:
        logical = tuple(self.logical.get(id(value), value.type) for _, value in operands)
        shapes = atom.operand_shapes()
        try:
            axes = operand_axes(call.target.op, logical)
        except ValueError as error:
            raise LoweringError(f"{_label(call)} {error}") from error
        whole = (logical[0].shape[0], logical[0].shape[1], logical[1].shape[1])
        tile = (shapes[0][0], shapes[0][1], shapes[1][1])
        repeat = tuple(full // part for full, part in zip(whole, tile, strict=True))
        if any(full % part for full, part in zip(whole, tile, strict=True)):
            axis = next(i for i, (full, part) in enumerate(zip(whole, tile)) if full % part)
            raise LoweringError(
                f"{_label(call)} axis {('m', 'n', 'k')[axis]} extent {whole[axis]} "
                f"is not divisible by atom {tile[axis]}"
            )
        order = call.target.order or tuple(range(3))
        if mesh is None:
            raise LoweringError(f"{_label(call)} has atom axes but no declared physical mesh")
        try:
            groups = tuple(issue_frames(mesh, atom.required_execution_mesh, repeat, tile))
        except ValueError as error:
            raise LoweringError(f"{_label(call)} {error}") from error
        for frame, lows in groups:
            frame = self._physical_frame(frame)
            try:
                desired, declared_rows = atom.operand_tiles(logical, frame, axes)
            except ValueError as error:
                raise LoweringError(f"{_label(call)} {error}") from error
            rows = [1, 1, 1]
            row_axes = tuple(None if row is None else row[0] for row in declared_rows)
            for axis, count in (row for row in declared_rows if row is not None):
                rows[axis] = max(rows[axis], count)

            def emit(depth, offsets, copies):
                if depth == len(order):
                    issue, issued = _Cursor(), []
                    for (role, value), shape, type_, mapped, row_axis in zip(
                        operands, shapes, desired, axes, row_axes, strict=True
                    ):
                        shift = 0 if row_axis is None else copies.get(row_axis, 0) * tile[row_axis]
                        issued.append(
                            self._window(
                                value,
                                tuple(offsets[axis] for axis in mapped),
                                shape,
                                _with_frame(self._shift_fragment(type_, shift), frame),
                                issue,
                                f"{role}_view",
                            )
                        )
                    issue.add(Evaluate(type(call.target.op)(atom=atom.on(frame)), tuple(issued)))
                    return issue.build()
                axis = order[depth]
                low, extent = lows.get(axis, 0), tile[axis] if axis in lows else whole[axis]
                beside = min(rows[axis], extent // tile[axis])
                if extent % (tile[axis] * beside):
                    raise LoweringError(
                        f"{_label(call)} axis {('m', 'n', 'k')[axis]} extent {extent} "
                        f"is not divisible by row {tile[axis]} * {beside}"
                    )
                counter, body = Var(self.names.fresh(f"o_{('m', 'n', 'k')[axis]}"), type=_INDEX), _Cursor()
                for copy in range(beside):
                    start = counter if copy == 0 else simplify_dim(DimAdd, (counter, copy * tile[axis]))
                    for statement in emit(depth + 1, {**offsets, axis: start}, {**copies, axis: copy}).body:
                        body.add(statement)
                return Sequential((For(counter, i64_const(low), i64_const(low + extent), i64_const(tile[axis] * beside), body.build()),))

            body = emit(0, {}, {})
            cursor.add(MeshScope(frame, Var(self.names.fresh("threads"), type=_BINDING), body))

    @staticmethod
    def _shift_fragment(type_: TensorType, shift: int) -> TensorType:
        if shift == 0:
            return type_
        layout = type_.layout
        if isinstance(layout, ShardLayout):
            held = layout.layout
            if isinstance(held, ComposedLayout) and isinstance(held.inner, Swizzle):
                layout = replace(layout, layout=replace(held, offset=held.offset + shift))
        elif isinstance(layout, ComposedLayout) and isinstance(layout.inner, Swizzle):
            layout = replace(layout, offset=layout.offset + shift)
        return replace(type_, layout=layout)

    @staticmethod
    def _instruction_params(op) -> tuple:
        schema = getattr(type(op), "_op_schema", None)
        if schema is None:
            raise LoweringError(f"{type(op).__name__} is not a registered operation")
        params = tuple(param for param in schema.signature if param.kind == "input")
        if any(param.effect is None for param in params):
            raise LoweringError(f"{type(op).__name__} does not declare every operand memory effect")
        return params

    def _physical_frame(self, mesh: Mesh) -> Mesh:
        frame = _frame(mesh)
        known = next((held for held in self.frames if held == frame), None)
        if known is not None:
            return known
        self.frames.append(frame)
        return frame

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
                    self._physical_frame(mesh),
                    Var(self.names.fresh("threads"), type=_BINDING),
                    statement,
                )
            )

    def _emit_cast(
        self,
        source: Expr,
        target: Expr,
        logical_type: TensorType,
        cursor: _Cursor,
    ) -> None:
        mesh = self._holder_mesh(logical_type)
        if mesh is None:
            raise LoweringError("register cast result has no holder mesh")
        frame = self._physical_frame(mesh)
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

    def _emit_copy(
        self,
        source: Expr,
        target: Expr,
        cursor: _Cursor,
        *,
        starts: tuple[Expr, ...] | None = None,
        sizes: tuple | None = None,
        desired: TensorType | None = None,
    ) -> None:
        source_type = self.logical.get(id(source), source.type)
        if not isinstance(source_type, TensorType) or not isinstance(target.type, TensorType):
            raise LoweringError("copy lowering requires tensor operands")
        mesh = self._holder_mesh(source_type) or self._holder_mesh(target.type)
        issue = cursor if mesh is None else _Cursor()
        read = source
        if mesh is not None:
            frame = self._physical_frame(mesh)
            read = self._whole_operand(
                source,
                source_type,
                frame,
                issue,
                f"{getattr(source, 'name', 'value')}_view",
            )
        written = target
        if starts is not None:
            if sizes is None or desired is None:
                raise LoweringError("a windowed copy needs sizes and a destination type")
            written = self._insert_target(target, starts, sizes, desired, issue)
        issue.add(Evaluate(Copy(), (read, written)))
        if mesh is None:
            return
        cursor.add(
            MeshScope(
                frame,
                Var(self.names.fresh("threads"), type=_BINDING),
                issue.build(),
            )
        )

    def _finish(self, result: Expr, cursor: _Cursor) -> None:
        assert self.output is not None
        if self.output_seed is not None and not self.output_initialized:
            self._emit_fill(self.output, self.output_seed.type, cursor)
            self.output_initialized = True
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
