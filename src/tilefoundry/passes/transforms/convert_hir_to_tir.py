"""Lower scheduled HIR to effect-form TIR.

The pass consumes the checked, inlined view produced by ``analyze(memory)``.
Memory metadata owns storage addresses; registered access relations own
instruction issue geometry. This file only turns those two facts into statements.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

import isl

from tilefoundry.analysis import AnalysisResult, MemoryMetadata, analyze
from tilefoundry.analysis.iteration_scope import IterationScope, walk_scopes
from tilefoundry.analysis.liveness import storage_source
from tilefoundry.inspection.analysis_report import render_analysis
from tilefoundry.inspection.values import ReportIdentity, ReportSelection
from tilefoundry.ir.core import (
    BindingMetadata,
    Call,
    Constant,
    Expr,
    SourceSpanMetadata,
    Tuple,
    Var,
    attach_metadata,
    get_metadata,
    op_identifier,
)
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp, operand_relations
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor._view_layout import derive_view_layout
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.zeros import Zeros
from tilefoundry.ir.tir.async_copy import CopyAsync, is_indexed_schedule
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
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
    iteration_universe,
    projected_axes,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.candidates import candidate_ops, sole_candidate
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext
from tilefoundry.visitor_registry.registries import typeinfer_registry
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout

_INDEX = TensorType.umat_scalar()
_BINDING = TensorType.scalar(DType.i64, storage=StorageKind.RMEM)
_DIM_BINARY = {
    BinaryKind.ADD: DimAdd,
    BinaryKind.SUB: DimSub,
    BinaryKind.MUL: DimMul,
    BinaryKind.FLOOR_DIV: DimFloorDiv,
    BinaryKind.MOD: DimMod,
}


def _iteration_geometry(relations) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Static extents and authored axis names of one relation domain."""
    domain = iteration_universe(relations)
    if domain is None:
        raise ValueError("access relations state no iteration space")
    extents, names = [], []
    for axis in range(domain.tuple_dim()):
        low, high = domain.dim_min_val(axis), domain.dim_max_val(axis)
        if not (low.is_int() and high.is_int() and low.get_num_si() == 0):
            raise ValueError(f"iteration axis {axis} is not a zero-based static extent")
        extents.append(high.get_num_si() + 1)
        names.append(domain.get_dim_name(isl.dim_type.SET, axis) or str(axis))
    return tuple(extents), tuple(names)


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


def _scope_names(root: IterationScope, names: Names) -> dict[int, str]:
    """Name emitted MeshRegions in pre-order, omitting zero-only scopes.

    The root mesh is emitted separately as cta by run, not visit_MeshRegion.
    """
    found = {}
    for scope in walk_scopes(root):
        region = scope.owner
        if not isinstance(region, MeshRegion) or region is root.owner.body:
            continue
        if not (isinstance(region.body, Call) and isinstance(region.body.target, Zeros)):
            found[id(region)] = names.fresh("scope")
    return found


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

    result: AnalysisResult
    authored: Function

    def __post_init__(self) -> None:
        ExprVisitor.__init__(self)
        self.module = self.result.module
        self.function = self.result.function
        self.names = Names(self.authored)
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
        self.scopes = self.result.scopes
        self.scope_for_call = {
            expr_id: scope
            for scope in walk_scopes(self.scopes)
            for expr_id, (call, _relations) in scope.relations.items()
            if id(call) == expr_id
        }
        self.type_ctx = TypeInferContext(scope=FunctionScope(self.module, self.function))
        for param in self.function.params:
            physical = Var(param.name, type=param.type, is_const=param.is_const)
            self._memo[id(param)] = (param, physical)
            if isinstance(param.type, TensorType):
                self.logical[id(physical)] = param.type

    def run(self) -> PrimFunction:
        if self.function.body is None:
            raise LoweringError(f"{self.function.name!r} has no body to lower")
        if not isinstance(self.function.return_type, TensorType):
            raise LoweringError("scheduled lowering currently requires one tensor result")

        self.scope_names = _scope_names(self.scopes, self.names)
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
            result = self.visit(body.body, inner)
        else:
            result = self.visit(body, inner)
        self._finish(result, inner)
        statements = inner.build()
        if isinstance(body, MeshRegion):
            statements = Sequential(
                (MeshScope(root_mesh, Var(self.names.fresh("cta"), type=_BINDING), statements),)
            )

        statements = _MeshScopeCoalescer().visit(statements)

        params = tuple(self._memo[id(param)][1] for param in self.function.params)
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
        self._memo[id(seed)] = (seed, self.output)
        self.logical[id(self.output)] = seed.type
        return self.output

    def _material_root(self, value: Expr) -> Expr:
        seen: set[int] = set()
        while id(value) not in seen:
            seen.add(id(value))
            following = storage_source(value, self.bindings)
            if following is not None:
                value = following
                continue
            if isinstance(value, LoopRegion) and len(value.yield_values) == 1:
                value = value.args[0]
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
        if not isinstance(type_, TensorType):
            raise LoweringError(f"{_label(expr)} has non-tensor material result {type_!r}")
        storage_type = _storage_type(type_)
        stem = self.names.binding(self.authored_values.get(id(expr)))
        if storage_type.storage is StorageKind.RMEM:
            var = Var(stem, type=storage_type)
            cursor.bind(
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
            self.logical[id(var)] = type_
            return var
        fields = tuple(self._literal_view(storage_type, offset) for offset in offsets)
        if len(fields) == 1:
            var = Var(stem, type=storage_type)
            cursor.bind(var, fields[0])
            self.logical[id(var)] = type_
            return var
        tuple_type = TupleType(tuple(field.type for field in fields))
        stages = Var(self.names.fresh(f"{stem}_stages"), type=tuple_type)
        cursor.bind(stages, Tuple(fields, type=tuple_type))
        self.logical[id(stages)] = type_
        self.staged[id(expr)] = (stages, self._staging_loop(expr))
        return self._stage(expr)

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
            known = self._memo.get(id(loop.induction_var))
            induction = None if known is None else known[1]
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
        known = self._memo.get(id(expr))
        return expr if known is None else known[1]

    def _bind_region_args(self, region: LoopRegion | MeshRegion, cursor: _Cursor) -> None:
        self.bindings.update(zip(map(id, region.params), region.args, strict=True))
        values = tuple(self.visit(arg, cursor) for arg in region.args)
        for param, value in zip(region.params, values, strict=True):
            self._memo[id(param)] = (param, value)

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

    def dispatch_visit(self, expr: Expr, ctx: _Cursor) -> Expr:
        self._pair_authored(expr)
        return super().dispatch_visit(expr, ctx)

    @staticmethod
    def visit_Constant(expr: Constant, _cursor: _Cursor) -> Expr:
        return expr

    @staticmethod
    def visit_Var(expr: Var, _cursor: _Cursor) -> Expr:
        raise LoweringError(f"unbound HIR value {expr.name!r} reached lowering")

    def visit_Tuple(self, expr: Tuple, cursor: _Cursor) -> Expr:
        elements = tuple(self.visit(element, cursor) for element in expr.elements)
        return Tuple(elements, type=TupleType(tuple(element.type for element in elements)))

    def visit_MeshRegion(self, region: MeshRegion, cursor: _Cursor) -> Expr:
        self._bind_region_args(region, cursor)
        if isinstance(region.body, Call) and isinstance(region.body.target, Zeros):
            return self.visit(region.body, cursor)
        inner = _Cursor()
        self.owner_cursors[id(region)] = inner
        result = self.visit(region.body, inner)
        built = inner.build()
        if built.body:
            cursor.add(
                MeshScope(
                    self._physical_frame(region.mesh),
                    Var(self.scope_names[id(region)], type=_BINDING),
                    built,
                )
            )
        return result

    def visit_LoopRegion(self, loop: LoopRegion, cursor: _Cursor) -> Expr:
        self._bind_region_args(loop, cursor)
        init = tuple(self._known(param) for param in loop.params[: len(loop.yield_values)])
        induction = Var(loop.induction_var.name, type=_INDEX)
        self._memo[id(loop.induction_var)] = (loop.induction_var, induction)
        inner = _Cursor()
        self.owner_cursors[id(loop)] = inner
        self.visit(loop.body, inner)
        yielded = tuple(self.visit(value, inner) for value in loop.yield_values)
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
        return result

    def _dim(self, value) -> Expr:
        number = static_dim_value(value)
        if number is not None:
            return i64_const(number)
        if isinstance(value, Var):
            known = self._memo.get(id(value))
            if known is None:
                raise LoweringError(f"dimension uses unbound value {value.name!r}")
            return known[1]
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

    def visit_Call(self, call: Call, cursor: _Cursor) -> Expr:
        target = call.target
        if is_dim_op_call(call):
            result = self._dim(call)
        else:
            visitor = getattr(self, f"visit_{type(target).__name__}", None)
            if visitor is None:
                result = self._lower_automatic_instruction(call, cursor)
            else:
                result = visitor(call, cursor)
        return result

    def _lower_automatic_instruction(self, call: Call, cursor: _Cursor) -> Expr:
        target = call.target
        candidates = candidate_ops(type(target))
        if not candidates:
            raise LoweringError(
                f"{_label(call)} uses unknown HIR call {type(target).__name__}"
            )
        op = sole_candidate(target)
        if op is None:
            available = ", ".join(op_identifier(candidate) for candidate in candidates)
            raise LoweringError(
                f"{_label(call)} has instruction candidates [{available}] but no automatic "
                "selection; write tf.schedule to choose the instruction and its attributes"
            )
        if not isinstance(call.type, TensorType):
            raise LoweringError(f"{_label(call)} candidate result is not a tensor")
        if call.type.storage is not StorageKind.RMEM:
            name = type(target)._op_schema.name
            identifier = op_identifier(type(op))
            raise LoweringError(
                f"{_label(call)} is an unscheduled {call.type.storage} {name}; "
                f"{identifier} accepts only "
                "rmem operands, so write an explicit tf.schedule for each storage transition"
            )
        return self._lower_instruction(call, op, cursor)

    def visit_MeshCoord(self, call: Call, _cursor: _Cursor) -> Expr:
        return self._dim(call)

    def visit_Binary(self, call: Call, cursor: _Cursor) -> Expr:
        if isinstance(call.type, TensorType) and call.type.shape == ():
            return self._dim(call)
        return self._lower_automatic_instruction(call, cursor)

    def visit_TupleGetItem(self, call: Call, cursor: _Cursor) -> Expr:
        source = self.visit(call.args[0], cursor)
        index = self._dim(call.args[1])
        if isinstance(source, Tuple) and isinstance(index, Constant):
            return source.elements[index.value]
        return Call(call.target, (source, index), type=call.type)

    def visit_Slice(self, call: Call, cursor: _Cursor) -> Expr:
        base = self.visit(call.args[0], cursor)
        starts_arg = call.args[1]
        starts = (
            tuple(self._window_start(value, cursor) for value in starts_arg.elements)
            if isinstance(starts_arg, Tuple)
            else (self._window_start(starts_arg, cursor),)
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

    def visit_Reshape(self, call: Call, cursor: _Cursor) -> Expr:
        source = self.visit(call.args[0], cursor)
        desired = replace(call.type, layout=_plain_layout(call.type))
        return self._window(
            source,
            tuple(i64_const(0) for _ in source.type.shape),
            tuple(source.type.shape),
            desired,
            cursor,
            "tile",
        )

    def visit_Bitcast(self, call: Call, cursor: _Cursor) -> Expr:
        source = self.visit(call.args[0], cursor)
        if isinstance(call.type.layout, ShardLayout):
            frame = self._physical_frame(call.type.layout.mesh)
            inner = _Cursor()
            value = self._window(
                source,
                tuple(i64_const(0) for _ in source.type.shape),
                tuple(source.type.shape),
                _with_frame(call.type, frame),
                inner,
                "tile",
            )
            cursor.add(MeshScope(frame, Var(self.names.fresh("threads"), type=_BINDING), inner.build()))
            self.logical[id(value)] = call.type
            return value
        return self._window(
            source,
            tuple(i64_const(0) for _ in source.type.shape),
            tuple(source.type.shape),
            call.type,
            cursor,
            "tile",
        )

    def visit_Reshard(self, call: Call, cursor: _Cursor) -> Expr:
        if aliased_operand(call) is None:
            return self._lower_automatic_instruction(call, cursor)
        source = self.visit(call.args[0], cursor)
        return self._window(
            source,
            tuple(i64_const(0) for _ in source.type.shape),
            tuple(source.type.shape),
            call.type,
            cursor,
            "tile",
        )

    def visit_Transpose(self, call: Call, cursor: _Cursor) -> Expr:
        source = self.visit(call.args[0], cursor)
        result = self._declare(call, call.type, 1, cursor)
        source_type = self.logical.get(id(source), source.type)
        perm = call.target.perm

        def transposed(layout: Layout) -> Layout:
            return Layout(
                tuple(layout.shape[p] for p in perm),
                None if layout.strides is None else tuple(layout.strides[p] for p in perm),
            )

        scope = self.scope_for_call[id(call)]
        layout = derive_output_shard_layout(
            (source_type,), scope.relations[id(call)][1], tuple(call.type.shape),
            fresh_strides=False,
        )
        if layout is None:
            layout = derive_view_layout(
                replace(source_type, layout=_plain_layout(source_type)),
                tuple(call.type.shape),
                transposed,
            )
        if layout is None:
            raise LoweringError(f"{_label(call)} cannot transpose its source layout")
        desired = replace(call.type, layout=layout)
        starts = tuple(i64_const(0) for _ in source.type.shape)
        view = self._window(
            source,
            starts,
            tuple(source.type.shape),
            desired,
            cursor,
            "tile",
        )
        self.logical[id(view)] = desired
        self._emit_instruction(
            call, Copy(), (("src", view), ("dst", result)), scope.enclosing_mesh(),
            None, result, cursor,
        )
        return result

    def visit_Zeros(self, call: Call, cursor: _Cursor) -> Expr:
        if call.type.storage is StorageKind.GMEM:
            return self._ensure_output_seed(call)
        result = self._declare(
            call,
            call.type,
            1,
            self.owner_cursors[id(self.function)],
        )
        self._emit_fill(result, call.type, cursor)
        return result

    def _window_start(self, value: Expr, cursor: _Cursor) -> Expr:
        if isinstance(value, Call) and isinstance(value.target, HirBinary):
            operation = _DIM_BINARY.get(value.target.kind)
            if operation is not None:
                return simplify_dim(
                    operation, tuple(self._window_start(arg, cursor) for arg in value.args)
                )
        return (
            self.visit(value, cursor)
            if isinstance(value, Call) and not is_dim_op_call(value)
            else self._dim(value)
        )

    def visit_IndexSelect(self, call: Call, cursor: _Cursor) -> Expr:
        raise LoweringError(
            f"{_label(call)} is an unscheduled IndexSelect; write "
            "tf.schedule((x, idx), op=T.copy_async(smem_layout=..., fill=...))"
        )

    def visit_InsertSlice(self, call: Call, cursor: _Cursor) -> Expr:
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
        target = self.visit(destination, cursor)
        update_root = self._material_root(call.args[1])
        direct_write = (
            isinstance(update_root, Call)
            and isinstance(update_root.target, ScheduleOp)
            and isinstance(update_root.type, TensorType)
            and update_root.type.storage is target.type.storage
            and id(update_root) not in self._memo
        )
        if direct_write:
            self.output_windows[id(update_root)] = call
        update = self.visit(call.args[1], cursor)
        if direct_write:
            return target
        starts, desired = self._insert_window(call, target, update.type, cursor)
        self._emit_copy(
            update, target, cursor, starts=starts, sizes=tuple(update.type.shape), desired=desired
        )
        return target

    def _insert_window(self, write: Call, target: Expr, desired: TensorType, cursor: _Cursor):
        offsets = write.args[2]
        starts = (
            tuple(self._window_start(value, cursor) for value in offsets.elements)
            if isinstance(offsets, Tuple)
            else (
                self._window_start(offsets, cursor),
                *(i64_const(0) for _ in target.type.shape[1:]),
            )
        )
        keys = Tuple(starts, type=TupleType(tuple(start.type for start in starts)))
        cut = Call(
            Slice(sizes=tuple(desired.shape), strides=(1,) * len(desired.shape)),
            (target, keys),
            type=desired,
        )
        inferred = typeinfer_registry.lookup(Slice)(
            cut, TypeInferContext(memo={id(arg): (arg, arg.type) for arg in (target, *starts)})
        )
        type_ = getattr(inferred, "type", inferred)
        return starts, replace(desired, layout=type_.layout, storage=target.type.storage)

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

    def visit_ScheduleOp(self, call: Call, cursor: _Cursor) -> Expr:
        return self._lower_instruction(call, call.target.op, cursor)

    def _lower_instruction(self, call: Call, op, cursor: _Cursor) -> Expr:
        if access_relation_registry.lookup(type(op)) is None:
            raise LoweringError(
                f"{_label(call)} selects {type(op).__name__}, which has no "
                "registered access relation"
            )
        scope = self.scope_for_call.get(id(call))
        mesh = scope.enclosing_mesh() if scope is not None else None
        params = self._instruction_params(op, len(call.args))
        reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
        writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
        produced = tuple(param for param in writes if not param.effect & MemoryEffect.READ)
        output_window = self.output_windows.get(id(call))
        written = None
        buffers = call.target.buffers if isinstance(call.target, ScheduleOp) else 1
        if produced and output_window is None:
            staged = isinstance(call.type, TensorType) and call.type.storage in (
                StorageKind.SMEM,
                StorageKind.GMEM,
            )
            if (
                isinstance(call.type, TensorType)
                and call.type.storage is StorageKind.RMEM
                and buffers != 1
            ):
                raise LoweringError(f"{_label(call)} staged buffer has no owning scope")
            if staged:
                destination = self.owner_cursors.get(id(self._staging_owner(call)))
            else:
                destination = self.owner_cursors[id(self.function)]
            if destination is None:
                raise LoweringError(f"{_label(call)} staged buffer has no owning scope")
            written = self._declare(call, call.type, buffers, destination)
        read_values = {}
        for param, operand in zip(reads, call.args, strict=True):
            value = self.visit(operand, cursor)
            if (
                isinstance(op, CopyAsync)
                and is_indexed_schedule(call.args)
                and param.name == "src"
                and (
                    value.type.layout is None
                    or any(value is self._memo[id(argument)][1] for argument in self.function.params)
                )
            ):
                type_ = replace(
                    value.type,
                    layout=value.type.layout
                    or Layout(tuple(value.type.shape), tuple(compact_row_major(value.type.shape))),
                )
                value = self._window(
                    value,
                    tuple(i64_const(0) for _ in value.type.shape),
                    tuple(value.type.shape),
                    type_,
                    cursor,
                    "source",
                    whole_tensor=True,
                )
            if (
                isinstance(value.type, TensorType)
                and value.type.shape == ()
                and value.type.storage is StorageKind.UMAT
            ):
                scalar_type = replace(value.type, storage=StorageKind.RMEM)
                scalar = Var(self.names.fresh("scalar"), type=scalar_type)
                cursor.bind(
                    scalar, Call(AllocTensor(tensor_type=scalar_type), (), type=scalar_type)
                )
                cursor.add(Evaluate(Fill(), (scalar, value)))
                self.logical[id(scalar)] = scalar_type
                value = scalar
            read_values[param.name] = value
        if id(call) in self.staged:
            written = self._stage(call)
        operands: list[tuple[str, Expr | None]] = []
        for param in params:
            value = read_values[param.name] if param.effect & MemoryEffect.READ else written
            if value is None and output_window is None:
                raise LoweringError(f"{_label(call)} has no storage for {param.name}")
            operands.append((param.name, value))

        written = self._emit_instruction(
            call, op, tuple(operands), mesh, output_window, written, cursor
        )
        result_param = next(param for param in params if param.effect & MemoryEffect.WRITE)
        if result_param.effect & MemoryEffect.READ:
            return read_values[result_param.name]
        if written is None:
            raise LoweringError(f"{_label(call)} produced no addressable result")
        return written

    def _output_window(self, write: Call, desired: TensorType, cursor: _Cursor) -> Expr:
        target = self.visit(write.args[0], cursor)
        starts, desired = self._insert_window(write, target, desired, cursor)
        return self._window(target, starts, tuple(desired.shape), desired, cursor, "window")

    def _emit_instruction(self, call, op, operands, mesh, output_window, written, cursor):
        atom = getattr(op, "atom", None)
        if atom is None:
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
                    self.logical[id(value)], written = call.type, value
                desired = self.logical.get(id(value), value.type)
                issued.append(self._whole_operand(value, desired, frame, issue, f"{role}_frame"))
            issue.add(Evaluate(self._issued_op(op), tuple(issued)))
            cursor.add(MeshScope(frame, Var(self.names.fresh("threads"), type=_BINDING), issue.build()))
            return written
        if any(value is None for _, value in operands):
            raise LoweringError(f"{_label(call)} cannot issue an atom into an output window")
        logical = tuple(self.logical.get(id(value), value.type) for _, value in operands)
        shapes = atom.operand_shapes()
        try:
            relations = operand_relations(op, logical)
            projected = tuple(projected_axes(boundary) for boundary in relations[: len(logical)])
            if any(axis is None for mapped in projected for axis in mapped):
                raise ValueError(
                    f"{atom.reference_name} access relation does not project every operand axis"
                )
            axes = tuple(
                tuple(axis for axis in mapped if axis is not None) for mapped in projected
            )
            whole, axis_names = _iteration_geometry(relations)
            single_types = tuple(
                replace(type_, shape=shape)
                for type_, shape in zip(logical, shapes, strict=True)
            )
            tile, _ = _iteration_geometry(operand_relations(op, single_types))
        except ValueError as error:
            raise LoweringError(f"{_label(call)} {error}") from error
        if len(whole) != len(tile):
            raise LoweringError(
                f"{_label(call)} whole and atom iteration ranks differ: "
                f"{len(whole)} vs {len(tile)}"
            )
        repeat = tuple(full // part for full, part in zip(whole, tile, strict=True))
        if any(full % part for full, part in zip(whole, tile, strict=True)):
            axis = next(i for i, (full, part) in enumerate(zip(whole, tile)) if full % part)
            raise LoweringError(
                f"{_label(call)} axis {axis_names[axis]} extent {whole[axis]} "
                f"is not divisible by atom {tile[axis]}"
            )
        order = (
            call.target.order if isinstance(call.target, ScheduleOp) else None
        ) or tuple(range(len(whole)))
        if mesh is None:
            raise LoweringError(f"{_label(call)} has atom axes but no declared physical mesh")
        try:
            groups = tuple(issue_frames(mesh, atom.required_execution_mesh, repeat, tile))
        except ValueError as error:
            raise LoweringError(f"{_label(call)} {error}") from error
        for frame, lows in groups:
            frame = self._physical_frame(frame)
            try:
                desired, declared_rows = atom.operand_tiles(logical, frame, axes, repeat=repeat)
            except ValueError as error:
                raise LoweringError(f"{_label(call)} {error}") from error
            rows = [1] * len(whole)
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
                    issue.add(Evaluate(type(op)(atom=atom.on(frame)), tuple(issued)))
                    return issue.build()
                axis = order[depth]
                low, extent = lows.get(axis, 0), tile[axis] if axis in lows else whole[axis]
                beside = min(rows[axis], extent // tile[axis])
                if extent % (tile[axis] * beside):
                    raise LoweringError(
                        f"{_label(call)} axis {axis_names[axis]} extent {extent} "
                        f"is not divisible by row {tile[axis]} * {beside}"
                    )
                counter = Var(self.names.fresh(f"o_{axis_names[axis]}"), type=_INDEX)
                body = _Cursor()
                for copy in range(beside):
                    start = counter if copy == 0 else simplify_dim(DimAdd, (counter, copy * tile[axis]))
                    for statement in emit(depth + 1, {**offsets, axis: start}, {**copies, axis: copy}).body:
                        body.add(statement)
                return Sequential((For(counter, i64_const(low), i64_const(low + extent), i64_const(tile[axis] * beside), body.build()),))

            body = emit(0, {}, {})
            cursor.add(MeshScope(frame, Var(self.names.fresh("threads"), type=_BINDING), body))
        return written

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
    def _instruction_params(op, operand_count: int | None = None) -> tuple:
        schema = getattr(type(op), "_op_schema", None)
        if schema is None:
            raise LoweringError(f"{type(op).__name__} is not a registered operation")
        params = tuple(param for param in schema.signature if param.kind == "input")
        if any(param.effect is None for param in params):
            raise LoweringError(f"{type(op).__name__} does not declare every operand memory effect")
        if operand_count is not None:
            required_reads = sum(
                bool(param.effect & MemoryEffect.READ)
                for param in params
                if not param.optional
            )
            optional_reads = tuple(
                param for param in params if param.optional and param.effect & MemoryEffect.READ
            )
            supplied_optional = operand_count - required_reads
            included = {id(param) for param in optional_reads[: max(0, supplied_optional)]}
            params = tuple(
                param for param in params if not param.optional or id(param) in included
            )
        return params

    @staticmethod
    def _issued_op(op):
        """Keep required dispatch attributes; view-selection attributes are consumed."""
        attributes = {
            param.name: getattr(op, param.name)
            for param in type(op)._op_schema.signature
            if param.kind == "attribute" and (not param.has_default or param.name == "fill")
        }
        return type(op)(**attributes)

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
        *,
        whole_tensor: bool = False,
    ) -> Var:
        """Build a buffer view.

        Args:
            base: Source buffer.
            starts: Source coordinates of the window's first element.
            sizes: Source window extents.
            desired: Result tensor type and layout.
            cursor: Scope receiving the bound view.
            stem: Prefix for the view's generated name.
            whole_tensor: Runtime table sources avoid unsupported PtrOf(Slice) CUDA (side finding).
        """
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
        source = base
        if not whole_tensor:
            keys = Tuple(starts, type=TupleType(tuple(start.type for start in starts)))
            cut_type = TensorType(tuple(sizes), held.dtype, None, held.storage)
            source = Call(
                Slice(sizes=tuple(sizes), strides=(1,) * len(sizes)),
                (base, keys),
                type=cut_type,
            )
            inferred = typeinfer_registry.lookup(Slice)(
                source, TypeInferContext(memo={id(arg): (arg, arg.type) for arg in (base, *starts)})
            )
            source.type = getattr(inferred, "type", inferred)
        pointer = Call(PtrOf(), (source,), type=PointerType(held.dtype, held.storage))
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
        lowered = Lowering(result, authored).run()
        lowered.metadata = result.function.metadata
        for record in render_analysis(result).summary:
            if isinstance(record, (ReportIdentity, ReportSelection)):
                attach_metadata(lowered, record)
        functions = tuple(
            lowered if function is authored else function for function in module.functions
        )
        return replace(module, functions=functions)


__all__ = ["ConvertHIRToTIR", "LoweringError"]
