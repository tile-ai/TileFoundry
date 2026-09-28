"""Lower scheduled HIR to effect-form TIR.

The pass consumes the checked, inlined view produced by ``analyze(memory)``.
Memory metadata owns storage addresses; ``IssuePlan`` owns instruction issue
geometry.  This file only turns those two public facts into statements.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import prod

import isl

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
from tilefoundry.ir.hir.schedule import IssuePlan, ScheduleOp, issue_plan
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.zeros import Zeros
from tilefoundry.ir.pattern import MeshPattern, PatternMatcher
from tilefoundry.ir.tir.cast import Cast as TirCast
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
    Split,
    StorageKind,
    Swizzle,
    TensorType,
    TupleType,
    make_mesh,
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
from tilefoundry.ir.types.mesh import separate, starts
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.ir.types.utils import i64_const, static_dim_value
from tilefoundry.ir.visitor import ExprVisitor, StmtMutator, collect_exprs
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


def _scope_pattern(op) -> MeshPattern | None:
    schema = getattr(type(op), "_op_schema", None)
    if schema is None:
        return None
    return next(
        (
            param.pattern
            for param in schema.signature
            if param.kind == "attribute"
            and param.name == "scope"
            and isinstance(param.pattern, MeshPattern)
        ),
        None,
    )


def _squeezed_scope(mesh: Mesh, pattern: MeshPattern) -> Mesh:
    """Select the declared topology levels and remove lexical unit modes."""
    by_name = {
        getattr(level.topologies[0], "name", level.topologies[0]): level for level in separate(mesh)
    }
    if any(name not in by_name for name in pattern.topologies):
        raise LoweringError(
            f"instruction scope requires topologies {pattern.topologies}, got {tuple(by_name)}"
        )
    selected = make_mesh(*(by_name[name] for name in pattern.topologies))
    layout = flatten(selected.layout)
    if not isinstance(layout, Layout) or layout.strides is None:
        raise LoweringError("instruction scope requires a static strided physical mesh")
    modes = tuple(
        (extent, stride)
        for extent, stride in zip(flatten(layout.shape), flatten(layout.strides), strict=True)
        if extent != 1
    )
    if not modes:
        modes = ((1, 1),)
    outer = Layout(tuple(extent for extent, _ in modes), tuple(stride for _, stride in modes))
    frame = Mesh(
        selected.topologies,
        ComposedLayout(None, starts(selected)[0], outer),
        tuple(f"d{index}" for index in range(len(modes))),
    )
    matcher = PatternMatcher()
    if not matcher.match(pattern, frame) or not matcher.solve():
        raise LoweringError(f"instruction scope pattern does not match issuing mesh {frame!r}")
    return frame


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
        self.frames: list[Mesh] = []
        self.current_mesh: list[Mesh] = []
        self.current_loops: list[tuple[LoopRegion, Var]] = []
        self.output: Var | None = None
        self.scratch: list[Var] = []
        self.scopes = build_scopes(self.module, self.function)
        self.scope_for_call = {
            expr_id: scope
            for scope in walk_scopes(self.scopes)
            for expr_id, (call, _accesses) in scope.accesses["narrow"].items()
            if id(call) == expr_id
        }
        self.address_owner = {
            id(expr): id(self._staging_owner(expr))
            for expr in collect_exprs(self.function.body)
            if isinstance(expr, Call)
            and isinstance(expr.target, ScheduleOp)
            and isinstance(expr.type, TensorType)
            and expr.type.storage in (StorageKind.SMEM, StorageKind.GMEM)
        }
        self.type_ctx = TypeInferContext(scope=FunctionScope(self.module, self.function))
        self.bindings = {
            id(param): value
            for expr in collect_exprs(self.function.body)
            if isinstance(expr, MeshRegion)
            for param, value in zip(expr.params, expr.args, strict=True)
        }
        self.bindings.update(
            {
                id(param): value
                for expr in collect_exprs(self.function.body)
                if isinstance(expr, LoopRegion)
                for param, value in zip(expr.carried_args, expr.init_args, strict=True)
            }
        )
        self.output_windows = {
            id(root): expr
            for expr in collect_exprs(self.function.body)
            if isinstance(expr, Call)
            and isinstance(expr.target, InsertSlice)
            and isinstance((root := self._material_root(expr.args[1])), Call)
            and isinstance(root.target, ScheduleOp)
            and isinstance(root.type, TensorType)
            and root.type.storage is StorageKind.GMEM
        }
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
        if isinstance(body, MeshRegion):
            root_mesh = self._physical_frame(body.mesh)
            inner = _Cursor()
            self._seed_output(inner)
            self._prepare_buffers(self.function, inner)
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
            self._seed_output(inner)
            self._prepare_buffers(self.function, inner)
            result = self.lower(body, inner)
            self._finish(result, inner)
            statements = inner.build()

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
        scope = self.scope_for_call.get(id(expr))
        cursor = scope
        while cursor is not None:
            if isinstance(cursor.owner, LoopRegion):
                return self.function if cursor.parent is None else cursor.parent.owner
            cursor = cursor.parent
        return self.function

    def _prepare_buffers(self, owner: Expr, cursor: _Cursor) -> None:
        """Declare eager values and staged buffers at their owning scope."""
        for expr in collect_exprs(self.function.body):
            if not isinstance(expr, Call):
                continue
            if (
                isinstance(expr.target, Zeros)
                and isinstance(expr.type, TensorType)
                and expr.type.storage is StorageKind.GMEM
            ):
                assert self.output is not None
                self.memo[id(expr)] = self.output
                self.logical[id(self.output)] = expr.type
                continue
            if isinstance(expr.target, (Zeros, HirCast)):
                if owner is self.function:
                    self._declare(expr, expr.type, 1, cursor)
                continue
            if not isinstance(expr.target, ScheduleOp):
                continue
            if id(expr) in self.output_windows or self.address_owner.get(id(expr)) != id(owner):
                continue
            params = self._instruction_params(expr.target.op)
            writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
            produced = tuple(param for param in writes if not param.effect & MemoryEffect.READ)
            if produced:
                self._declare(expr, expr.type, expr.target.buffers, cursor)

    def _seed_output(self, cursor: _Cursor) -> None:
        """Materialize a returned zero seed unless window writes cover it whole."""
        assert self.output is not None
        seeds = tuple(
            expr
            for expr in collect_exprs(self.function.body)
            if isinstance(expr, Call)
            and isinstance(expr.target, Zeros)
            and isinstance(expr.type, TensorType)
            and expr.type.storage is StorageKind.GMEM
        )
        if not seeds:
            return
        if len(seeds) != 1:
            raise LoweringError("scheduled lowering cannot choose among multiple gmem seeds")
        seed = seeds[0]
        writes = tuple(
            expr
            for expr in collect_exprs(self.function.body)
            if isinstance(expr, Call)
            and isinstance(expr.target, InsertSlice)
            and self._material_root(expr.args[0]) is seed
        )
        if not any(self._covers_output(write) for write in writes):
            self._emit_fill(self.output, seed.type, cursor)
        self.memo[id(seed)] = self.output
        self.logical[id(self.output)] = seed.type
        self.emitted.add(id(seed))

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

    def _covers_output(self, write: Call) -> bool:
        """Prove that one rectangular insert site tiles the complete output."""
        assert self.output is not None
        scope = self.scope_for_call.get(id(write))
        offsets = write.args[2]
        starts_ = offsets.elements if isinstance(offsets, Tuple) else (offsets,)
        update = write.args[1]
        if scope is None or not isinstance(update.type, TensorType):
            return False
        sizes = tuple(static_dim_value(size) for size in update.type.shape)
        output_shape = tuple(static_dim_value(size) for size in self.output.type.shape)
        if (
            any(size is None or size < 1 for size in sizes)
            or any(size is None or size < 1 for size in output_shape)
            or len(starts_) != len(output_shape)
        ):
            return False

        loops = []
        cursor = scope
        while cursor is not None:
            if isinstance(cursor.owner, LoopRegion):
                loops.append(cursor.owner)
            cursor = cursor.parent
        loops.reverse()
        if len(loops) != scope.depth:
            return False

        origins: set[tuple[int, ...]] = set()

        def record(point) -> None:
            values = {
                id(loop.induction_var): point.get_coordinate_val(isl.dim_type.SET, index).num_si()
                for index, loop in enumerate(loops)
            }
            values.update(
                {
                    id(expr): point.get_coordinate_val(isl.dim_type.PARAM, index).num_si()
                    for index, expr in enumerate(scope.domain_params.values())
                }
            )
            try:
                origin = tuple(self._constant_index(start, values) for start in starts_)
            except (ArithmeticError, LoweringError):
                return
            if all(
                start >= 0 and start % size == 0 and start + size <= extent
                for start, size, extent in zip(origin, sizes, output_shape, strict=True)
            ):
                origins.add(origin)

        try:
            scope.domain.foreach_point(record)
        except isl.Error:
            return False
        return len(origins) * prod(sizes) == prod(output_shape)

    def _constant_index(self, value: Expr, values: dict[int, int]) -> int:
        bound = self.bindings.get(id(value))
        if bound is not None:
            return self._constant_index(bound, values)
        if isinstance(value, Constant) and type(value.value) is int:
            return value.value
        if isinstance(value, Var) and id(value) in values:
            return values[id(value)]
        if isinstance(value, Call) and isinstance(value.target, MeshCoord):
            known = values.get(id(value))
            if known is not None:
                return known
        if isinstance(value, Call) and isinstance(value.target, HirBinary):
            lhs, rhs = (self._constant_index(arg, values) for arg in value.args)
            operations = {
                BinaryKind.ADD: lambda: lhs + rhs,
                BinaryKind.SUB: lambda: lhs - rhs,
                BinaryKind.MUL: lambda: lhs * rhs,
                BinaryKind.FLOOR_DIV: lambda: lhs // rhs,
                BinaryKind.MOD: lambda: lhs % rhs,
            }
            operation = operations.get(value.target.kind)
            if operation is not None:
                return operation()
        raise LoweringError(f"output window start {value!r} is not statically enumerable")

    def _declare(self, expr: Call, type_: object, buffers: int, cursor: _Cursor) -> Expr:
        known = self.memo.get(id(expr))
        if known is not None:
            return known
        if not isinstance(type_, TensorType):
            raise LoweringError(f"{_label(expr)} has non-tensor material result {type_!r}")
        storage_type = _storage_type(type_)
        stem = (
            self.names.fresh("value")
            if isinstance(expr.target, HirCast)
            else self.names.binding(expr)
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
        self._prepare_buffers(region, inner)
        self.current_mesh.append(region.mesh)
        result = self.lower(region.body, inner)
        self.current_mesh.pop()
        built = inner.build()
        if built.body:
            cursor.add(
                MeshScope(
                    self._physical_frame(region.mesh),
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
        self._prepare_buffers(loop, inner)
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
        if not (isinstance(destination, Call) and isinstance(destination.target, Zeros)):
            self.lower(destination, cursor)
        update = self.lower(call.args[1], cursor)
        if id(self._material_root(call.args[1])) in self.output_windows:
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
        self.type_ctx.current_mesh = mesh
        try:
            plan = issue_plan(call, self.type_ctx)
        except (TypeError, ValueError) as error:
            raise LoweringError(f"{_label(call)} cannot derive an issue plan: {error}") from error

        params = self._instruction_params(call.target.op)
        reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
        writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
        produced = tuple(param for param in writes if not param.effect & MemoryEffect.READ)
        output_window = self.output_windows.get(id(call))
        if produced and output_window is None and id(call) not in self.memo:
            if call.target.buffers != 1:
                raise LoweringError(f"{_label(call)} staged buffer has no owning scope")
            self._declare(call, call.type, 1, cursor)
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
                if written is None and output_window is None:
                    raise LoweringError(f"{_label(call)} has no storage for {param.name}")
                value = written
            operands.append((param.name, value, desired))

        if getattr(call.target.op, "atom", None) is not None:
            if any(value is None for _, value, _ in operands):
                raise LoweringError(f"{_label(call)} cannot issue an atom into an output window")
            participant = self._issuer_frame(
                call.target.op, tuple(desired for _, _, desired in operands), mesh
            )
            for frame, group_lows in self._atom_groups(call, plan, mesh, participant):
                statement = self._emit_atom_axes(
                    call, plan, tuple(operands), frame, group_lows, 0, {}, {}
                )
                cursor.add(
                    MeshScope(
                        frame,
                        Var(self.names.fresh("threads"), type=_BINDING),
                        statement,
                    )
                )
        else:
            frame = self._issuer_frame(
                call.target.op, tuple(desired for _, _, desired in operands), mesh
            )
            issue = _Cursor()
            issued_args = []
            for role, value, desired in operands:
                if value is None:
                    if output_window is None:
                        raise LoweringError(f"{_label(call)} has no output window")
                    value = self._output_window(output_window, desired, issue)
                    self.memo[id(call)] = value
                    self.logical[id(value)] = call.type
                    written = value
                issued_args.append(
                    self._whole_operand(value, desired, frame, issue, f"{role}_frame")
                )
            issue.add(Evaluate(type(call.target.op)(), tuple(issued_args)))
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

    def _atom_groups(
        self,
        call: Call,
        plan: IssuePlan,
        mesh: Mesh | None,
        participant: Mesh,
    ) -> tuple[tuple[Mesh, dict[str, int]], ...]:
        grouped = tuple(axis for axis in plan.axes if axis.is_group)
        if not grouped:
            return ((participant, {}),)
        if mesh is None:
            raise LoweringError(f"{_label(call)} has group axes but no physical mesh")
        physical_mesh = next(
            (
                layout.mesh
                for argument in call.args
                if isinstance((layout := getattr(argument.type, "layout", None)), ShardLayout)
            ),
            mesh,
        )
        physical = flatten(physical_mesh.layout)
        local = flatten(participant.layout)
        if (
            not isinstance(physical, Layout)
            or physical.strides is None
            or not isinstance(local, Layout)
        ):
            raise LoweringError(f"{_label(call)} group axes need a static strided physical mesh")
        shape = tuple(flatten(physical.shape))
        strides = tuple(flatten(physical.strides))
        local_shape = tuple(flatten(local.shape))
        outer_rank = len(shape) - len(local_shape)
        expected = tuple(axis.repeat for axis in grouped)
        if (
            outer_rank != len(grouped)
            or tuple(shape[outer_rank:]) != local_shape
            or tuple(shape[:outer_rank]) != expected
        ):
            raise LoweringError(
                f"{_label(call)} has {len(grouped)} group axis/axes {expected}, but physical "
                f"mesh {shape} is not outer modes followed by participant frame {local_shape}"
            )
        base = starts(physical_mesh)[0]
        groups = []
        for coordinates in product(*(range(axis.repeat) for axis in grouped)):
            offset = base + sum(
                coordinate * strides[index] for index, coordinate in enumerate(coordinates)
            )
            layout = Layout(shape[outer_rank:], strides[outer_rank:])
            frame = self._physical_frame(
                Mesh(
                    physical_mesh.topologies,
                    ComposedLayout(None, offset, layout),
                    tuple(f"d{index}" for index in range(len(local_shape))),
                )
            )
            lows = {
                axis.name: coordinate * axis.atom
                for axis, coordinate in zip(grouped, coordinates, strict=True)
            }
            groups.append((frame, lows))
        return tuple(groups)

    def _emit_atom_axes(
        self,
        call: Call,
        plan: IssuePlan,
        operands: tuple[tuple[str, Expr, TensorType], ...],
        frame: Mesh,
        group_lows: dict[str, int],
        depth: int,
        offsets: dict[str, Expr],
        slices: dict[str, int],
    ) -> Sequential:
        if depth == len(plan.order):
            return self._emit_atom_call(call, plan, operands, frame, offsets, slices)
        axis = plan.axes[plan.order[depth]]
        low = group_lows.get(axis.name, 0)
        stop = low + axis.atom if axis.is_group else axis.extent
        extent = stop - low
        if extent % axis.atom:
            raise LoweringError(
                f"{_label(call)} axis {axis.name} extent {extent} is not divisible by "
                f"atom {axis.atom}"
            )
        beside = min(axis.row_copies, extent // axis.atom)
        if extent % (axis.atom * beside):
            raise LoweringError(
                f"{_label(call)} axis {axis.name} extent {extent} is not divisible by "
                f"row {axis.atom} * {beside}"
            )
        counter = Var(self.names.fresh(f"o_{axis.name}"), type=_INDEX)
        body = _Cursor()
        for copy in range(beside):
            start = counter if copy == 0 else simplify_dim(DimAdd, (counter, copy * axis.atom))
            nested = self._emit_atom_axes(
                call,
                plan,
                operands,
                frame,
                group_lows,
                depth + 1,
                {**offsets, axis.name: start},
                {**slices, axis.name: copy},
            )
            for statement in nested.body:
                body.add(statement)
        return Sequential(
            (
                For(
                    counter,
                    i64_const(low),
                    i64_const(stop),
                    i64_const(axis.atom * beside),
                    body.build(),
                ),
            )
        )

    def _emit_atom_call(
        self,
        call: Call,
        plan: IssuePlan,
        operands: tuple[tuple[str, Expr, TensorType], ...],
        frame: Mesh,
        offsets: dict[str, Expr],
        slices: dict[str, int],
    ) -> Sequential:
        issue = _Cursor()
        issued_args = []
        atoms = {axis.name: axis.atom for axis in plan.axes}
        for (role, value, desired), axes, row_axes in zip(
            operands, plan.operand_axes, plan.operand_rows, strict=True
        ):
            if any(axis is None for axis in axes):
                raise LoweringError(
                    f"{_label(call)} operand {role!r} is not a projection of work axes"
                )
            mapped = tuple(axis for axis in axes if axis is not None)
            starts_ = tuple(offsets[axis] for axis in mapped)
            shift = sum(slices.get(axis, 0) * atoms[axis] for axis in row_axes)
            issued_args.append(
                self._window(
                    value,
                    starts_,
                    tuple(desired.shape),
                    _with_frame(self._shift_fragment(desired, shift), frame),
                    issue,
                    f"{role}_view",
                )
            )
        atom = call.target.op.atom.on(frame)
        issue.add(Evaluate(type(call.target.op)(atom=atom), tuple(issued_args)))
        return issue.build()

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

    def _issuer_frame(self, op, types: tuple[TensorType, ...], lexical: Mesh | None) -> Mesh:
        for type_ in types:
            layout = type_.layout
            if isinstance(layout, ShardLayout):
                return self._physical_frame(layout.mesh)
        if lexical is None:
            raise LoweringError("instruction issue has no lexical or sharded physical mesh")
        pattern = _scope_pattern(op)
        return self._physical_frame(
            lexical if pattern is None else _squeezed_scope(lexical, pattern)
        )

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
        frame = self._physical_frame(mesh)
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
