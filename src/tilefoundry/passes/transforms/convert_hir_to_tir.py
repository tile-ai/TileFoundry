"""Lower scheduled HIR to effect-form TIR.

The pass analyzes a checked program and lowers each retained definition once.
Memory metadata owns storage addresses; registered access relations own
instruction issue geometry. This file only turns those two facts into statements.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from math import prod

import isl

from tilefoundry.analysis import MemoryMetadata, analyze
from tilefoundry.analysis.access import BufferAlias, SSAValue
from tilefoundry.analysis.check import check_program
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
from tilefoundry.ir.hir.schedule import ScheduleOp, instruction_schema, operand_relations
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.specialize import origin_of
from tilefoundry.ir.hir.tensor.empty import Empty
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.zeros import Zeros
from tilefoundry.ir.isl_interop import isl_to_dim
from tilefoundry.ir.tir.async_copy import CopyAsync, is_indexed_copy
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
from tilefoundry.ir.types.layout import flatten, resolve_layout
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.ir.types.utils import (
    i64_const,
    participant_layout,
    static_dim_value,
)
from tilefoundry.ir.visitor import collect_exprs, expr_children
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.visitor_registry.access_relation import (
    iteration_universe,
    projected_axes,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.call_frame import CallFrame, ValuationVisitor
from tilefoundry.visitor_registry.candidates import (
    candidate_operands,
    candidate_ops,
    sole_candidate,
)
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.instruction import ScheduleRefusal, check_call
from tilefoundry.visitor_registry.registries import typeinfer_registry

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


@dataclass(frozen=True, eq=False)
class _View:
    """An address recipe, materialized only in its consuming scope."""

    base: Expr | _View
    starts: tuple[Expr, ...]
    sizes: tuple
    desired: TensorType
    stem: str = "tile"
    whole_tensor: bool = False

    @property
    def type(self) -> TensorType:
        return self.desired


class _Scope:
    """Append statements while making a let binding contain later entries."""

    def __init__(self, frame: Mesh | None = None, owner: Expr | None = None) -> None:
        self.frame = frame
        self.owner = owner
        self.bound: dict[int, Var] = {}
        self.views: dict[int, tuple[_View, Var]] = {}
        self._root: list = []
        self._block = self._root

    def add(self, stmt) -> None:
        self._block.append(("stmt", stmt))

    def bind(self, var: Var, value: Expr) -> Var:
        inner: list = []
        self._block.append(("let", var, value, inner))
        self._block = inner
        self.bound[id(var)] = var
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


class LoweringContext:
    """Own the open lexical scopes and their statement destinations."""

    def __init__(self, function: Function) -> None:
        self.blocks = [_Scope(owner=function)]
        self.loops = []

    @property
    def current(self) -> _Scope:
        return self.blocks[-1]

    @property
    def frame(self) -> Mesh | None:
        return self.current.frame

    def bind(self, var: Var, value: Expr) -> Var:
        return self.current.bind(var, value)

    def add(self, stmt) -> None:
        self.current.add(stmt)

    @contextmanager
    def declaration_scope(self, owner: Expr | None = None):
        scope = (
            self.current
            if owner is None
            else next((scope for scope in self.blocks if scope.owner is owner), None)
        )
        if scope is None:
            raise LoweringError("staged buffer has no owning scope")
        self.blocks.append(scope)
        try:
            yield scope
        finally:
            self.blocks.pop()

    @contextmanager
    def mesh_scope(self, frame: Mesh, binding: Var, owner: Expr | None = None):
        scope = _Scope(frame, owner)
        scope.bound[id(binding)] = binding
        self.blocks.append(scope)
        try:
            yield scope
        finally:
            self.blocks.pop()
        self.add(MeshScope(frame, binding, scope.build()))

    @contextmanager
    def loop(
        self, induction: Var, start: Expr, stop: Expr, step: Expr, owner: LoopRegion | None = None
    ):
        scope = _Scope(self.frame, owner)
        scope.bound[id(induction)] = induction
        self.blocks.append(scope)
        self.loops.append((induction, start, stop, step))
        try:
            yield scope
        finally:
            self.loops.pop()
            self.blocks.pop()
        self.add(For(induction, start, stop, step, scope.build()))


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
        layout = replace(participant_layout(layout), mesh=frame)
    return replace(type_, layout=layout)


def _label(call: Expr) -> str:
    if isinstance(call, Var):
        return call.name
    binding = get_metadata(call, BindingMetadata)
    return binding.name if binding is not None else type(call.target).__name__


@dataclass
class Lowering(ValuationVisitor[Expr | _View]):
    """Mechanical lowering of one analyzed HIR function."""

    context: TypeInferContext
    authored: Function
    callees: dict[int, Lowering] = field(default_factory=dict)

    @property
    def module(self):
        return self.context.frame.module

    @property
    def function(self):
        return self.context.frame.function

    def visit(self, expr, ctx=None):
        if ctx is None:
            ctx = self.locations.get(id(expr), self.context)
            if isinstance(expr, (MeshRegion, LoopRegion)):
                ctx = ctx.parent
        return super().visit(expr, ctx)

    def _carry_initial(self, value):
        for loop in self.locations.get(id(value), self.context).enclosing_loops():
            for param, initial in zip(loop.params[:len(loop.yield_values)], loop.args):
                if param is value:
                    return initial
        return None

    def __post_init__(self) -> None:
        super().__init__()
        self.names = Names(self.authored)
        self.logical: dict[int, TensorType] = {}
        self.staged: dict[int, tuple[Var, tuple[LoopRegion, ...]]] = {}
        self.frames: list[Mesh] = []
        self.outputs: list[tuple[tuple[int, ...], Var]] = []
        self.output_storage: dict[int, Var] = {}
        self.scratch: list[Var] = []
        self.ctx = LoweringContext(self.function)
        values = {id(value): value for value in collect_exprs(self.function.body)}
        calls = {key: value for key, value in values.items() if isinstance(value, Call)}
        self.output_windows = {
            id(call): memory.alias
            for call in calls.values()
            if (memory := get_metadata(call, MemoryMetadata)) is not None
            and memory.alias is not None
        }
        self.authored_values: dict[int, Expr] = {}
        if self.function.body is not None and self.authored.body is not None:
            self.authored_values[id(self.function.body)] = self.authored.body
        self.locations = self.index_contexts(self.function, self.context)
        self.output_ties = {}
        for value in values.values():
            memory = get_metadata(value, MemoryMetadata)
            alias = None if memory is None else memory.alias
            if (
                alias is not None
                and self._carry_initial(alias.target.expr) is not None
                and self._carry_initial(value) is not alias.target.expr
                and alias.relation.relation.is_identity()
                and getattr(value.type, "shape", None) == alias.target.expr.type.shape
            ):
                root = self._return_storage(value, (), aliases=False)
                region = self.locations.get(id(root))
                loops = () if region is None else region.enclosing_loops()
                if any(any(alias.target.expr is param for param in loop.params) for loop in loops):
                    self.output_ties[id(root)] = alias
                self.output_ties[id(value)] = alias
        self.output_windows = {
            key: alias for key, alias in self.output_windows.items() if key not in self.output_ties
        }
        self.register_scopes = self._register_scope_plan()
        for param in self.function.params:
            physical = Var(param.name, type=param.type, is_const=param.is_const)
            self._memo[id(param)] = (param, physical)
            if isinstance(param.type, TensorType):
                self.logical[id(physical)] = param.type

    def run(self) -> PrimFunction:
        if self.function.body is None:
            raise LoweringError(f"{self.function.name!r} has no body to lower")
        roots = []
        seen = set()
        for path, type_ in self._return_leaves(self.function.return_type):
            if not isinstance(type_, TensorType):
                raise LoweringError(f"return leaf {path!r} has non-tensor type {type_!r}")
            root = self._return_storage(self.function.body, path)
            if root is not None:
                if id(root) in seen:
                    raise LoweringError(f"return leaf {path!r} shares storage with another output")
                seen.add(id(root))
            roots.append((path, type_, root))
        for path, type_, root in roots:
            output_type = (
                replace(type_, storage=StorageKind.GMEM)
                if type_.storage is StorageKind.UMAT
                else type_
            )
            output = Var(self.names.fresh("out"), type=output_type)
            self.outputs.append((path, output))
            if (
                isinstance(root, Call)
                and (
                    isinstance(root.target, (Zeros, Empty))
                    or self.callee_frame(root, self.context.frame) is not None
                )
                and getattr(root.target, "buffers", 1) == 1
                and replace(root.type, layout=resolve_layout(root.type.layout, root.type.shape))
                == replace(
                    output_type, layout=resolve_layout(output_type.layout, output_type.shape)
                )
                and id(root) not in self.output_storage
            ):
                self.output_storage[id(root)] = output
            else:
                raise LoweringError(
                    f"return leaf {path!r} cannot be placed on its output: "
                    f"{_label(root) if root is not None else None}"
                )
        body = self.function.body
        result = self.visit(body)
        if not isinstance(body, MeshRegion):
            self._finish(result)
        statements = self.ctx.blocks[0].build()

        params = tuple(self._memo[id(param)][1] for param in self.function.params)
        return PrimFunction(
            self.function.name,
            (*params, *self.scratch, *(output for _path, output in self.outputs)),
            statements,
            output_count=len(self.outputs),
            target=self.context.frame.target,
        )

    def _staging_owner(self, expr: Call) -> Expr:
        scope = self.locations.get(id(expr))
        outermost = None
        while scope is not None:
            if isinstance(scope.owner, LoopRegion):
                outermost = scope
            scope = scope.parent
        return (
            self.function
            if outermost is None or outermost.parent is None
            else outermost.parent.owner
        )

    def _staging_loops(self, expr: Call) -> tuple[LoopRegion, ...]:
        scope = self.locations.get(id(expr))
        return () if scope is None else scope.enclosing_loops()

    @staticmethod
    def _return_leaves(type_, path: tuple[int, ...] = ()):
        if isinstance(type_, TupleType):
            for index, field in enumerate(type_.fields):
                yield from Lowering._return_leaves(field, (*path, index))
        else:
            yield path, type_

    def _return_storage(
        self, value: Expr, path: tuple[int, ...], *, aliases: bool = True
    ) -> Expr | None:
        """Follow the HIR storage edges that lowering already implements."""
        if isinstance(value, MeshRegion):
            return self._return_storage(value.body, path, aliases=aliases)
        if isinstance(value, LoopRegion):
            if not value.yield_values:
                return None
            slot, tail = (0, path) if len(value.yield_values) == 1 else (path[0], path[1:])
            return self._return_storage(value.params[slot], tail, aliases=aliases)
        if isinstance(value, Tuple):
            return (
                self._return_storage(value.elements[path[0]], path[1:], aliases=aliases)
                if path
                else None
            )
        memory = get_metadata(value, MemoryMetadata)
        if aliases and memory is not None and memory.alias is not None:
            return self._return_storage(
                memory.alias.target.expr,
                (*self._alias_path(memory.alias.target), *path),
                aliases=aliases,
            )
        if isinstance(value, Var):
            initial = self._carry_initial(value)
            if initial is not None:
                if (
                    memory is not None
                    and memory.alias is not None
                    and memory.alias.target.expr is initial
                ):
                    return self._return_storage(initial, path, aliases=aliases)
                return value if not path else None
            root = self._capture_root(value)
            return (
                self._return_storage(root, path, aliases=aliases)
                if root is not value
                else value
                if not path
                else None
            )
        if isinstance(value, Call):
            if isinstance(value.target, TupleGetItem):
                index = static_dim_value(value.args[1])
                if index is not None:
                    return self._return_storage(value.args[0], (index, *path), aliases=aliases)
                return None
            slot = aliased_operand(value)
            if slot is not None:
                return self._return_storage(value.args[slot], path, aliases=aliases)
        return value if not path else None

    def _register_scope_plan(self) -> dict[int, Expr]:
        """Place genuinely shared registers at their outermost HIR definition.

        Region results share their body's storage; ties and views follow the
        same storage trace as outputs. Allocation's peak-estimation buffers
        are not physical register sharing and do not determine placement.
        """
        locations = {}
        for key, scope in self.locations.items():
            if key != id(scope.owner) or scope.parent is None:
                continue
            for path, type_ in self._return_leaves(scope.owner.type):
                if not isinstance(type_, TensorType) or type_.storage is not StorageKind.RMEM:
                    continue
                root = self._return_storage(scope.owner, path)
                if root is None:
                    continue
                previous = locations.get(id(root), self.locations.get(id(root)))
                receiver = scope.parent
                if previous is None:
                    locations[id(root)] = receiver
                else:
                    ancestors = {id(ctx.owner) for ctx in previous.ancestors()}
                    while id(receiver.owner) not in ancestors:
                        receiver = receiver.parent
                    locations[id(root)] = receiver
        return {key: scope.owner for key, scope in locations.items()}

    def _declare(
        self,
        expr: Expr,
        type_: object,
        buffers: int,
    ) -> Expr:
        owner = (
            self._staging_owner(expr)
            if buffers > 1
            else self.register_scopes.get(id(expr))
            if isinstance(type_, TensorType) and type_.storage is StorageKind.RMEM
            else None
        )
        with self.ctx.declaration_scope(owner):
            return self._declare_storage(expr, type_, buffers)

    def _declare_storage(self, expr: Expr, type_: object, buffers: int):
        if not isinstance(type_, TensorType):
            raise LoweringError(f"{_label(expr)} has non-tensor material result {type_!r}")
        if type_.storage is StorageKind.RMEM:
            mesh = self._holder_mesh(type_)
            storage_type = type_ if mesh is None else _with_frame(type_, self._physical_frame(mesh))
        else:
            storage_type = _storage_type(type_)
        stem = self.names.binding(
            self.authored_values.get(id(expr)), expr.name if isinstance(expr, Var) else "value"
        )
        if storage_type.storage is StorageKind.RMEM:
            var = Var(stem, type=storage_type)
            self.ctx.bind(
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
            view = _View(scratch, (i64_const(0),), (elements,), desired, stem)
            self.logical[id(view)] = type_
            return view
        fields = tuple(self._literal_view(storage_type, offset) for offset in offsets)
        if len(fields) == 1:
            view = _View(fields[0].args[0], (), tuple(storage_type.shape), storage_type, stem)
            self.logical[id(view)] = type_
            return view
        tuple_type = TupleType(tuple(field.type for field in fields))
        stages = Var(self.names.fresh(f"{stem}_stages"), type=tuple_type)
        self.ctx.bind(stages, Tuple(fields, type=tuple_type))
        self.logical[id(stages)] = type_
        self.staged[id(expr)] = (stages, self._staging_loops(expr))
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
        stages, loops = self.staged[id(expr)]
        index = i64_const(0)
        for depth, loop in enumerate(loops):
            known = self._memo.get(id(loop.induction_var))
            induction = None if known is None else known[1]
            if not isinstance(induction, Var):
                raise LoweringError(f"{_label(expr)} staged selection has no owning induction")
            start = static_dim_value(loop.start)
            stop = static_dim_value(loop.extent)
            step = static_dim_value(loop.step)
            if step is None:
                raise LoweringError(
                    f"{_label(expr)} staged selection needs a literal step for "
                    f"{loop.induction_var.name!r}"
                )
            if depth:
                if start is None or stop is None:
                    raise LoweringError(
                        f"{_label(expr)} staged selection needs a constant trip count for "
                        f"{loop.induction_var.name!r}"
                    )
                trips = max(0, (stop - start + step - 1) // step)
                index = simplify_dim(DimMul, (index, trips))
            moved = (
                induction
                if start == 0
                else simplify_dim(DimSub, (induction, self._dim(loop.start)))
            )
            ordinal = moved if step == 1 else simplify_dim(DimFloorDiv, (moved, step))
            index = ordinal if depth == 0 else simplify_dim(DimAdd, (index, ordinal))
        if loops:
            index = simplify_dim(DimMod, (index, len(stages.type.fields)))
        field = stages.type.fields[0]
        return Call(TupleGetItem(), (stages, index), type=field)

    def _known(self, expr: Expr) -> Expr:
        known = self._memo.get(id(expr))
        return expr if known is None else known[1]

    def _bind_region_args(self, region: LoopRegion | MeshRegion) -> None:
        values = tuple(self.visit(arg) for arg in region.args)
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

    def dispatch_visit(self, expr: Expr, ctx=None):
        self._pair_authored(expr)
        return super().dispatch_visit(expr, ctx)

    def visit_Constant(self, expr: Constant, _ctx=None) -> Expr:
        return expr

    def visit_Var(self, expr: Var, _ctx=None) -> Expr:
        raise LoweringError(f"unbound HIR value {expr.name!r} reached lowering")

    def visit_Tuple(self, expr: Tuple, _ctx=None) -> Expr:
        elements = tuple(self.visit(element) for element in expr.elements)
        return Tuple(elements, type=TupleType(tuple(element.type for element in elements)))

    def _check_escape(self, value, scope: _Scope, region: Expr) -> None:
        def free_vars(value):
            if isinstance(value, Var):
                yield value
            elif isinstance(value, _View):
                yield from free_vars(value.base)
                for start in value.starts:
                    yield from free_vars(start)
            elif isinstance(value, Expr):
                for child in expr_children(value):
                    yield from free_vars(child)

        for var in free_vars(value):
            if id(var) in scope.bound:
                mesh = region.mesh if isinstance(region, MeshRegion) else scope.frame
                raise LoweringError(f"value {var.name!r} escapes region mesh {mesh!r}")

    def visit_MeshRegion(self, region: MeshRegion, ctx):
        self._bind_region_args(region)
        frame = self._physical_frame(region.mesh)
        binding = Var(
            self.names.fresh("cta" if region is self.function.body else "scope"), type=_BINDING
        )
        with self.ctx.mesh_scope(frame, binding, region) as scope:
            result = self.visit(region.body, self.region_context(region, ctx))
            self._check_escape(result, scope, region)
            if region is self.function.body:
                self._finish(result)
        return result

    def visit_LoopRegion(self, loop: LoopRegion, ctx) -> Expr:
        self._bind_region_args(loop)
        carried_values = []
        for slot, param in enumerate(loop.params[: len(loop.yield_values)]):
            memory = get_metadata(param, MemoryMetadata)
            if memory is not None and memory.alias is not None:
                if memory.alias.target.expr is not loop.args[slot]:
                    raise LoweringError(
                        f"loop parameter {param.name!r} has an alias that is not a carry tie"
                    )
                carried = self._alias_value(param, memory.alias.target)
            else:
                tie = self.output_ties.get(id(param))
                carried = (
                    self._declare(param, param.type, 1)
                    if tie is None
                    else self._alias_value(param, tie.target)
                )
                self._emit_copy(self.visit(loop.args[slot]), carried)
            self._memo[id(param)] = (param, carried)
            carried_values.append(carried)
        init = tuple(carried_values)
        induction = Var(loop.induction_var.name, type=_INDEX)
        self._memo[id(loop.induction_var)] = (loop.induction_var, induction)
        with self.ctx.loop(
            induction, self._dim(loop.start), self._dim(loop.extent), self._dim(loop.step), loop
        ) as scope:
            inner = self.region_context(loop, ctx)
            self.visit(loop.body, inner)
            yielded = tuple(self.visit(value, inner) for value in loop.yield_values)
            for param, source, carried, value in zip(
                loop.params, loop.yield_values, init, yielded, strict=False
            ):
                memory = get_metadata(source, MemoryMetadata)
                tied = (
                    memory is not None
                    and memory.alias is not None
                    and self._capture_root(memory.alias.target.expr) is param
                )
                if not tied:
                    self._emit_copy(value, carried)
        result = (
            init[0]
            if len(init) == 1
            else Tuple(init, type=TupleType(tuple(value.type for value in init)))
        )
        self._check_escape(result, scope, loop)
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

    def visit_Call(self, call: Call, _ctx=None) -> Expr:
        target = call.target
        if self.callee_frame(call, self.context.frame) is not None:
            return self._lower_call(call)
        if is_dim_op_call(call):
            result = self._dim(call)
        else:
            visitor = getattr(self, f"visit_{type(target).__name__}", None)
            if visitor is None:
                result = self._lower_automatic_instruction(call)
            else:
                result = visitor(call)
        return result

    def _lower_call(self, call: Call) -> Expr | _View:
        frame = self.callee_frame(call, self.context.frame)
        lowering = self.callees.get(id(frame.function))
        if lowering is None:
            region = self.locations[id(call)]
            lowering = Lowering(
                TypeInferContext(frame=frame, owner=frame.function, current_mesh=region.current_mesh),
                origin_of(frame.function) or frame.function,
                self.callees,
            )
            lowering.lowered = lowering.run()
            self.callees[id(frame.function)] = lowering
        args = tuple(self.visit(arg) for arg in call.args)
        destination = self.output_storage.get(id(call))
        if destination is None:
            destination = self._declare_result(call, 1)
        self.logical[id(destination)] = call.type
        scratch = []
        for param in lowering.scratch:
            forwarded = Var(self.names.fresh(param.name), type=param.type)
            self.scratch.append(forwarded)
            scratch.append(forwarded)
        target = call.target
        if isinstance(target, Function):
            target = lowering.lowered
        elif isinstance(target, Call):
            target = replace(target, args=(target.args[0], self._dim(target.args[1])))
        params = tuple(
            param for param in lowering.lowered.params
            if isinstance(call.target, Function) or not param.is_const
        )
        passed = []
        for value, param in zip((*args, *scratch, destination), params, strict=True):
            if isinstance(param.type, TensorType) and replace(
                value.type, layout=resolve_layout(value.type.layout, value.type.shape)
            ) != replace(param.type, layout=resolve_layout(param.type.layout, param.type.shape)):
                value = _View(
                    value, tuple(i64_const(0) for _ in value.type.shape), tuple(value.type.shape),
                    param.type, "argument", whole_tensor=True,
                )
            passed.append(self._operand(value))
        self.ctx.add(Evaluate(target, tuple(passed)))
        return destination

    def _lower_automatic_instruction(self, call: Call) -> Expr:
        target = call.target
        candidates = candidate_ops(type(target))
        if not candidates:
            raise LoweringError(f"{_label(call)} uses unknown HIR call {type(target).__name__}")
        op = sole_candidate(call)
        if op is None:
            available = ", ".join(op_identifier(candidate) for candidate in candidates)
            raise LoweringError(
                f"{_label(call)} has instruction candidates [{available}] but no automatic "
                "selection; write tf.schedule to choose the instruction and its attributes"
            )
        if not isinstance(call.type, TensorType):
            raise LoweringError(f"{_label(call)} candidate result is not a tensor")
        _attributes, types = candidate_operands(call, type(op))
        params = self._instruction_params(op, len(types))
        reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
        operands = []
        for param, arg in zip(reads, call.args, strict=True):
            desired = types[params.index(param)]
            value = self.visit(arg)
            if desired != arg.type:
                value = _View(
                    value,
                    tuple(i64_const(0) for _ in value.type.shape),
                    tuple(value.type.shape),
                    desired,
                )
            operands.append(value)
        return self._lower_instruction(call, op, operands=tuple(operands))

    def visit_MeshCoord(self, call: Call) -> Expr:
        return self._dim(call)

    def visit_Binary(self, call: Call) -> Expr:
        if isinstance(call.type, TensorType) and call.type.shape == ():
            return self._dim(call)
        return self._lower_automatic_instruction(call)

    def visit_TupleGetItem(self, call: Call) -> Expr:
        source = self.visit(call.args[0])
        index = self._dim(call.args[1])
        if isinstance(source, Tuple) and isinstance(index, Constant):
            return source.elements[index.value]
        return Call(call.target, (source, index), type=call.type)

    def visit_Slice(self, call: Call):
        base = self.visit(call.args[0])
        starts_arg = call.args[1]
        starts = (
            tuple(self._window_start(value) for value in starts_arg.elements)
            if isinstance(starts_arg, Tuple)
            else (self._window_start(starts_arg),)
        )
        if any(stride != 1 for stride in call.target.strides):
            raise LoweringError(f"{_label(call)} has a strided Slice, which is not contiguous")
        return _View(base, starts, tuple(call.target.sizes), call.type)

    def visit_Reshape(self, call: Call):
        source = self.visit(call.args[0])
        return _View(
            source,
            tuple(i64_const(0) for _ in source.type.shape),
            tuple(source.type.shape),
            call.type,
        )

    def visit_Bitcast(self, call: Call):
        return self.visit_Reshape(call)

    def visit_Reshard(self, call: Call):
        if aliased_operand(call) is None:
            return self._lower_automatic_instruction(call)
        return self.visit_Reshape(call)

    def visit_Transpose(self, call: Call) -> Expr | _View:
        return self._lower_automatic_instruction(call)

    def _declare_result(self, call: Call, buffers: int) -> Expr:
        tie = self.output_ties.get(id(call))
        if tie is not None:
            return self._alias_value(call, tie.target)
        if call.type.storage is StorageKind.RMEM and buffers != 1:
            raise LoweringError(f"{_label(call)} staged buffer has no owning scope")
        return self._declare(call, call.type, buffers)

    def visit_Empty(self, call: Call) -> Expr:
        placed = self.output_storage.get(id(call))
        if placed is not None:
            self.logical[id(placed)] = call.type
            return placed
        return self._declare_result(call, call.target.buffers)

    def _bind_window_alias(self, producer: Call, window: BufferAlias) -> None:
        self._alias_value(producer, window.target)
        for value in window.relation.values.values():
            if (
                isinstance(value, Expr)
                and not isinstance(value, Constant)
                and not (isinstance(value, Call) and isinstance(value.target, MeshCoord))
            ):
                self._alias_value(producer, value)

    def visit_Zeros(self, call: Call) -> Expr | _View:
        alias = self.output_windows.get(id(call))
        if alias is not None:
            window = self._output_window(alias, call.type)
            self.logical[id(window)] = call.type
            self._emit_fill(window, call.type)
            return window
        result = self.output_storage.get(id(call))
        if result is None:
            result = self._declare_result(call, 1)
        else:
            self.logical[id(result)] = call.type
        self._emit_fill(result, call.type)
        return result

    def _window_start(self, value: Expr) -> Expr:
        if isinstance(value, Call) and isinstance(value.target, HirBinary):
            operation = _DIM_BINARY.get(value.target.kind)
            if operation is not None:
                return simplify_dim(operation, tuple(self._window_start(arg) for arg in value.args))
        return (
            self.visit(value)
            if isinstance(value, Call) and not is_dim_op_call(value)
            else self._dim(value)
        )

    def visit_IndexSelect(self, call: Call) -> Expr:
        raise LoweringError(
            f"{_label(call)} is an unscheduled IndexSelect; write "
            "tf.schedule((x, tf.empty(...), idx), op=T.copy_async(fill=...))"
        )

    def visit_InsertSlice(self, call: Call) -> Expr:
        target = self.visit(call.args[0])
        update = self.visit(call.args[1])
        memory = get_metadata(call, MemoryMetadata)
        tied = (
            memory is not None
            and memory.alias is not None
            and self._capture_root(memory.alias.target.expr) is self._capture_root(call.args[0])
        )
        result = target if tied else self._declare_result(call, 1)
        if not tied:
            self._emit_copy(target, result)
        update_source = self._capture_root(call.args[1])
        if isinstance(update_source, Call) and isinstance(update_source.target, ScheduleOp):
            slot = aliased_operand(update_source)
            update_source = self._capture_root(update_source.args[slot])
        root = self._return_storage(update_source, (), aliases=False)
        if root is not None:
            update_source = root
        memory = get_metadata(update_source, MemoryMetadata)
        destination = self._capture_root(call.args[0])
        if (
            tied
            and memory is not None
            and memory.alias is not None
            and memory.alias.target.expr is destination
        ):
            return result
        starts, desired = self._insert_window(call, result, update.type)
        self._emit_copy(
            update, result, starts=starts, sizes=tuple(update.type.shape), desired=desired
        )
        return result

    def _capture_root(self, value: Expr) -> Expr:
        return self.locations.get(id(value), self.context).capture_root(value)

    def _insert_window(self, write: Call, target: Expr, desired: TensorType):
        offsets = write.args[2]
        starts = (
            tuple(self._window_start(value) for value in offsets.elements)
            if isinstance(offsets, Tuple)
            else (
                self._window_start(offsets),
                *(i64_const(0) for _ in target.type.shape[1:]),
            )
        )
        return starts, self._window_type(target, starts, desired)

    @staticmethod
    def _window_type(target: Expr, starts: tuple[Expr, ...], desired: TensorType) -> TensorType:
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
        return replace(desired, layout=type_.layout, storage=target.type.storage)

    def _insert_target(
        self,
        target: Expr,
        starts: tuple[Expr, ...],
        sizes: tuple,
        desired: TensorType,
    ) -> Expr:
        return self._bind_window(
            target,
            starts,
            sizes,
            desired,
            "window",
        )

    def visit_ScheduleOp(self, call: Call) -> Expr:
        return self._lower_instruction(call, call.target.op)

    def _lower_instruction(self, call: Call, op, *, operands=None) -> Expr:
        scope = self.locations.get(id(call))
        mesh = scope.enclosing_mesh() if scope is not None else None
        if isinstance(call.target, ScheduleOp):
            types = tuple(arg.type for arg in call.args)
            repeat, order = call.target.repeat, call.target.order
        else:
            _attributes, types = candidate_operands(call, type(op))
            repeat = order = None
        try:
            _bindings, projection = check_call(
                op, types, mesh, frame=scope.frame, repeat=repeat, order=order
            )
        except ScheduleRefusal as error:
            raise LoweringError(str(error)) from error
        params = projection.params
        reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
        writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
        produced = tuple(param for param in writes if not param.effect & MemoryEffect.READ)
        output_window = self.output_windows.get(id(call))
        destination_operand = None
        if isinstance(call.target, ScheduleOp) and produced:
            destination_operand = call.args[params.index(produced[0])]
            output_window = self.output_windows.get(id(destination_operand))
        if output_window is not None:
            self._bind_window_alias(call, output_window)
        written = None
        if produced and output_window is None and not isinstance(call.target, ScheduleOp):
            written = self._declare_result(call, 1)
        if destination_operand is not None and output_window is None:
            tie = self.output_ties.get(id(call))
            written = (
                self.visit(destination_operand) if tie is None else self._alias_value(call, tie.target)
            )
            if tie is not None:
                self._memo[id(destination_operand)] = (destination_operand, written)
        read_values = {}
        supplied = params if isinstance(call.target, ScheduleOp) else reads
        for param, operand in zip(
            supplied, call.args if operands is None else operands, strict=True
        ):
            if param.effect == MemoryEffect.WRITE and output_window is not None:
                read_values[param.name] = None
                continue
            value = (
                written
                if operand is destination_operand
                else self.visit(operand)
                if operands is None
                else operand
            )
            if (
                isinstance(op, CopyAsync)
                and is_indexed_copy(call.args)
                and param.name == "src"
                and (
                    value.type.layout is None
                    or any(
                        value is self._memo[id(argument)][1] for argument in self.function.params
                    )
                )
            ):
                type_ = replace(
                    value.type,
                    layout=resolve_layout(value.type.layout, value.type.shape),
                )
                value = _View(
                    value,
                    tuple(i64_const(0) for _ in value.type.shape),
                    tuple(value.type.shape),
                    type_,
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
                with self.ctx.declaration_scope():
                    self.ctx.bind(
                        scalar, Call(AllocTensor(tensor_type=scalar_type), (), type=scalar_type)
                    )
                self.ctx.add(Evaluate(Fill(), (scalar, self._operand(value))))
                self.logical[id(scalar)] = scalar_type
                value = scalar
            read_values[param.name] = value
        operands: list[tuple[str, Expr | None]] = []
        for param in params:
            value = (
                read_values[param.name]
                if isinstance(call.target, ScheduleOp) or param.effect & MemoryEffect.READ
                else written
            )
            if value is None and output_window is None:
                raise LoweringError(f"{_label(call)} has no storage for {param.name}")
            operands.append((param.name, value))

        written = self._emit_instruction(
            call, op, tuple(operands), mesh, output_window, written, projection
        )
        if destination_operand is not None and output_window is not None and written is not None:
            self._memo[id(destination_operand)] = (destination_operand, written)
        result_param = next(param for param in params if param.effect & MemoryEffect.WRITE)
        if result_param.effect & MemoryEffect.READ:
            return read_values[result_param.name]
        if written is None:
            raise LoweringError(f"{_label(call)} produced no addressable result")
        return written

    def _alias_path(self, value: SSAValue) -> tuple[int, ...]:
        return tuple(
            path for path, held in self._return_leaves(value.expr.type) if isinstance(held, TensorType)
        )[value.index]

    def _alias_value(self, producer: Expr, value: Expr | SSAValue) -> Expr:
        path = self._alias_path(value) if isinstance(value, SSAValue) else ()
        value = value.expr if isinstance(value, SSAValue) else value
        known = self._memo.get(id(value))
        if known is None:
            raise LoweringError(f"{_label(producer)} alias requires lowered value {_label(value)!r}")
        result = known[1]
        for index in path:
            result = (
                result.elements[index]
                if isinstance(result, Tuple)
                else Call(TupleGetItem(), (result, i64_const(index)), type=result.type.fields[index])
            )
        return result

    def _output_window(self, alias: BufferAlias, desired: TensorType) -> _View:
        target = self._alias_value(alias.target.expr, alias.target)
        origin = alias.relation.relation
        for axis in range(origin.dim(isl.dim_type.IN)):
            origin = origin.fix_val(isl.dim_type.IN, axis, isl.val(0))
        image = origin.range()
        values = {name: self._dim(value) for name, value in alias.relation.values.items()}
        starts = tuple(
            i64_const(start) if isinstance(start, int) else start
            for axis in range(image.dim(isl.dim_type.SET))
            for start in (isl_to_dim(image.dim_min(axis).gist_params(image.params()), values),)
        )
        desired = self._window_type(target, starts, desired)
        return _View(target, starts, tuple(desired.shape), desired, "window")

    def _emit_instruction(self, call, op, operands, mesh, output_window, written, projection):
        atom = getattr(op, "atom", None)
        if atom is not None and output_window is not None:
            raise LoweringError(f"{_label(call)} cannot issue an atom into an output window")
        if atom is None:
            if mesh is None:
                raise LoweringError(f"{_label(call)} instruction issue has no lexical mesh")
            declared = next(
                (
                    held
                    for _role, value in operands
                    if value is not None
                    and (held := self._holder_mesh(self.logical.get(id(value), value.type)))
                    is not None
                ),
                None,
            )
            try:
                frame = None if declared is None else self._physical_frame(declared)
            except ValueError as error:
                raise LoweringError(f"{_label(call)} {error}") from error
            with self._issue_scope(frame):
                frame = self.ctx.frame
                issued = []
                for role, value in operands:
                    if value is None:
                        value = self._output_window(output_window, call.type)
                        self.logical[id(value)], written = call.type, value
                    desired = self.logical.get(id(value), value.type)
                    issued.append(self._whole_operand(value, desired, frame, f"{role}_frame"))
                self.ctx.add(Evaluate(self._issued_op(op), tuple(issued)))
            return written
        if any(value is None for _, value in operands):
            raise LoweringError(f"{_label(call)} cannot issue an atom into an output window")
        logical = tuple(self.logical.get(id(value), value.type) for _, value in operands)
        try:
            relations = operand_relations(op, logical)
            axes = tuple(projected_axes(boundary) for boundary in relations[: len(logical)])
            whole, axis_names = _iteration_geometry(relations)
        except ValueError as error:
            raise LoweringError(f"{_label(call)} {error}") from error
        tile = tuple(
            size * parallel
            for size, parallel in zip(projection.shape, projection.parallel, strict=True)
        )
        rows = [1] * len(tile)
        for row in projection.rows:
            if row is not None:
                axis, count = row
                rows[axis] = max(rows[axis], count)
        row_axes = tuple(None if row is None else row[0] for row in projection.rows)

        def emit(depth, offsets, copies):
            if depth == len(projection.order):
                issued = []
                for (role, value), desired, mapped, row_axis in zip(
                    operands, projection.issued, axes, row_axes, strict=True
                ):
                    shift = 0 if row_axis is None else copies.get(row_axis, 0) * tile[row_axis]
                    desired = self._shift_fragment(desired, shift)
                    logical_type = self.logical.get(id(value), value.type)
                    issued.append(
                        self._whole_operand(value, desired, self.ctx.frame, f"{role}_view")
                        if tuple(logical_type.shape) == tuple(desired.shape)
                        else self._bind_window(
                            value,
                            tuple(offsets[axis] for axis in mapped),
                            desired.shape,
                            _with_frame(desired, self.ctx.frame),
                            f"{role}_view",
                        )
                    )
                self.ctx.add(Evaluate(op, tuple(issued)))
                return
            axis = projection.order[depth]
            beside = min(rows[axis], whole[axis] // tile[axis])
            counter = Var(self.names.fresh(f"o_{axis_names[axis]}"), type=_INDEX)
            with self.ctx.loop(
                counter, i64_const(0), i64_const(whole[axis]), i64_const(tile[axis] * beside)
            ):
                for copy in range(beside):
                    start = (
                        counter if copy == 0 else simplify_dim(DimAdd, (counter, copy * tile[axis]))
                    )
                    emit(depth + 1, {**offsets, axis: start}, {**copies, axis: copy})

        emit(0, {}, {})
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
        try:
            params, _writes = instruction_schema(op, operand_count)
        except ValueError as error:
            raise LoweringError(str(error)) from error
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

    @contextmanager
    def _issue_scope(self, frame: Mesh | None):
        current = self.ctx.frame
        same = (
            current is not None
            and frame is not None
            and replace(current, layout=resolve_layout(current.layout))
            == replace(frame, layout=resolve_layout(frame.layout))
        )
        if frame is None or same:
            yield
        else:
            binding = Var(self.names.fresh("threads"), type=_BINDING)
            with self.ctx.mesh_scope(frame, binding):
                yield

    def _operand(self, value):
        if not isinstance(value, _View):
            return value
        scope = self.ctx.current
        known = scope.views.get(id(value))
        if known is not None:
            return known[1]
        desired = value.desired
        if isinstance(desired.layout, ShardLayout):
            holder = self._holder_mesh(desired)
            frame = next((
                scope.frame for scope in reversed(self.ctx.scopes)
                if scope.frame is not None and scope.frame.topologies == holder.topologies
            ), None)
            if frame is None:
                raise LoweringError(f"a sharded view requires an issuing mesh: {desired.layout!r}")
            desired = _with_frame(desired, frame)
        if isinstance(value.base, Constant) and isinstance(value.base.type, PointerType):
            result = self.ctx.bind(
                Var(value.stem, type=desired),
                self._literal_view(desired, value.base.value),
            )
        else:
            result = self._bind_window(
                value.base,
                value.starts,
                value.sizes,
                desired,
                value.stem,
                whole_tensor=value.whole_tensor,
            )
        scope.views[id(value)] = (value, result)
        self.logical[id(result)] = self.logical.get(id(value), value.desired)
        return result

    def _whole_operand(
        self,
        value: Expr,
        desired: TensorType,
        frame: Mesh,
        stem: str,
    ) -> Expr:
        value = self._operand(value)
        desired = _with_frame(desired, frame)
        actual = value.type
        if isinstance(actual, TensorType) and isinstance(actual.layout, ShardLayout):
            actual = replace(actual, layout=participant_layout(actual.layout))
        if isinstance(actual, TensorType):
            actual = replace(actual, layout=resolve_layout(actual.layout, actual.shape))
        comparable = replace(desired, layout=resolve_layout(desired.layout, desired.shape))
        if actual == comparable or (
            not isinstance(desired.layout, ShardLayout)
            and isinstance(value.type, TensorType)
            and tuple(value.type.shape) == tuple(desired.shape)
            and value.type.dtype == desired.dtype
            and value.type.storage is desired.storage
        ):
            return value
        return self._bind_window(
            value,
            tuple(i64_const(0) for _ in value.type.shape),
            tuple(value.type.shape),
            desired,
            stem,
        )

    def _bind_window(
        self,
        base: Expr,
        starts: tuple[Expr, ...],
        sizes: tuple,
        desired: TensorType,
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
            stem: Prefix for the view's generated name.
            whole_tensor: Runtime table sources avoid unsupported PtrOf(Slice) CUDA (side finding).
        """
        base = self._operand(base)
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
                source,
                TypeInferContext(
                    memo={id(arg): (arg, arg.type) for arg in (base, *starts)},
                    loops=tuple(self.ctx.loops),
                ),
            )
            source.type = getattr(inferred, "type", inferred)
        pointer = Call(PtrOf(), (source,), type=PointerType(held.dtype, held.storage))
        view = Call(
            TensorView(layout=desired.layout, shape=tuple(desired.shape)),
            (pointer,),
            type=desired,
        )
        return self.ctx.bind(Var(self.names.fresh(stem), type=desired), view)

    def _holder_mesh(self, type_: TensorType) -> Mesh | None:
        layout = type_.layout
        return participant_layout(layout).mesh if isinstance(layout, ShardLayout) else None

    def _emit_fill(self, target, logical_type: TensorType) -> None:
        mesh = self._holder_mesh(logical_type)
        frame = None if mesh is None else self._physical_frame(mesh)
        zero = Constant(0.0, type=TensorType.scalar(logical_type.dtype))
        with self._issue_scope(frame):
            written = self._operand(target)
            self.ctx.add(Evaluate(Fill(), (written, zero)))

    def _emit_copy(
        self,
        source,
        target,
        *,
        starts: tuple[Expr, ...] | None = None,
        sizes: tuple | None = None,
        desired: TensorType | None = None,
    ) -> None:
        source_type = self.logical.get(id(source), source.type)
        if not isinstance(source_type, TensorType) or not isinstance(target.type, TensorType):
            raise LoweringError("copy lowering requires tensor operands")
        mesh = self._holder_mesh(source_type) or self._holder_mesh(target.type)
        frame = None if mesh is None else self._physical_frame(mesh)
        with self._issue_scope(frame):
            read = (
                self._operand(source)
                if frame is None
                else self._whole_operand(
                    source, source_type, self.ctx.frame, f"{getattr(source, 'name', 'value')}_view"
                )
            )
            written = self._operand(target)
            if starts is not None:
                if sizes is None or desired is None:
                    raise LoweringError("a windowed copy needs sizes and a destination type")
                if self.ctx.frame is not None:
                    desired = _with_frame(desired, self.ctx.frame)
                written = self._insert_target(written, starts, sizes, desired)
            self.ctx.add(Evaluate(Copy(), (read, written)))

    def _finish(self, result: Expr | _View) -> None:
        for path, output in self.outputs:
            leaf = result
            for index in path:
                leaf = (
                    leaf.elements[index]
                    if isinstance(leaf, Tuple)
                    else Call(
                        TupleGetItem(), (leaf, i64_const(index)), type=leaf.type.fields[index]
                    )
                )
            if leaf is not output:
                raise LoweringError(
                    f"return leaf {path!r} lowered to different storage than its output"
                )


@dataclass
class ConvertHIRToTIR(ModulePass):
    """Replace one selected HIR entry with its scheduled TIR function."""

    entry: str | None = None
    dims: Mapping[str, int] | None = None

    name = "convert-hir-to-tir"
    requires = ()

    def run(self, module: Module) -> Module:
        authored = module.entry_function() if self.entry is None else module.lookup(self.entry)
        if not isinstance(authored, Function):
            raise LoweringError(f"{getattr(authored, 'name', self.entry)!r} is not an HIR function")
        result = analyze(*check_program(module, authored, dims=self.dims), analysis=("memory",))
        lowered = {}
        lowering = Lowering(
            TypeInferContext(frame=CallFrame(result.module, result.function), owner=result.function),
            authored,
            lowered,
        )
        root = lowering.lowered = lowering.run()
        lowered[id(result.function)] = lowering
        root.metadata = result.function.metadata
        for record in render_analysis(result).summary:
            if isinstance(record, (ReportIdentity, ReportSelection)):
                attach_metadata(root, record)
        modules = set()

        def lower_module(node):
            if id(node) in modules:
                return
            modules.add(id(node))
            for slot in node.children.values():
                for child in slot.elements:
                    lower_module(child)
            node.functions = tuple(lowered[id(fn)].lowered if id(fn) in lowered else fn
                                   for fn in node.functions)

        lower_module(result.module)
        return result.module


__all__ = ["ConvertHIRToTIR", "LoweringError"]
