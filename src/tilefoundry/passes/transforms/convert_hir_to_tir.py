"""Lower scheduled HIR to effect-form TIR.

The pass analyzes a checked program and lowers each retained definition once.
Allocation summaries own storage addresses; registered access relations own
instruction issue geometry. This file only turns those two facts into statements.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from math import prod

import isl

from tilefoundry.analysis import MemoryMetadata, analyze
from tilefoundry.analysis.access import AliasKind, SSAValue, name_loop_parameters, referenced_values
from tilefoundry.analysis.allocation import AllocationSummary, _edge, address_alias, root
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.errors import AnalysisError
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
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.isl_interop import isl_to_dim
from tilefoundry.ir.tir.arith import Binary as TirBinary
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
from tilefoundry.ir.types.layout_algebra import composition
from tilefoundry.ir.types.layout_correspondence import layout_correspondence
from tilefoundry.ir.types.shard_layout import shard_layout_of, split_target_axes
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.ir.types.utils import (
    i64_const,
    participant_layout,
    static_dim_value,
)
from tilefoundry.ir.visitor import expr_children
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.utils.isl_utils import as_multi_aff
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    iteration_universe,
    leaves_of,
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


def _layout_addresses(shape, layout):
    flat = flatten(resolve_layout(layout, shape))
    coordinates = [f"c{axis}" for axis in range(len(shape))]
    digits = layout_correspondence(flat.shape, tuple(shape)).coordinates(coordinates)
    linear = " + ".join(f"{stride} * ({digit})" for stride, digit in zip(
        flat.strides, digits, strict=True
    )) or "0"
    offset = layout.offset if isinstance(layout, ComposedLayout) else 0
    return isl.map(f"{{ [{', '.join(coordinates)}] -> [{offset} + {linear}] }}")


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
        result = super().visit(expr, ctx)
        return self._value(expr, result) if SSAValue(expr) in self.addresses else result

    def __post_init__(self) -> None:
        super().__init__()
        self.names = Names(self.authored)
        self.logical: dict[int, TensorType] = {}
        self.staged: dict[int, tuple[Var, tuple[LoopRegion, ...]]] = {}
        self.stage_counters = {}
        self.frames: list[Mesh] = []
        self.outputs: list[tuple[tuple[int, ...], Var]] = []
        self.output_storage: dict[SSAValue, Var] = {}
        self.output_values = {}
        self.scratch: list[Var] = []
        self.ctx = LoweringContext(self.function)
        summary = get_metadata(self.function, AllocationSummary)
        self.addresses = {}
        self.address_precisions = summary.view_precisions
        self.storage = {}
        self.tensors = {}
        self.authored_values: dict[int, Expr] = {}
        if self.function.body is not None and self.authored.body is not None:
            self.authored_values[id(self.function.body)] = self.authored.body
        self.locations = self.index_contexts(self.function, self.context)
        for value in summary.executable_values().values():
            aliases = []
            for alias in value.aliases:
                if value.lifetime.memory_level == "rmem" and alias.kind is AliasKind.IN_PLACE:
                    continue
                relation = alias.relation.relation
                depth = relation.dim(isl.dim_type.IN) - len(leaves_of(value.value.expr.type)[value.value.index].shape)
                if depth:
                    loops = self.locations[id(value.value.expr)].enclosing_loops()[:depth]
                    relation, bindings = name_loop_parameters(relation, (loop.induction_var for loop in loops))
                    relation = relation.project_out(isl.dim_type.IN, 0, depth)
                    relation = relation.gist_params(relation.params())
                    alias = replace(alias, relation=AccessRelation(relation, referenced_values(
                        relation, {**alias.relation.values, **bindings}
                    )))
                aliases.append(alias)
            self.addresses[value.value] = replace(value, aliases=tuple(aliases))
        self.register_scopes = self._register_scope_plan()
        self.parameters = {}
        for param in self.function.params:
            physical = Var(param.name, type=param.type, is_const=param.is_const)
            self.parameters[id(param)] = physical
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
            key = SSAValue(self.function.body, len(roots))
            if key not in self.addresses:
                raise LoweringError(f"return leaf {path!r} cannot be placed on its output: None")
            address = self._address(key.expr, key.index)
            if address.target in seen:
                raise LoweringError(f"return leaf {path!r} shares storage with another output")
            seen.add(address.target)
            roots.append((path, type_, key, address))
        for path, type_, key, address in roots:
            output_type = replace(type_, storage=StorageKind.GMEM) if type_.storage is StorageKind.UMAT else type_
            output = Var(self.names.fresh("out"), type=output_type)
            self.outputs.append((path, output))
            value = self.addresses[key]
            returning = True
            while True:
                retained = (isinstance(value.value.expr, Call)
                            and self.callee_frame(value.value.expr, self.context.frame) is not None)
                if self._address(value.value.expr, value.value.index).writes or (returning and retained):
                    self.output_values[value.value] = (output, address)
                edge = _edge(value, self.addresses)
                if edge is None:
                    break
                returning = returning and not retained and not edge.writes and edge.kind in (
                    AliasKind.VIEW, AliasKind.FORWARD
                )
                value = self.addresses[edge.target]
            held = leaves_of(address.target.expr.type)[address.target.index]
            if address.relation.relation.is_identity() and held == type_:
                self.output_storage[address.target] = output
        body = self.function.body
        result = self.visit(body)
        if not isinstance(body, MeshRegion):
            self._finish(result)
        statements = self.ctx.blocks[0].build()

        params = tuple(self.parameters[id(param)] for param in self.function.params)
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

    def _address(self, value: Expr, index: int = 0):
        try:
            return address_alias(self.addresses[SSAValue(value, index)], self.addresses,
                                 precisions=self.address_precisions)
        except AnalysisError as error:
            raise LoweringError(str(error)) from error

    def _same_address(self, left, right):
        a, b = self._address(left), self._address(right)
        return a.target == b.target and a.relation.relation.is_equal(b.relation.relation)

    def _tensor(self, key: SSAValue):
        known = self.tensors.get(key)
        if known is not None:
            return known
        address = self._address(key.expr, key.index)
        value = self.addresses[key]
        while True:
            if value.value in self.output_values:
                output, returned = self.output_values[value.value]
                relation = address.relation.relation.apply_range(returned.relation.relation.reverse())
                relative = replace(address, relation=AccessRelation(
                    relation, {**address.relation.values, **returned.relation.values}
                ))
                desired = leaves_of(key.expr.type)[key.index]
                result = self._address_view(output, relative, desired)
                self.logical[id(result)] = desired
                self.tensors[key] = result
                return result
            edge = _edge(value, self.addresses)
            if edge is None:
                break
            value = self.addresses[edge.target]
        base = self.storage.get(address.target)
        if base is None:
            source = address.target.expr
            base = self.parameters.get(id(source))
            if base is None:
                base = self.output_storage.get(address.target)
            if base is None:
                base = self._declare(source, leaves_of(source.type)[address.target.index],
                                     getattr(source.target, "buffers", 1) if isinstance(source, Call) else 1)
            self.storage[address.target] = base
        desired = leaves_of(key.expr.type)[key.index]
        if desired.storage is StorageKind.GMEM and desired.dtype != base.type.dtype:
            raise LoweringError("gmem cross-dtype reuse has no offset-view representation")
        if desired.storage is StorageKind.SMEM and desired.dtype != base.type.dtype:
            source = address.target.expr
            offsets = get_metadata(source, MemoryMetadata).offsets
            with self.ctx.declaration_scope(self._staging_owner(source) if len(offsets) > 1 else None):
                result = self._smem_storage(key.expr, _storage_type(desired), offsets,
                                            self.names.fresh("value"), self._staging_loops(source))
        else:
            result = base if key == address.target else self._address_view(base, address, desired)
        self.logical[id(result)] = desired
        self.tensors[key] = result
        return result

    def _value(self, expr: Expr, fallback=None):
        index = 0
        def build(type_, lowered):
            nonlocal index
            if isinstance(type_, TupleType):
                elements = lowered.elements if isinstance(lowered, Tuple) else (None,) * len(type_.fields)
                return Tuple(tuple(build(field, value) for field, value in zip(
                    type_.fields, elements, strict=True
                )), type=type_)
            key = SSAValue(expr, index)
            index += 1
            if type_.shape == () and isinstance(lowered, (Call, Constant)):
                return lowered
            return self._tensor(key) if key in self.addresses else lowered
        return build(expr.type, fallback)

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
                index = tuple(self._return_leaves(scope.owner.type)).index((path, type_))
                root_value = self.addresses.get(SSAValue(scope.owner, index))
                if root_value is None:
                    continue
                source = root(root_value, self.addresses).value.expr
                previous = locations.get(id(source), self.locations.get(id(source)))
                receiver = scope.parent
                if previous is None:
                    locations[id(source)] = receiver
                else:
                    ancestors = {id(ctx.owner) for ctx in previous.ancestors()}
                    while id(receiver.owner) not in ancestors:
                        receiver = receiver.parent
                    locations[id(source)] = receiver
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
        return self._smem_storage(expr, storage_type, offsets, stem, self._staging_loops(expr))

    def _smem_storage(self, expr, storage_type, offsets, stem, loops):
        fields = tuple(self._literal_view(storage_type, offset) for offset in offsets)
        if len(fields) == 1:
            view = _View(fields[0].args[0], (), tuple(storage_type.shape), storage_type, stem)
            self.logical[id(view)] = storage_type
            return view
        tuple_type = TupleType(tuple(field.type for field in fields))
        stages = Var(self.names.fresh(f"{stem}_stages"), type=tuple_type)
        self.ctx.bind(stages, Tuple(fields, type=tuple_type))
        self.logical[id(stages)] = storage_type
        self.staged[id(expr)] = (stages, loops)
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
        if any(static_dim_value(bound) is None
               for loop in loops[1:] for bound in (loop.start, loop.extent, loop.step)):
            key = tuple(id(loop) for loop in loops)
            if key not in self.stage_counters:
                counter = Var(self.names.fresh("stage_counter"), type=_BINDING)
                one = Var(self.names.fresh("stage_one"), type=_BINDING)
                with self.ctx.declaration_scope(self._staging_owner(expr)):
                    for value, number in ((counter, 0), (one, 1)):
                        self.ctx.bind(value, Call(AllocTensor(tensor_type=_BINDING), (), type=_BINDING))
                        self.ctx.add(Evaluate(Fill(), (value, Constant(number, type=_BINDING))))
                self.stage_counters[key] = (counter, one)
            counter, _one = self.stage_counters[key]
            index = simplify_dim(DimMod, (counter, len(stages.type.fields)))
            return Call(TupleGetItem(), (stages, index), type=stages.type.fields[0])
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
            self._memo[id(param)] = (param, self._value(param, value) if SSAValue(param) in self.addresses else value)

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
            carried = self._value(param)
            if not self._same_address(param, loop.args[slot]):
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
                if not self._same_address(source, param):
                    self._emit_copy(value, carried)
            for key, (counter, one) in self.stage_counters.items():
                if key[-1] == id(loop):
                    self.ctx.add(Evaluate(TirBinary(kind=BinaryKind.ADD), (counter, one, counter)))
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
            return self._operand(known[1])
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
        if isinstance(value, Call):
            return self._operand(self.visit(value))
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
        summary = get_metadata(frame.function, AllocationSummary)
        values = {value.value: value for value in summary.values}
        for index in range(len(leaves_of(call.type))):
            returned = root(values[SSAValue(frame.function.body, index)], values).value
            formal = next((i for i, param in enumerate(frame.function.params)
                           if returned.expr is param), None)
            if formal is not None and self._address(call, index).target != self._address(call.args[formal], returned.index).target:
                raise LoweringError(f"return leaf {self._alias_path(SSAValue(call, index))!r} cannot be placed on its output: {_label(call)}")
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
        destination = self._value(call)
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
        for arg in call.args:
            self.visit(arg)
        return self._value(call)

    def visit_Reshape(self, call: Call):
        self.visit(call.args[0])
        return self._value(call)

    visit_Bitcast = visit_Reshape

    def visit_Reshard(self, call: Call):
        if aliased_operand(call) is None:
            return self._lower_automatic_instruction(call)
        return self.visit_Reshape(call)

    def visit_Transpose(self, call: Call) -> Expr | _View:
        return self._lower_automatic_instruction(call)

    def _declare_result(self, call: Call, buffers: int) -> Expr:
        if call.type.storage is StorageKind.RMEM and buffers != 1:
            raise LoweringError(f"{_label(call)} staged buffer has no owning scope")
        return self._value(call)

    def visit_Empty(self, call: Call) -> Expr:
        return self._declare_result(call, call.target.buffers)

    def visit_Zeros(self, call: Call) -> Expr | _View:
        return self._lower_fill(call, 0.0)

    def visit_FullLike(self, call: Call) -> Expr | _View:
        return self._lower_fill(call, call.target.value)

    def _lower_fill(self, call: Call, value) -> Expr | _View:
        result = self._declare_result(call, 1)
        self._emit_fill(result, call.type, value)
        return result

    def _window_start(self, value: Expr) -> Expr:
        if isinstance(value, Call) and isinstance(value.target, HirBinary):
            operation = _DIM_BINARY.get(value.target.kind)
            if operation is not None:
                return simplify_dim(operation, tuple(self._window_start(arg) for arg in value.args))
        return (
            self._operand(self.visit(value))
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
        tied = self._same_address(call, call.args[0])
        result = self._value(call)
        if not tied:
            self._emit_copy(target, result)
        value = self.addresses[SSAValue(call.args[1])]
        while (edge := _edge(value, self.addresses)) is not None:
            if edge.kind is AliasKind.WINDOW and edge.target == SSAValue(call):
                return result
            value = self.addresses[edge.target]
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
        destination_operand = call.args[params.index(produced[0])] if isinstance(call.target, ScheduleOp) and produced else None
        written = self._value(call) if produced else None
        read_values = {}
        supplied = params if isinstance(call.target, ScheduleOp) else reads
        for param, operand in zip(
            supplied, call.args if operands is None else operands, strict=True
        ):
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
            if value is None:
                raise LoweringError(f"{_label(call)} has no storage for {param.name}")
            operands.append((param.name, value))

        written = self._emit_instruction(
            call, op, tuple(operands), mesh, written, projection
        )
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

    def _address_view(self, target, address, desired):
        """Derive an exact physical view; a proved composition carries its origin."""
        relation = address.relation.relation
        if relation.is_identity() and tuple(desired.shape) == tuple(target.type.shape):
            zeros = tuple(i64_const(0) for _ in target.type.shape)
            return _View(target, zeros, tuple(target.type.shape), desired, "value", whole_tensor=True)
        layout = resolve_layout(_plain_layout(target.type), target.type.shape)
        placed = relation.apply_range(_layout_addresses(tuple(target.type.shape), layout))
        try:
            affine = as_multi_aff(placed).get_at(0)
            if any(affine.get_domain_local_space().get_div(div).get_coefficient_val(isl.dim_type.IN, axis).sgn()
                   for div in range(affine.dim(isl.dim_type.DIV))
                   for axis in range(len(desired.shape))):
                raise ValueError("address contains coordinate-dependent divisions")
            strides = tuple(int(affine.get_coefficient_val(isl.dim_type.IN, axis).num_si())
                            for axis in range(len(desired.shape)))
        except (ValueError, isl.Error) as error:
            if not isinstance(layout, ComposedLayout):
                raise LoweringError(f"non-affine address {relation}: {error}") from error
            strides = None
        values = {name: self._dim(value) for name, value in address.relation.values.items()}
        origin = relation
        for axis in range(relation.dim(isl.dim_type.IN)):
            origin = origin.fix_val(isl.dim_type.IN, axis, isl.val(0))
        point = origin.range()
        starts = tuple(isl_to_dim(point.dim_min(axis).gist_params(point.params()), values)
                       for axis in range(point.dim(isl.dim_type.SET)))
        starts = tuple(i64_const(start) if isinstance(start, int) else start for start in starts)
        whole = all(static_dim_value(start) == 0 for start in starts)
        if isinstance(layout, ComposedLayout):
            candidate = desired.layout
            if isinstance(candidate, ComposedLayout) or whole:
                if not isinstance(candidate, ComposedLayout):
                    candidate = replace(layout, outer=Layout(tuple(desired.shape), strides))
                if strides is not None and not affine.dim(isl.dim_type.PARAM):
                    candidate = replace(layout, offset=int(affine.get_constant_val().num_si()),
                                        outer=Layout(tuple(desired.shape), strides))
                elif strides is None:
                    linear = relation.apply_range(_layout_addresses(tuple(target.type.shape), Layout(
                        tuple(target.type.shape), compact_row_major(tuple(target.type.shape))
                    )))
                    linear = as_multi_aff(linear).get_at(0)
                    steps = tuple(int(linear.get_coefficient_val(isl.dim_type.IN, axis).num_si())
                                  for axis in range(len(desired.shape)))
                    offset = as_multi_aff(origin.apply_range(_layout_addresses(
                        tuple(target.type.shape), layout
                    ))).get_at(0).get_constant_val().num_si()
                    candidate = replace(layout, outer=composition(layout.outer, Layout(
                        tuple(desired.shape), steps
                    ), major="row"), offset=int(offset))
                declared = _layout_addresses(tuple(desired.shape), candidate).intersect_domain(relation.domain())
                if candidate.inner != layout.inner or not declared.is_equal(placed):
                    raise LoweringError(f"composed view does not preserve exact address {relation}")
                desired = replace(desired, layout=candidate)
                whole = True
                starts = tuple(i64_const(0) for _ in starts)
            else:
                desired = self._window_type(target, starts, desired)
        elif not isinstance(desired.layout, ShardLayout):
            desired = replace(desired, layout=Layout(tuple(desired.shape), strides))
        sizes = [1] * len(starts)
        shard = shard_layout_of(target.type.layout)
        while shard is not None:
            for axis in split_target_axes(shard, tuple(target.type.shape)):
                if axis is None:
                    continue
                if static_dim_value(starts[axis]) != 0:
                    raise LoweringError(f"nonzero pointer origin on split axis {axis}: {relation}")
                sizes[axis] = target.type.shape[axis]
            shard = shard_layout_of(shard.layout)
        return _View(target, starts, tuple(sizes), desired, "value", whole_tensor=whole)

    def _emit_instruction(self, call, op, operands, mesh, written, projection):
        atom = getattr(op, "atom", None)
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
                    desired = self.logical.get(id(value), value.type)
                    issued.append(self._whole_operand(value, desired, frame, f"{role}_frame"))
                self.ctx.add(Evaluate(self._issued_op(op), tuple(issued)))
            return written
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
                scope.frame for scope in reversed(self.ctx.blocks)
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

    def _emit_fill(self, target, logical_type: TensorType, value) -> None:
        mesh = self._holder_mesh(logical_type)
        frame = None if mesh is None else self._physical_frame(mesh)
        scalar = Constant(value, type=TensorType.scalar(logical_type.dtype))
        with self._issue_scope(frame):
            written = self._operand(target)
            self.ctx.add(Evaluate(Fill(), (written, scalar)))

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
        for index, (path, _output) in enumerate(self.outputs):
            if SSAValue(self.function.body, index) not in self.addresses:
                raise LoweringError(f"return leaf {path!r} cannot be placed on its output: None")


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
