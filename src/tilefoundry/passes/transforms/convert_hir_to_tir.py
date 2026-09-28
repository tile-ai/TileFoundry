"""Lower scheduled HIR to effect-form TIR.

The pass consumes the checked, inlined view produced by ``analyze(memory)``.
Memory metadata owns storage addresses; registered access relations own
instruction issue geometry. This file only turns those two facts into statements.
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
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.view import presented_layout_of
from tilefoundry.ir.hir.tensor.zeros import Zeros
from tilefoundry.ir.pattern import MeshPattern, PatternMatcher, SwitchPattern
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
from tilefoundry.ir.types.utils import i64_const, static_dim_value, tile_inner_type
from tilefoundry.ir.visitor import ExprVisitor, StmtMutator, expr_children
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    access_relation_registry,
    iteration_universe,
    relation_of,
)
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


def _read_pattern(param, op):
    pattern = param.pattern.read_on(op) if hasattr(param.pattern, "read_on") else param.pattern
    bindings = dict(getattr(getattr(op, "atom", None), "bindings", {}))
    while isinstance(pattern, SwitchPattern) and pattern.param in bindings:
        pattern = dict(pattern.branches).get(bindings[pattern.param])
    return pattern


def _inner_map(boundary) -> isl.map:
    """One scheduled boundary with repeat coordinates fixed at the first tile."""
    relation = relation_of(boundary.pattern)
    repeated = [
        axis
        for axis in range(relation.dim(isl.dim_type.IN))
        if (relation.get_dim_name(isl.dim_type.IN, axis) or "").startswith("r")
    ]
    for axis in reversed(repeated):
        relation = relation.fix_input_si(axis, 0)
        relation = relation.project_out(isl.dim_type.IN, axis, 1)
    return relation


def _set_extents(space: isl.set) -> tuple[int, ...]:
    return tuple(
        int(space.dim_max(axis).max_val().num_si())
        - int(space.dim_min(axis).min_val().num_si())
        + 1
        for axis in range(space.dim(isl.dim_type.SET))
    )


def _projects(relation: isl.map, source: int, target: int) -> bool:
    local = isl.local_space.from_space(relation.get_space())
    equal = isl.constraint.alloc_equality(local)
    equal = equal.set_coefficient_si(isl.dim_type.IN, source, 1)
    equal = equal.set_coefficient_si(isl.dim_type.OUT, target, -1)
    return relation.is_subset(isl.map.universe(relation.get_space()).add_constraint(equal))


def _operand_work_axes(relations: AccessRelations) -> tuple[tuple[int | None, ...], ...]:
    mapped = []
    for boundary in relations.inputs:
        relation = _inner_map(boundary)
        mapped.append(
            tuple(
                next(
                    (
                        source
                        for source in range(relation.dim(isl.dim_type.IN))
                        if _projects(relation, source, target)
                    ),
                    None,
                )
                for target in range(relation.dim(isl.dim_type.OUT))
            )
        )
    return tuple(mapped)


def _work_geometry(relations: AccessRelations, authored_order: tuple | None):
    universe = iteration_universe(relations)
    if universe is None:
        raise ValueError("instruction access relations state no iteration space")
    names = tuple(
        universe.get_dim_name(isl.dim_type.SET, axis) or f"d{axis}"
        for axis in range(universe.dim(isl.dim_type.SET))
    )
    repeat_positions = tuple(
        position for position, name in enumerate(names) if name.startswith("r")
    )
    if repeat_positions:
        rank = len(names) - len(repeat_positions)
        repeats = [1] * rank
        order = []
        for position in repeat_positions:
            axis = int(names[position][1:])
            repeats[axis] = _set_extents(universe)[position]
            order.append(axis)
        inner_universe = _inner_map(relations.outputs[0]).domain()
        atoms = _set_extents(inner_universe)
    else:
        rank = len(names)
        repeats = [1] * rank
        order = list(range(rank)) if authored_order is None else list(authored_order)
        atoms = _set_extents(universe)
    output = _inner_map(relations.outputs[0])
    axis_names = [f"d{axis}" for axis in range(rank)]
    if rank == 3 and output.dim(isl.dim_type.OUT) == 2:
        projected = {
            source: target
            for source in range(rank)
            for target in range(2)
            if _projects(output, source, target)
        }
        if len(projected) == 2:
            axis_names = [
                "k" if axis not in projected else ("m", "n")[projected[axis]]
                for axis in range(rank)
            ]
    repeats = tuple(repeats)
    atoms = tuple(atoms)
    return (
        tuple(axis_names),
        tuple(atom * repeat for atom, repeat in zip(atoms, repeats, strict=True)),
        atoms,
        repeats,
        tuple(order),
    )


def _swizzle_row(type_: TensorType) -> tuple[int, int] | None:
    layout = type_.layout
    if isinstance(layout, ShardLayout):
        layout = layout.layout
    if not (
        isinstance(layout, ComposedLayout)
        and isinstance(layout.inner, Swizzle)
        and isinstance(layout.outer, Layout)
        and layout.outer.strides is not None
    ):
        return None
    modes = [
        (axis, extent, step)
        for axis, (group, steps) in enumerate(
            zip(layout.outer.shape, layout.outer.strides, strict=True)
        )
        for extent, step in zip(flatten(group), flatten(steps), strict=True)
        if extent != 1
    ]
    if any(type(value) is not int for _, extent, step in modes for value in (extent, step)):
        return None
    units = [(axis, extent) for axis, extent, step in modes if step == 1]
    above = [step for _, _, step in modes if step > 1]
    if len(units) != 1 or not above:
        return None
    (axis, extent), next_step = units[0], min(above)
    fastest = [mode for mode in modes if mode[0] == axis][-1]
    if fastest[2] != 1 or next_step <= extent or next_step % extent:
        return None
    return axis, next_step // extent


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
        self.current_loops: list[tuple[LoopRegion, Var]] = []
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
        scope = self.scope_for_call.get(id(expr))
        cursor = scope
        while cursor is not None:
            if isinstance(cursor.owner, LoopRegion):
                return self.function if cursor.parent is None else cursor.parent.owner
            cursor = cursor.parent
        return self.function

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
        bindings: dict[int, Expr] = {}
        cursor = scope
        while cursor is not None:
            if isinstance(cursor.owner, LoopRegion):
                loops.append(cursor.owner)
                bindings.update(
                    zip(
                        map(id, cursor.owner.carried_args),
                        cursor.owner.init_args,
                        strict=True,
                    )
                )
            elif isinstance(cursor.owner, MeshRegion):
                bindings.update(
                    zip(map(id, cursor.owner.params), cursor.owner.args, strict=True)
                )
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
                origin = tuple(
                    self._constant_index(start, values, bindings) for start in starts_
                )
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

    def _constant_index(
        self,
        value: Expr,
        values: dict[int, int],
        bindings: dict[int, Expr],
    ) -> int:
        bound = bindings.get(id(value))
        if bound is not None:
            return self._constant_index(bound, values, bindings)
        if isinstance(value, Constant) and type(value.value) is int:
            return value.value
        if isinstance(value, Var) and id(value) in values:
            return values[id(value)]
        if isinstance(value, Call) and isinstance(value.target, MeshCoord):
            known = values.get(id(value))
            if known is not None:
                return known
        if isinstance(value, Call) and isinstance(value.target, HirBinary):
            lhs, rhs = (
                self._constant_index(arg, values, bindings) for arg in value.args
            )
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
            result = self._declare(
                call,
                call.type,
                1,
                self.owner_cursors[id(self.function)],
            )
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
        layout = _plain_layout(
            replace(call.type, layout=presented_layout_of(call, self.type_ctx))
        )
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
        layout = _plain_layout(
            replace(desired, layout=presented_layout_of(call, self.type_ctx))
        )
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
                if not self._covers_output(call):
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
        try:
            relations = (
                scope.stated_relations(call, self.type_ctx)
                if scope is not None
                else access_relation_registry.lookup(type(call.target))(call, self.type_ctx)
            )
            geometry = _work_geometry(relations, call.target.order)
        except (TypeError, ValueError, isl.Error) as error:
            raise LoweringError(f"{_label(call)} cannot read issue geometry: {error}") from error

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
            self._stage(call, self.staged[id(call)])
            if id(call) in self.staged
            else self.memo.get(id(call))
        )
        participant_scope = getattr(getattr(call.target.op, "atom", None), "required_scope", None)
        operand_mesh = next(
            (
                layout.mesh
                for value in call.args
                if isinstance((layout := value.type.layout), ShardLayout)
            ),
            mesh,
        )
        read_desired = tuple(
            tile_inner_type(
                TensorType(
                    value.type.shape,
                    value.type.dtype,
                    presented_layout_of(value, self.type_ctx),
                    value.type.storage,
                ),
                _set_extents(_inner_map(boundary).range()),
                participant=participant_scope,
                enclosing=operand_mesh,
                shard_attrs=getattr(
                    getattr(_read_pattern(param, call.target.op), "layout", None),
                    "attrs",
                    None,
                ),
            )
            for param, value, boundary in zip(reads, call.args, relations.inputs, strict=True)
        )
        desired_by_read = dict(zip((param.name for param in reads), read_desired, strict=True))
        operands = []
        for param in params:
            desired = (
                desired_by_read[param.name]
                if param.effect & MemoryEffect.READ
                else tile_inner_type(call.type, tuple(call.type.shape))
            )
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
            operand_axes = _operand_work_axes(relations)
            row_copies = [1] * len(geometry[0])
            operand_rows = []
            for desired, axes in zip(read_desired, operand_axes, strict=True):
                row = _swizzle_row(desired)
                rows = set()
                if row is not None and axes[row[0]] is not None:
                    axis = axes[row[0]]
                    row_copies[axis] = max(row_copies[axis], row[1])
                    rows.add(axis)
                operand_rows.append(frozenset(rows))
            for frame, group_lows in self._atom_groups(call, geometry, mesh, participant):
                statement = self._emit_atom_axes(
                    call,
                    geometry,
                    tuple(row_copies),
                    operand_axes,
                    tuple(operand_rows),
                    tuple(operands),
                    frame,
                    group_lows,
                    0,
                    {},
                    {},
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
        geometry,
        mesh: Mesh | None,
        participant: Mesh,
    ) -> tuple[tuple[Mesh, dict[int, int]], ...]:
        names, _extents, atoms, repeats, _order = geometry
        physical_mesh = next(
            (
                layout.mesh
                for argument in call.args
                if isinstance((layout := getattr(argument.type, "layout", None)), ShardLayout)
            ),
            mesh,
        )
        if physical_mesh is None:
            raise LoweringError(f"{_label(call)} has atom axes but no physical mesh")
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
        grouped = tuple(axis for axis, count in enumerate(repeats) if count > 1)[:outer_rank]
        expected = tuple(repeats[axis] for axis in grouped)
        if not grouped:
            return ((participant, {}),)
        if mesh is None:
            raise LoweringError(f"{_label(call)} has group axes but no physical mesh")
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
        for coordinates in product(*(range(repeats[axis]) for axis in grouped)):
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
                axis: coordinate * atoms[axis]
                for axis, coordinate in zip(grouped, coordinates, strict=True)
            }
            groups.append((frame, lows))
        return tuple(groups)

    def _emit_atom_axes(
        self,
        call: Call,
        geometry,
        row_copies: tuple[int, ...],
        operand_axes: tuple[tuple[int | None, ...], ...],
        operand_rows: tuple[frozenset[int], ...],
        operands: tuple[tuple[str, Expr, TensorType], ...],
        frame: Mesh,
        group_lows: dict[int, int],
        depth: int,
        offsets: dict[int, Expr],
        slices: dict[int, int],
    ) -> Sequential:
        names, extents, atoms, _repeats, order = geometry
        if depth == len(order):
            return self._emit_atom_call(
                call,
                atoms,
                operand_axes,
                operand_rows,
                operands,
                frame,
                offsets,
                slices,
            )
        axis = order[depth]
        low = group_lows.get(axis, 0)
        stop = low + atoms[axis] if axis in group_lows else extents[axis]
        extent = stop - low
        if extent % atoms[axis]:
            raise LoweringError(
                f"{_label(call)} axis {names[axis]} extent {extent} is not divisible by "
                f"atom {atoms[axis]}"
            )
        beside = min(row_copies[axis], extent // atoms[axis])
        if extent % (atoms[axis] * beside):
            raise LoweringError(
                f"{_label(call)} axis {names[axis]} extent {extent} is not divisible by "
                f"row {atoms[axis]} * {beside}"
            )
        counter = Var(self.names.fresh(f"o_{names[axis]}"), type=_INDEX)
        body = _Cursor()
        for copy in range(beside):
            start = counter if copy == 0 else simplify_dim(DimAdd, (counter, copy * atoms[axis]))
            nested = self._emit_atom_axes(
                call,
                geometry,
                row_copies,
                operand_axes,
                operand_rows,
                operands,
                frame,
                group_lows,
                depth + 1,
                {**offsets, axis: start},
                {**slices, axis: copy},
            )
            for statement in nested.body:
                body.add(statement)
        return Sequential(
            (
                For(
                    counter,
                    i64_const(low),
                    i64_const(stop),
                    i64_const(atoms[axis] * beside),
                    body.build(),
                ),
            )
        )

    def _emit_atom_call(
        self,
        call: Call,
        atoms: tuple[int, ...],
        operand_axes: tuple[tuple[int | None, ...], ...],
        operand_rows: tuple[frozenset[int], ...],
        operands: tuple[tuple[str, Expr, TensorType], ...],
        frame: Mesh,
        offsets: dict[int, Expr],
        slices: dict[int, int],
    ) -> Sequential:
        issue = _Cursor()
        issued_args = []
        for (role, value, desired), axes, row_axes in zip(
            operands, operand_axes, operand_rows, strict=True
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
