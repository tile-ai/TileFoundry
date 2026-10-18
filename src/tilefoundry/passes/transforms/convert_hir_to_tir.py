"""Lower scheduled HIR to effect-form TIR.

The pass analyzes a checked program and lowers each retained definition once.
StoragePlan interprets raw allocation decisions and registered access
relations; the visitor writes the resulting declarations and statements.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import product
from math import prod

import isl

from tilefoundry.analysis import MemoryMetadata, analyze
from tilefoundry.analysis.access import AliasKind, SSAValue, buffer_alias, forwarded_leaves
from tilefoundry.analysis.allocation import AllocationSummary
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
from tilefoundry.ir.hir.schedule import ScheduleOp, operand_relations
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
from tilefoundry.ir.visitor import collect_exprs, expr_children
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.utils.isl_utils import as_multi_aff
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    affine_term,
    identity_access,
    leaf_paths,
    leaves_of,
    projected_axes,
    relations_of,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.call_frame import CallFrame, CallFrameVisitor, ValuationVisitor
from tilefoundry.visitor_registry.candidates import (
    candidate_operands,
    candidate_ops,
    sole_candidate,
)
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.instruction import ScheduleRefusal, project
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


class LoweringError(ValueError):
    """A checked HIR construct has no sound TIR representation."""


@dataclass(frozen=True, eq=False)
class View:
    """An address recipe, materialized only in its consuming scope."""

    base: Expr | View
    starts: tuple[Expr, ...]
    sizes: tuple
    type: TensorType
    stem: str = "tile"
    whole_tensor: bool = False


@dataclass
class TirScope:
    mesh: Mesh | None = None
    region: Expr | None = None
    bound: dict = field(default_factory=dict)
    views: dict = field(default_factory=dict)
    entries: list = field(default_factory=list)
    result: Expr | View | None = None


@dataclass
class TirBuilder:
    """The only component that declares values and appends TIR statements."""

    blocks: list[TirScope]
    loops: list = field(default_factory=list)
    stage_counters: dict = field(default_factory=dict)

    def add(self, stmt):
        self.blocks[-1].entries.append(stmt)

    def bind(self, var, value):
        self.blocks[-1].entries.append((var, value))
        self.blocks[-1].bound[id(var)] = var
        return var

    def build(self, region=None):
        body = ()
        for entry in reversed((region or self.blocks[-1]).entries):
            body = (
                (LetStmt(*entry, Sequential(body)),) if isinstance(entry, tuple) else (entry, *body)
            )
        return Sequential(body)

    @contextmanager
    def region(self, *, owner=None, mesh=None, binding=None, loop=None):
        if mesh is not None and owner is None:
            current = self.blocks[-1].mesh
            if current is not None and replace(
                current, layout=resolve_layout(current.layout)
            ) == replace(mesh, layout=resolve_layout(mesh.layout)):
                mesh = None
        if mesh is not None or loop is not None:
            region = TirScope(mesh if mesh is not None else self.blocks[-1].mesh, owner)
            if binding is not None:
                region.bound[id(binding)] = binding
        else:
            region = (
                self.blocks[-1]
                if owner is None
                else next((r for r in self.blocks if r.region is owner), None)
            )
            if region is None:
                raise LoweringError("staged buffer has no owning scope")
        self.blocks.append(region)
        if loop is not None:
            region.bound[id(loop[0])] = loop[0]
            self.loops.append(loop)
        try:
            yield region
        finally:
            self.blocks.pop()
            if loop is not None:
                self.loops.pop()
        if mesh is not None or loop is not None:
            pending = [region.result]
            while pending:
                value = pending.pop()
                if isinstance(value, Var) and id(value) in region.bound:
                    raise LoweringError(
                        f"value {value.name!r} escapes region mesh {mesh or region.mesh!r}"
                    )
                if isinstance(value, View):
                    pending.extend((value.base, *value.starts))
                elif isinstance(value, Expr) and not isinstance(value, Var):
                    pending.extend(expr_children(value))
        if mesh is not None:
            self.add(MeshScope(mesh, binding, self.build(region)))
        elif loop is not None:
            self.add(For(*loop, self.build(region)))

    def declare(self, leaf, *, ctx, values, desired=None):
        """Declare one allocation root in its owning scope."""
        expr = leaf.expr
        type_ = desired or leaves_of(expr.type)[leaf.index]
        buffers = getattr(expr.target, "buffers", 1) if isinstance(expr, Call) else 1
        owner = (
            ctx.storage.placement[id(expr)][0]
            if buffers > 1
            else ctx.storage.placement.get(id(expr), (None, (), None))[2]
            if type_.storage is StorageKind.RMEM
            else None
        )
        with ctx.builder.region(owner=owner):
            layout = type_.layout.layout if isinstance(type_.layout, ShardLayout) else type_.layout
            storage_type = replace(type_, layout=layout)
            stem = ctx.names.fresh(
                expr.name if isinstance(expr, Var) else "value",
                ctx.authored_values.get(id(expr)),
            )
            if type_.storage is StorageKind.RMEM:
                storage_type = type_
                var = Var(stem, type=storage_type)
                ctx.builder.bind(
                    var, Call(AllocTensor(tensor_type=storage_type), (), type=storage_type)
                )
                return var
            if storage_type.storage not in (StorageKind.SMEM, StorageKind.GMEM):
                raise LoweringError(
                    f"{_label(expr)} has addressable result in unsupported storage {storage_type.storage}"
                )
            offsets = ctx.storage.answers[leaf].offsets
            if not offsets:
                raise LoweringError(
                    f"{_label(expr)} has an addressable {storage_type.storage} result but no offsets"
                )
            if len(offsets) != buffers:
                raise LoweringError(
                    f"{_label(expr)} declares {buffers} buffer(s), but memory analysis placed {len(offsets)}"
                )
            if storage_type.storage is StorageKind.GMEM:
                if buffers != 1:
                    raise LoweringError(f"{_label(expr)} uses staged internal gmem")
                elements = prod(tuple(storage_type.shape))
                scratch = Var(
                    ctx.names.fresh("scratch"),
                    type=TensorType((elements,), storage_type.dtype, None, StorageKind.GMEM),
                )
                ctx.scratch.append(scratch)
                storage_type = replace(
                    storage_type,
                    layout=Layout(
                        tuple(storage_type.shape),
                        tuple(compact_row_major(tuple(storage_type.shape))),
                    ),
                )
                view = View(scratch, (i64_const(0),), (elements,), storage_type, stem)
                return view
            fields = tuple(ctx.storage.view(offset, None, storage_type) for offset in offsets)
            if len(fields) == 1:
                view = View(fields[0].args[0], (), tuple(storage_type.shape), storage_type, stem)
                return view
            stages = Var(
                ctx.names.fresh(f"{stem}_stages"),
                type=TupleType(tuple(field.type for field in fields)),
            )
            ctx.builder.bind(stages, Tuple(fields, type=stages.type))
            loops = ctx.storage.placement[id(expr)][1]
            if any(
                static_dim_value(bound) is None
                for loop in loops[1:]
                for bound in (loop.start, loop.extent, loop.step)
            ):
                key = tuple(id(loop) for loop in loops)
                if key not in self.stage_counters:
                    counter = Var(ctx.names.fresh("stage_counter"), type=_BINDING)
                    one = Var(ctx.names.fresh("stage_one"), type=_BINDING)
                    with ctx.builder.region(owner=ctx.storage.placement[id(expr)][0]):
                        for value, number in ((counter, 0), (one, 1)):
                            ctx.builder.bind(
                                value,
                                Call(AllocTensor(tensor_type=_BINDING), (), type=_BINDING),
                            )
                            ctx.builder.add(
                                Evaluate(Fill(), (value, Constant(number, type=_BINDING)))
                            )
                    self.stage_counters[key] = (counter, one)
                counter, _one = self.stage_counters[key]
                index = simplify_dim(DimMod, (counter, len(stages.type.fields)))
                return Call(TupleGetItem(), (stages, index), type=stages.type.fields[0])
            index = i64_const(0)
            for depth, loop in enumerate(loops):
                induction = values.get(id(loop.induction_var))
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
                    else simplify_dim(DimSub, (induction, values[id(loop.start)]))
                )
                ordinal = moved if step == 1 else simplify_dim(DimFloorDiv, (moved, step))
                index = ordinal if depth == 0 else simplify_dim(DimAdd, (index, ordinal))
            if loops:
                index = simplify_dim(DimMod, (index, len(stages.type.fields)))
            field = stages.type.fields[0]
            return Call(TupleGetItem(), (stages, index), type=field)

    def value(self, value, *, ctx):
        """Materialize a recipe in its consuming scope without changing it."""
        if not isinstance(value, View):
            return value
        scope = self.blocks[-1]
        known = scope.views.get(id(value))
        if known is None:
            base = self.value(value.base, ctx=ctx)
            expression = ctx.storage.materialization(value, base, tuple(self.loops))
            stem = (
                value.stem
                if isinstance(base, Constant) and isinstance(base.type, PointerType)
                else ctx.names.fresh(value.stem)
            )
            result = self.bind(Var(stem, type=expression.type), expression)
            scope.views[id(value)] = (value, result)
        return scope.views[id(value)][1]

    def transfer(
        self,
        source,
        target,
        *,
        starts: tuple[Expr, ...] | None = None,
        sizes: tuple | None = None,
        desired: TensorType | None = None,
        ctx,
        op=None,
        source_type=None,
    ) -> None:
        source_type = source.type if source_type is None else source_type
        if not isinstance(source_type, TensorType) or not isinstance(target.type, TensorType):
            raise LoweringError("copy lowering requires tensor operands")
        mesh = next(
            (
                participant_layout(type_.layout).mesh
                for type_ in (source_type, target.type)
                if isinstance(type_.layout, ShardLayout)
            ),
            None,
        )
        with self.region(mesh=mesh, binding=Var(ctx.names.fresh("threads"), type=_BINDING)):
            if isinstance(op, Fill):
                self.add(Evaluate(op, (self.value(target, ctx=ctx), source)))
                return
            read = (
                self.value(source, ctx=ctx)
                if mesh is None
                else self.value(ctx.storage.operand(source, source_type), ctx=ctx)
            )
            written = self.value(target, ctx=ctx)
            if starts is not None:
                if sizes is None or desired is None:
                    raise LoweringError("a windowed copy needs sizes and a destination type")
                written = self.value(
                    ctx.storage.operand(
                        written, desired, starts=starts, sizes=sizes, stem="window"
                    ),
                    ctx=ctx,
                )
            ctx.builder.add(Evaluate(Copy(), (read, written)))


@dataclass
class Names:
    """Stable authored names plus deterministic generated-name suffixes."""

    used: set[str]

    def fresh(self, stem: str, authored=None) -> str:
        binding = get_metadata(authored, BindingMetadata) if authored is not None else None
        stem = binding.name if binding is not None else stem
        if stem not in self.used:
            self.used.add(stem)
            return stem
        index = 1
        while f"{stem}_{index}" in self.used:
            index += 1
        result = f"{stem}_{index}"
        self.used.add(result)
        return result

    def pair(self, expr: Expr, ctx) -> None:
        authored = ctx.authored_values.get(id(expr))
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
                ctx.authored_values[id(actual)] = source


def _label(call: Expr) -> str:
    if isinstance(call, Var):
        return call.name
    binding = get_metadata(call, BindingMetadata)
    return binding.name if binding is not None else type(call.target).__name__


@dataclass(frozen=True)
class StorageAnswer:
    partner: SSAValue | int | None
    chain: tuple[SSAValue, ...]
    aliases: tuple
    offsets: tuple[int, ...]
    relation: AccessRelation | None
    refusal: str | None
    kind: AliasKind | None = None


@dataclass
class StoragePlan:
    """Interpret raw allocation decisions without publishing derived geometry."""

    function: Function
    frame: CallFrame | None = None
    mesh: Mesh | None = None

    @cached_property
    def regions(self):
        values = (*self.function.params, *collect_exprs(self.function.body))
        memo = {id(expr): (expr, expr.type) for expr in values}
        ctx = TypeInferContext(
            frame=self.frame, owner=self.function, current_mesh=self.mesh, memo=memo
        )
        return ValuationVisitor.index_contexts(self.function, ctx)

    @cached_property
    def answers(self):
        summary = get_metadata(self.function, AllocationSummary)
        values = {value.value: value for value in summary.values}
        links, operations, outputs, refusals = {}, {}, {}, {}
        for leaf, value in values.items():
            ctx = self.regions.get(id(leaf.expr))
            if ctx is None:
                continue
            forwarded = forwarded_leaves(leaf.expr, ctx)
            position = aliased_operand(leaf.expr) if isinstance(leaf.expr, Call) else None
            if forwarded is not None:
                partner = forwarded[leaf.index]
                kind = AliasKind.FORWARD
            elif position is not None:
                partner = SSAValue(leaf.expr.args[position])
                kind = AliasKind.VIEW
            else:
                alias = next(
                    (
                        alias
                        for alias in value.aliases
                        if alias.target != leaf
                        and alias.target in values
                        and not (
                            value.lifetime.memory_level == "rmem"
                            and alias.kind is AliasKind.IN_PLACE
                        )
                    ),
                    None,
                )
                partner = None if alias is None else alias.target
                kind = None if alias is None else alias.kind
            if partner is not None and partner != leaf:
                links[leaf] = partner
                operations[leaf] = (kind, leaf.expr)
        for holder, alias, representative in summary.address_decisions:
            value = values[representative]
            if value.lifetime.memory_level == "rmem" and alias.kind is AliasKind.IN_PLACE:
                continue
            if holder != representative:
                chain = self.chain(holder, links)
                refusal = (
                    self.identity_refusal(
                        chain[: chain.index(representative) + 1], operations, values
                    )
                    if representative in chain
                    else f"address for {values[holder].lifetime.binding} requires an inverse relation to an unreachable storage representative: {alias.relation.relation}"
                )
                if refusal is not None:
                    refusals[representative] = refusal
            links[representative] = alias.target
            operations[representative] = (
                alias.kind,
                alias.target.expr if alias.kind is AliasKind.WINDOW else holder.expr,
            )
        seen = set()
        for index, (path, type_) in enumerate(leaf_paths(self.function.return_type)):
            if not isinstance(type_, TensorType):
                continue
            leaf = SSAValue(self.function.body, index)
            if leaf not in values:
                refusals[leaf] = f"return leaf {path!r} cannot be placed on its output: None"
                continue
            chain = self.chain(leaf, links)
            root = chain[-1]
            if root in seen:
                refusals[leaf] = f"return leaf {path!r} shares storage with another output"
                continue
            seen.add(root)
            returning = True
            for offset, item in enumerate(chain):
                retained = (
                    isinstance(item.expr, Call)
                    and CallFrameVisitor.callee_frame(item.expr, self.frame) is not None
                )
                writes = any(
                    operations.get(member, (None, None))[0] is AliasKind.INSERT_SLICE
                    or any(alias.writes for alias in values[member].aliases)
                    for member in chain[offset:]
                )
                if writes or (returning and retained):
                    if (
                        refusal := self.identity_refusal(chain[: offset + 1], operations, values)
                    ) is not None:
                        refusals[item] = refusal
                    outputs[item] = index
                kind = operations.get(item, (None, None))[0]
                returning = (
                    returning and not retained and kind in (AliasKind.VIEW, AliasKind.FORWARD)
                )
            if (
                not any(root.expr is param for param in self.function.params)
                and leaves_of(root.expr.type)[root.index] == type_
                and all(
                    operations.get(item, (None, None))[0] is not AliasKind.VIEW
                    or (
                        (
                            alias := buffer_alias(
                                item.expr,
                                self.regions[id(item.expr)],
                                relations_of(item.expr, self.regions[id(item.expr)]),
                            )
                        )
                        is not None
                        and alias.relation.relation.is_identity()
                    )
                    for item in chain[:-1]
                )
            ):
                outputs[root] = index
        for leaf in values:
            if id(leaf.expr) not in self.regions or not isinstance(leaf.expr, Call):
                continue
            callee = CallFrameVisitor.callee_frame(leaf.expr, self.frame)
            if callee is None:
                continue
            child = StoragePlan(callee.function, callee)
            returned = child.answers[SSAValue(callee.function.body, leaf.index)].chain[-1]
            formal = next(
                (i for i, param in enumerate(callee.function.params) if returned.expr is param),
                None,
            )
            if (
                formal is not None
                and self.chain(leaf, links)[-1]
                != self.chain(SSAValue(leaf.expr.args[formal], returned.index), links)[-1]
            ):
                path = tuple(
                    path
                    for path, type_ in leaf_paths(leaf.expr.type)
                    if isinstance(type_, TensorType)
                )[leaf.index]
                refusals[leaf] = (
                    f"return leaf {path!r} cannot be placed on its output: {_label(leaf.expr)}"
                )
        answers = {}
        for leaf, value in values.items():
            if id(leaf.expr) not in self.regions:
                continue
            relation = identity_access(len(leaves_of(leaf.expr.type)[leaf.index].shape))
            kind, origin = operations.get(leaf, (None, leaf.expr))
            if leaf not in outputs and kind is AliasKind.VIEW:
                ctx = self.regions[id(origin)]
                alias = buffer_alias(origin, ctx, relations_of(origin, ctx))
                if alias is None:
                    refusals[leaf] = (
                        f"non-EXACT address for {value.lifetime.binding}: {value.aliases[0].relation.relation}"
                    )
                else:
                    relation = alias.relation
            elif leaf not in outputs and kind is AliasKind.WINDOW:
                offsets = origin.args[2]
                shape = leaves_of(leaf.expr.type)[leaf.index].shape
                offsets = (
                    offsets.elements
                    if isinstance(offsets, Tuple)
                    else (offsets, *(0 for _ in shape[1:]))
                )
                coordinates, images, guards, parameters = [], [], [], {}
                for axis, (offset, extent) in enumerate(zip(offsets, shape, strict=True)):
                    term, bound = affine_term(offset, f"o{axis}")
                    parameters.update(bound)
                    coordinates.append(f"i{axis}")
                    images.append(f"i{axis} + ({term})")
                    guards.append(f"0 <= i{axis} < {extent}")
                prefix = f"[{', '.join(parameters)}] -> " if parameters else ""
                relation = AccessRelation(
                    isl.map(
                        f"{prefix}{{ [{', '.join(coordinates)}] -> [{', '.join(images)}] : {' and '.join(guards)} }}"
                    ),
                    parameters,
                )
            aliases = tuple(
                alias
                for alias in value.aliases
                if value.lifetime.memory_level != "rmem" or alias.kind is not AliasKind.IN_PLACE
            )
            answers[leaf] = StorageAnswer(
                outputs.get(leaf, links.get(leaf)),
                self.chain(leaf, links),
                aliases,
                getattr(get_metadata(leaf.expr, MemoryMetadata), "offsets", ()),
                relation,
                None,
                kind,
            )
        return {
            leaf: replace(
                answer,
                refusal=next((refusals[item] for item in answer.chain if item in refusals), None),
            )
            for leaf, answer in answers.items()
        }

    def chain(self, leaf, links):
        result = []
        while leaf not in result:
            result.append(leaf)
            if leaf not in links:
                return tuple(result)
            leaf = links[leaf]
        raise LoweringError("allocation: storage links contain a cycle")

    def identity_refusal(self, chain, operations, values) -> str | None:
        """Check the steps whose inverse would otherwise be silently discarded."""
        for leaf, partner in zip(chain, chain[1:]):
            kind, origin = operations[leaf]
            if kind is AliasKind.FORWARD:
                continue
            value = values[leaf]
            accepted = next(
                (
                    alias
                    for alias in value.aliases
                    if alias.kind is kind and alias.target == partner
                ),
                None,
            )
            same_shape = tuple(leaves_of(leaf.expr.type)[leaf.index].shape) == tuple(
                leaves_of(partner.expr.type)[partner.index].shape
            )
            if kind is AliasKind.VIEW:
                ctx = self.regions[id(origin)]
                declared = buffer_alias(origin, ctx, relations_of(origin, ctx))
                if same_shape and declared is not None and declared.relation.relation.is_identity():
                    continue
            elif accepted is not None:
                if same_shape and kind in (
                    AliasKind.CARRY_INIT,
                    AliasKind.CARRY_YIELD,
                    AliasKind.INSERT_SLICE,
                ):
                    continue
                if kind is AliasKind.IN_PLACE and value.lifetime.memory_level != "rmem":
                    continue
            relation = declared if kind is AliasKind.VIEW and declared is not None else accepted
            return (
                f"address for {values[chain[0]].lifetime.binding} requires a non-identity inverse relation "
                f"at {value.lifetime.binding}: {None if relation is None else relation.relation.relation}"
            )
        return None

    @cached_property
    def placement(self):
        """Place genuinely shared registers at their outermost HIR definition.

        Region results share their body's storage; ties and views follow the
        same storage trace as outputs. Allocation's peak-estimation buffers
        are not physical register sharing and do not determine placement.
        """
        locations = {}
        for key, region in self.regions.items():
            if key != id(region.owner) or region.parent is None:
                continue
            for index, (_path, type_) in enumerate(leaf_paths(region.owner.type)):
                if not isinstance(type_, TensorType) or type_.storage is not StorageKind.RMEM:
                    continue
                leaf = SSAValue(region.owner, index)
                if leaf not in self.answers or self.answers[leaf].refusal is not None:
                    continue
                source = self.answers[leaf].chain[-1].expr
                previous = locations.get(id(source), self.regions.get(id(source)))
                receiver = region.parent
                if previous is None:
                    locations[id(source)] = receiver
                else:
                    ancestors = {id(region.owner) for region in previous.ancestors()}
                    while id(receiver.owner) not in ancestors:
                        receiver = receiver.parent
                    locations[id(source)] = receiver
        result = {}
        for key, region in self.regions.items():
            outermost = next(
                (
                    scope
                    for scope in reversed(tuple(region.ancestors()))
                    if isinstance(scope.owner, LoopRegion)
                ),
                None,
            )
            owner = (
                self.function
                if outermost is None or outermost.parent is None
                else outermost.parent.owner
            )
            result[key] = (
                owner,
                region.enclosing_loops(),
                locations[key].owner if key in locations else None,
            )
        return result

    def runtime_values(self, expr, memo):
        """List HIR inputs the caller must lower before tensor calculations."""
        dimensions = {}
        leaves = (
            (expr,)
            if isinstance(expr, SSAValue)
            else tuple(SSAValue(expr, index) for index in range(len(leaves_of(expr.type))))
        )
        for leaf in leaves:
            if leaf not in self.answers:
                continue
            if (refusal := self.answers[leaf].refusal) is not None:
                raise LoweringError(refusal)
            for member in self.answers[leaf].chain:
                if id(member.expr) in memo:
                    continue
                for value in self.answers[member].relation.values.values():
                    dimensions[id(value)] = value
                for loop in self.placement[id(member.expr)][1]:
                    if static_dim_value(loop.start) != 0:
                        dimensions[id(loop.start)] = loop.start
        return tuple(dimensions.values())

    def hir_view(self, call, operands, values):
        """Keep the view authored by this HIR operation, including identity views."""
        region = self.regions[id(call)]
        alias = buffer_alias(call, region, relations_of(call, region))
        if alias is None:
            raise LoweringError(self.answers[SSAValue(call)].refusal)
        return self.view(operands[aliased_operand(call)], alias.relation, call.type, values=values)

    def source(self, leaf, memo):
        """Read a storage binding; HIR evaluation remains exclusively in Visit."""
        windows = []
        current = leaf
        while True:
            if current != leaf and id(current.expr) in memo:
                return current, tuple(reversed(windows))
            answer = self.answers[current]
            if answer.refusal is not None:
                raise LoweringError(answer.refusal)
            if isinstance(answer.partner, int):
                return answer.partner, tuple(reversed(windows))
            if answer.partner is None:
                return current, tuple(reversed(windows))
            if answer.kind is AliasKind.WINDOW:
                windows.append(current)
            current = answer.partner

    def operand(self, value, desired, *, starts=None, sizes=None, stem="tile", whole_tensor=False):
        """Choose an existing operand or the window required at its use."""
        held = value.type
        if not isinstance(held, TensorType):
            raise LoweringError(f"a tensor window cannot use {held!r} as its base")
        if (
            starts is not None
            and tuple(sizes) == tuple(held.shape)
            and all(static_dim_value(start) == 0 for start in starts)
        ):
            starts = None
        if starts is None:
            comparable = replace(desired, layout=resolve_layout(desired.layout, desired.shape))
            actual = (
                held
                if whole_tensor
                else replace(held, layout=participant_layout(held.layout))
                if isinstance(held.layout, ShardLayout)
                else held
            )
            actual = replace(actual, layout=resolve_layout(actual.layout, actual.shape))
            if actual == comparable or (
                not whole_tensor
                and not isinstance(desired.layout, ShardLayout)
                and held.shape == desired.shape
                and held.dtype == desired.dtype
                and held.storage is desired.storage
            ):
                return value
            starts, sizes = tuple(i64_const(0) for _ in held.shape), tuple(held.shape)
        return View(value, starts, sizes, desired, stem, whole_tensor)

    def materialization(self, view, base, loops):
        """Describe one physical TIR view without adding any statements."""
        desired = view.type
        if isinstance(base, Constant) and isinstance(base.type, PointerType):
            return self.view(base.value, None, desired)
        held = base.type
        if len(view.starts) != len(held.shape):
            raise LoweringError(
                f"a rank-{len(held.shape)} buffer is windowed by {len(view.starts)} starts"
            )
        if desired.layout is None:
            desired = replace(desired, layout=resolve_layout(None, desired.shape))
        source = base
        if not view.whole_tensor:
            keys = Tuple(view.starts, type=TupleType(tuple(start.type for start in view.starts)))
            source = Call(
                Slice(sizes=tuple(view.sizes), strides=(1,) * len(view.sizes)),
                (base, keys),
                type=TensorType(tuple(view.sizes), held.dtype, None, held.storage),
            )
            inferred = typeinfer_registry.lookup(Slice)(
                source,
                TypeInferContext(
                    memo={id(arg): (arg, arg.type) for arg in (base, *view.starts)},
                    loops=loops,
                ),
            )
            source.type = getattr(inferred, "type", inferred)
        pointer = Call(PtrOf(), (source,), type=PointerType(held.dtype, held.storage))
        return Call(
            TensorView(layout=desired.layout, shape=tuple(desired.shape)), (pointer,), type=desired
        )

    def window(self, target, desired, starts, *, instruction=False):
        """Derive a window at already lowered coordinates.

        The layout offset is relative to the sliced pointer: reduce the
        displacement modulo the swizzle period; the pointer holds its full offset.
        """
        physical = (
            desired.layout.layout if isinstance(desired.layout, ShardLayout) else desired.layout
        )
        if instruction and not (
            target.type.storage is StorageKind.SMEM
            and isinstance(physical, ComposedLayout)
            and isinstance(physical.inner, Swizzle)
        ):
            return starts, desired
        keys = Tuple(starts, type=TupleType(tuple(value.type for value in starts)))
        cut = Call(
            Slice(sizes=tuple(desired.shape), strides=(1,) * len(desired.shape)),
            (target, keys),
            type=desired,
        )
        inferred = typeinfer_registry.lookup(Slice)(
            cut, TypeInferContext(memo={id(arg): (arg, arg.type) for arg in (target, *starts)})
        )
        type_ = getattr(inferred, "type", inferred)
        layout = type_.layout.layout if isinstance(type_.layout, ShardLayout) else type_.layout
        if isinstance(layout, ComposedLayout) and isinstance(layout.inner, Swizzle):
            swizzle = layout.inner
            period = 1 << (swizzle.bits + swizzle.base + swizzle.shift)
            layout = replace(layout, offset=layout.offset % period)
        if instruction and isinstance(desired.layout, ShardLayout):
            if isinstance(layout, ComposedLayout) and isinstance(
                desired.layout.layout, ComposedLayout
            ):
                layout = replace(desired.layout.layout, offset=layout.offset)
            return starts, replace(desired, layout=replace(desired.layout, layout=layout))
        if isinstance(type_.layout, ShardLayout):
            layout = replace(type_.layout, layout=layout)
        return starts, replace(desired, layout=layout, storage=target.type.storage)

    def view(self, target, address, desired, *, values=None):
        """Derive an exact physical view; a proved composition carries its origin."""
        if isinstance(target, int):
            pointer = Constant(target, type=PointerType(desired.dtype, desired.storage))
            return Call(
                TensorView(
                    dtype=desired.dtype.name,
                    storage=desired.storage,
                    layout=desired.layout,
                    shape=tuple(desired.shape),
                ),
                (pointer,),
                type=desired,
            )

        def addresses(shape, layout):
            flat = flatten(resolve_layout(layout, shape))
            coordinates = [f"c{axis}" for axis in range(len(shape))]
            digits = layout_correspondence(flat.shape, tuple(shape)).coordinates(coordinates)
            linear = (
                " + ".join(
                    f"{stride} * ({digit})"
                    for stride, digit in zip(flat.strides, digits, strict=True)
                )
                or "0"
            )
            offset = layout.offset if isinstance(layout, ComposedLayout) else 0
            return isl.map(f"{{ [{', '.join(coordinates)}] -> [{offset} + {linear}] }}")

        relation = address.relation
        if relation.is_identity() and tuple(desired.shape) == tuple(target.type.shape):
            if not isinstance(desired.layout, ShardLayout):
                layout = target.type.layout
                desired = replace(
                    desired, layout=layout.layout if isinstance(layout, ShardLayout) else layout
                )
            zeros = tuple(i64_const(0) for _ in target.type.shape)
            return View(
                target, zeros, tuple(target.type.shape), desired, "value", whole_tensor=True
            )
        layout = resolve_layout(
            target.type.layout.layout
            if isinstance(target.type.layout, ShardLayout)
            else target.type.layout,
            target.type.shape,
        )
        placed = relation.apply_range(addresses(tuple(target.type.shape), layout))
        try:
            affine = as_multi_aff(placed).get_at(0)
            if isinstance(layout, ComposedLayout) and any(
                affine.get_domain_local_space()
                .get_div(div)
                .get_coefficient_val(isl.dim_type.IN, axis)
                .sgn()
                for div in range(affine.dim(isl.dim_type.DIV))
                for axis in range(len(desired.shape))
            ):
                raise ValueError("address contains coordinate-dependent divisions")
            strides = tuple(
                int(affine.get_coefficient_val(isl.dim_type.IN, axis).num_si())
                for axis in range(len(desired.shape))
            )
        except (ValueError, isl.Error) as error:
            if not isinstance(layout, ComposedLayout):
                raise LoweringError(f"non-affine address {relation}: {error}") from error
            strides = None
        parameters = {name: values[id(value)] for name, value in address.values.items()}
        origin = relation
        for axis in range(relation.dim(isl.dim_type.IN)):
            origin = origin.fix_val(isl.dim_type.IN, axis, isl.val(0))
        point = origin.range()
        starts = tuple(
            isl_to_dim(point.dim_min(axis).gist_params(point.params()), parameters)
            for axis in range(point.dim(isl.dim_type.SET))
        )
        starts = tuple(i64_const(start) if isinstance(start, int) else start for start in starts)
        whole = all(static_dim_value(start) == 0 for start in starts)
        if isinstance(layout, ComposedLayout):
            candidate = desired.layout
            if isinstance(candidate, ComposedLayout) or whole:
                if not isinstance(candidate, ComposedLayout):
                    candidate = replace(layout, outer=Layout(tuple(desired.shape), strides))
                if strides is not None and not affine.dim(isl.dim_type.PARAM):
                    candidate = replace(
                        layout,
                        offset=int(affine.get_constant_val().num_si()),
                        outer=Layout(tuple(desired.shape), strides),
                    )
                elif strides is None:
                    linear = relation.apply_range(
                        addresses(
                            tuple(target.type.shape),
                            Layout(
                                tuple(target.type.shape),
                                compact_row_major(tuple(target.type.shape)),
                            ),
                        )
                    )
                    linear = as_multi_aff(linear).get_at(0)
                    steps = tuple(
                        int(linear.get_coefficient_val(isl.dim_type.IN, axis).num_si())
                        for axis in range(len(desired.shape))
                    )
                    offset = (
                        as_multi_aff(
                            origin.apply_range(addresses(tuple(target.type.shape), layout))
                        )
                        .get_at(0)
                        .get_constant_val()
                        .num_si()
                    )
                    candidate = replace(
                        layout,
                        outer=composition(
                            layout.outer, Layout(tuple(desired.shape), steps), major="row"
                        ),
                        offset=int(offset),
                    )
                declared = addresses(tuple(desired.shape), candidate).intersect_domain(
                    relation.domain()
                )
                if candidate.inner != layout.inner or not declared.is_equal(placed):
                    raise LoweringError(f"composed view does not preserve exact address {relation}")
                desired = replace(desired, layout=candidate)
                whole = True
                starts = tuple(i64_const(0) for _ in starts)
            else:
                desired = self.window(target, desired, starts)[1]
        else:
            offset = isl.map.from_domain_and_range(
                relation.domain(),
                origin.apply_range(addresses(tuple(target.type.shape), layout)).range(),
            )
            declared = addresses(tuple(desired.shape), Layout(tuple(desired.shape), strides))
            declared = declared.flat_range_product(offset).apply_range(
                isl.map("{ [a, b] -> [a + b] }")
            )
            if not declared.is_equal(placed):
                raise LoweringError(f"non-affine address {relation}")
            if not isinstance(desired.layout, ShardLayout):
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
        return View(target, starts, tuple(sizes), desired, "value", whole_tensor=whole)


@dataclass
class IssuePlan:
    """Instruction selection and one shared geometric projection, without admission."""

    def op(self, call):
        if isinstance(call.target, ScheduleOp):
            return call.target.op
        candidates = candidate_ops(type(call.target))
        if not candidates:
            raise LoweringError(
                f"{_label(call)} uses unknown HIR call {type(call.target).__name__}"
            )
        op = sole_candidate(call)
        if op is None:
            available = ", ".join(op_identifier(candidate) for candidate in candidates)
            raise LoweringError(
                f"{_label(call)} has instruction candidates [{available}] but no automatic "
                "selection; write tf.schedule to choose the instruction and its attributes"
            )
        return op

    def plan(self, call, op, mesh):
        if isinstance(call.target, ScheduleOp):
            types = tuple(arg.type for arg in call.args)
            repeat, order = call.target.repeat, call.target.order
        else:
            _attributes, types = candidate_operands(call, type(op))
            repeat = order = None
        try:
            projection = project(op, types, mesh, repeat=repeat, order=order)
            owner = next(
                (
                    participant_layout(type_.layout).mesh
                    for type_ in types
                    if isinstance(type_.layout, ShardLayout)
                ),
                None,
            )
            return projection, types, owner
        except (ScheduleRefusal, ValueError) as error:
            raise LoweringError(str(error)) from error

    def source_layout(self, type_):
        return resolve_layout(type_.layout, type_.shape)

    def points(self, projection, op):
        serial = tuple(
            count // bind for count, bind in zip(projection.repeat, projection.bind, strict=True)
        )
        step = tuple(
            size * bind for size, bind in zip(projection.shape, projection.bind, strict=True)
        )
        axes = tuple(
            projected_axes(boundary)
            for boundary in operand_relations(op, projection.operands)[: len(projection.params)]
        )
        for counters in product(*(range(serial[axis]) for axis in projection.order)):
            offsets = tuple(
                dict(zip(projection.order, counters, strict=True))[axis] * step[axis]
                for axis in range(len(serial))
            )
            yield tuple(tuple(i64_const(offsets[axis]) for axis in mapped) for mapped in axes)


@dataclass(kw_only=True)
class LoweringContext(TypeInferContext):
    """Carry storage answers and shared emission state through lexical visits."""

    storage: StoragePlan
    issues: IssuePlan
    builder: TirBuilder
    names: Names
    outputs: list[tuple[tuple[int, ...], Var]] = field(default_factory=list)
    scratch: list[Var] = field(default_factory=list)
    allocations: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    authored_values: dict = field(default_factory=dict)

    def value(self, leaf):
        result = self.memo[id(leaf.expr)][1]
        pending = [result]
        index = 0
        while pending:
            value = pending.pop()
            if isinstance(value, Tuple):
                pending.extend(reversed(value.elements))
            elif index == leaf.index:
                return value
            else:
                index += 1
        raise LoweringError(f"missing lowered leaf {leaf.index} of {_label(leaf.expr)}")

    def result(self, expr):
        """Select result storage using completed Visit results and root declarations."""
        values = {key: value for key, (_expr, value) in self.memo.items()}
        values.update(
            (id(value), self.dim(value, self.memo))
            for value in self.storage.runtime_values(expr, self.memo)
        )
        fields = []
        for index, desired in enumerate(leaves_of(expr.type)):
            leaf = SSAValue(expr, index)
            source, windows = self.storage.source(leaf, self.memo)
            if isinstance(source, int):
                result = self.outputs[source][1]
            elif id(source.expr) in self.memo:
                result = self.value(source)
            else:
                result = self.allocations.get(source)
                if result is None:
                    result = self.builder.declare(source, ctx=self, values=values)
                    self.allocations[source] = result
            if desired.storage is StorageKind.GMEM and desired.dtype != result.type.dtype:
                raise LoweringError("gmem cross-dtype reuse has no offset-view representation")
            if desired.storage is StorageKind.SMEM and desired.dtype != result.type.dtype:
                root = self.storage.answers[leaf].chain[-1]
                result = self.builder.declare(root, ctx=self, values=values, desired=desired)
            for window in windows:
                result = self.storage.view(
                    result,
                    self.storage.answers[window].relation,
                    leaves_of(window.expr.type)[window.index],
                    values=values,
                )
                result = self.storage.operand(
                    result.base,
                    result.type,
                    starts=result.starts,
                    sizes=result.sizes,
                    stem=result.stem,
                    whole_tensor=result.whole_tensor,
                )
            fields.append(result)
        fields = iter(fields)

        def pack(type_):
            return (
                Tuple(tuple(pack(field) for field in type_.fields), type=type_)
                if isinstance(type_, TupleType)
                else next(fields)
            )

        return pack(expr.type)

    def dim(self, value, values) -> Expr:
        number = static_dim_value(value)
        if number is not None:
            return i64_const(number)
        if isinstance(value, Var):
            known = values.get(id(value))
            if known is None:
                raise LoweringError(f"dimension uses unbound value {value.name!r}")
            return self.builder.value(known[1], ctx=self)
        if isinstance(value, Call) and isinstance(value.target, MeshCoord):
            return Call(
                MeshCoord(mesh=value.target.mesh),
                value.args,
                type=value.type,
            )
        if isinstance(value, Call) and isinstance(value.target, HirBinary):
            dim_op = _DIM_BINARY.get(value.target.kind)
            if dim_op is None or not isinstance(value.type, TensorType) or value.type.shape != ():
                raise LoweringError(
                    f"dimension uses unsupported binary operation {value.target.kind}"
                )
            return simplify_dim(dim_op, tuple(self.dim(arg, values) for arg in value.args))
        if isinstance(value, Call) and is_dim_op_call(value):
            return Call(
                value.target, tuple(self.dim(arg, values) for arg in value.args), type=value.type
            )
        if isinstance(value, Call):
            return self.builder.value(values[id(value)][1], ctx=self)
        if isinstance(value, Expr):
            return values.get(id(value), (value, value))[1]
        raise LoweringError(f"dimension {value!r} is not lowerable")


@dataclass
class Lowering(ValuationVisitor[Expr | View]):
    """Mechanical lowering of one analyzed HIR function."""

    context: TypeInferContext
    authored: Function
    callees: dict[int, Lowering] = field(default_factory=dict)

    def visit(self, expr, ctx=None, *, dimension=False):
        if ctx is None:
            ctx = self.locations.get(id(expr), self.context)
            if isinstance(expr, (MeshRegion, LoopRegion)):
                ctx = ctx.parent
        if dimension:
            if isinstance(expr, Expr):
                for value in collect_exprs(expr):
                    if isinstance(value, Call) and not (
                        is_dim_op_call(value)
                        or isinstance(value.target, MeshCoord)
                        or isinstance(value.target, HirBinary)
                        and value.type.shape == ()
                    ):
                        self.visit(value, ctx=ctx)
            return ctx.dim(expr, self._memo)
        known = ctx.memo.get(id(expr))
        if known is not None:
            return known[1]
        ctx.names.pair(expr, ctx)
        if isinstance(expr, Constant):
            return expr
        if isinstance(expr, Var) and id(expr) not in self._memo:
            raise LoweringError(f"unbound HIR value {expr.name!r} reached lowering")
        if isinstance(expr, Tuple):
            elements = tuple(self.visit(element, ctx=ctx) for element in expr.elements)
            result = Tuple(elements, type=TupleType(tuple(element.type for element in elements)))
        else:
            if isinstance(expr, (LoopRegion, MeshRegion)) and id(expr) not in self._memo:
                values = tuple(self.visit(arg, ctx=ctx) for arg in expr.args)
                for index, (param, value) in enumerate(zip(expr.params, values, strict=True)):
                    if not isinstance(expr, LoopRegion) or index >= len(expr.yield_values):
                        ctx.memo[id(param)] = (param, value)
            result = super().visit(expr, ctx)
        self._memo[id(expr)] = (expr, result)
        return result

    def __post_init__(self) -> None:
        super().__init__()
        self.context = LoweringContext(
            **vars(self.context),
            storage=StoragePlan(
                self.context.frame.function, self.context.frame, self.context.current_mesh
            ),
            builder=TirBuilder([TirScope(region=self.context.frame.function)]),
            names=Names({param.name for param in self.authored.params}),
            issues=IssuePlan(),
        )
        ctx = self.context
        ctx.memo = self._memo
        if ctx.frame.function.body is not None and self.authored.body is not None:
            ctx.authored_values[id(ctx.frame.function.body)] = self.authored.body
        self.locations = self.index_contexts(ctx.frame.function, self.context)
        for param in ctx.frame.function.params:
            physical = Var(param.name, type=param.type, is_const=param.is_const)
            ctx.parameters[id(param)] = physical
            self._memo[id(param)] = (param, physical)

    def run(self) -> PrimFunction:
        ctx = self.context
        if ctx.frame.function.body is None:
            raise LoweringError(f"{ctx.frame.function.name!r} has no body to lower")
        returns = tuple(leaf_paths(ctx.frame.function.return_type))
        for index, (path, type_) in enumerate(returns):
            if not isinstance(type_, TensorType):
                raise LoweringError(f"return leaf {path!r} has non-tensor type {type_!r}")
            leaf = SSAValue(ctx.frame.function.body, index)
            if leaf not in ctx.storage.answers:
                raise LoweringError(f"return leaf {path!r} cannot be placed on its output: None")
            if (refusal := ctx.storage.answers[leaf].refusal) is not None:
                raise LoweringError(refusal)
        for path, type_ in returns:
            output_type = (
                replace(type_, storage=StorageKind.GMEM)
                if type_.storage is StorageKind.UMAT
                else type_
            )
            output = Var(ctx.names.fresh("out"), type=output_type)
            ctx.outputs.append((path, output))
        self.visit(ctx.frame.function.body, ctx=ctx)
        statements = ctx.builder.build(ctx.builder.blocks[0])

        params = tuple(ctx.parameters[id(param)] for param in ctx.frame.function.params)
        return PrimFunction(
            ctx.frame.function.name,
            (*params, *ctx.scratch, *(output for _path, output in ctx.outputs)),
            statements,
            output_count=len(ctx.outputs),
            target=self.context.frame.target,
        )

    def visit_MeshRegion(self, region: MeshRegion, ctx):
        mesh = region.mesh
        binding = Var(
            ctx.names.fresh("cta" if region is ctx.frame.function.body else "scope"), type=_BINDING
        )
        with ctx.builder.region(mesh=mesh, binding=binding, owner=region) as scope:
            result = self.visit(region.body, self.region_context(region, ctx))
            scope.result = result
        return result

    def visit_LoopRegion(self, loop: LoopRegion, ctx) -> Expr:
        carried_values = []
        for slot, param in enumerate(loop.params[: len(loop.yield_values)]):
            initial = self.visit(loop.args[slot], ctx=ctx)
            carried = ctx.result(param)
            if not any(
                alias.kind is AliasKind.CARRY_INIT and alias.target == SSAValue(loop.args[slot])
                for alias in ctx.storage.answers[SSAValue(param)].aliases
            ):
                ctx.builder.transfer(initial, carried, ctx=ctx, source_type=loop.args[slot].type)
            self._memo[id(param)] = (param, carried)
            carried_values.append(carried)
        init = tuple(carried_values)
        induction = Var(loop.induction_var.name, type=_INDEX)
        self._memo[id(loop.induction_var)] = (loop.induction_var, induction)
        with ctx.builder.region(
            loop=(
                induction,
                self.visit(loop.start, ctx=ctx, dimension=True),
                self.visit(loop.extent, ctx=ctx, dimension=True),
                self.visit(loop.step, ctx=ctx, dimension=True),
            ),
            owner=loop,
        ) as scope:
            inner = self.region_context(loop, ctx)
            self.visit(loop.body, inner)
            yielded = tuple(self.visit(value, inner) for value in loop.yield_values)
            for param, source, carried, value in zip(
                loop.params, loop.yield_values, init, yielded, strict=False
            ):
                if not any(
                    alias.kind is AliasKind.CARRY_YIELD and alias.target == SSAValue(param)
                    for alias in ctx.storage.answers[SSAValue(source)].aliases
                ):
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

    def visit_Call(self, call: Call, ctx) -> Expr:
        target = call.target
        if (
            is_dim_op_call(call)
            or isinstance(target, MeshCoord)
            or (isinstance(target, HirBinary) and call.type.shape == ())
        ):
            return self.visit(call, ctx=ctx, dimension=True)
        values = {}
        if SSAValue(call) in ctx.storage.answers:
            if (refusal := ctx.storage.answers[SSAValue(call)].refusal) is not None:
                raise LoweringError(refusal)
            values = {
                id(value): self.visit(value, ctx=ctx, dimension=True)
                for value in ctx.storage.runtime_values(call, ctx.memo)
            }
        operands = tuple(self.visit(arg, ctx=ctx) for arg in call.args)
        if self.callee_frame(call, ctx.frame) is not None:
            return self._lower_call(call, operands, ctx=ctx)
        name = type(target).__name__
        if name == "TupleGetItem":
            source, index = operands
            return (
                source.elements[index.value]
                if isinstance(source, Tuple) and isinstance(index, Constant)
                else Call(target, (source, index), type=call.type)
            )
        if name in ("Slice", "Reshape", "Bitcast") or (
            name == "Reshard" and aliased_operand(call) is not None
        ):
            return ctx.storage.hir_view(call, operands, values)
        if name in ("Empty", "Zeros", "FullLike"):
            if name == "Empty" and call.type.storage is StorageKind.RMEM and target.buffers != 1:
                raise LoweringError(f"{_label(call)} staged buffer has no owning scope")
            result = ctx.result(call)
            if name != "Empty":
                scalar = Constant(
                    getattr(target, "value", 0.0), type=TensorType.scalar(call.type.dtype)
                )
                ctx.builder.transfer(scalar, result, op=Fill(), ctx=ctx)
            return result
        if name == "IndexSelect":
            raise LoweringError(
                f"{_label(call)} is an unscheduled IndexSelect; write "
                "tf.schedule((x, tf.empty(...), idx), op=T.copy_async(fill=...))"
            )
        if name == "InsertSlice":
            return self.visit_InsertSlice(call, operands, ctx)
        op = ctx.issues.op(call)
        return getattr(self, f"visit_{type(op).__name__}", self.instruction)(
            call, op, operands, ctx
        )

    def _lower_call(self, call: Call, args, *, ctx) -> Expr | View:
        frame = self.callee_frame(call, ctx.frame)
        lowering = self.callees.get(id(frame.function))
        if lowering is None:
            lowering = Lowering(
                TypeInferContext(frame=frame, owner=frame.function, current_mesh=ctx.current_mesh),
                origin_of(frame.function) or frame.function,
                self.callees,
            )
            lowering.lowered = lowering.run()
            self.callees[id(frame.function)] = lowering
        destination = ctx.result(call)
        scratch = []
        for param in lowering.context.scratch:
            forwarded = Var(ctx.names.fresh(param.name), type=param.type)
            ctx.scratch.append(forwarded)
            scratch.append(forwarded)
        target = call.target
        if isinstance(target, Function):
            target = lowering.lowered
        elif isinstance(target, Call):
            target = replace(
                target, args=(target.args[0], self.visit(target.args[1], ctx=ctx, dimension=True))
            )
        params = tuple(
            param
            for param in lowering.lowered.params
            if isinstance(call.target, Function) or not param.is_const
        )
        passed = []
        for value, param in zip((*args, *scratch, destination), params, strict=True):
            passed.append(
                ctx.builder.value(
                    ctx.storage.operand(value, param.type, whole_tensor=True, stem="argument"),
                    ctx=ctx,
                )
                if isinstance(param.type, TensorType)
                else ctx.builder.value(value, ctx=ctx)
            )
        ctx.builder.add(Evaluate(target, tuple(passed)))
        return destination

    def visit_InsertSlice(self, call: Call, operands, ctx) -> Expr:
        target, update = operands[:2]
        result = ctx.result(call)
        if not any(
            alias.kind is AliasKind.INSERT_SLICE and alias.target == SSAValue(call.args[0])
            for alias in ctx.storage.answers[SSAValue(call)].aliases
        ):
            ctx.builder.transfer(target, result, ctx=ctx, source_type=call.args[0].type)
        if any(
            alias.kind is AliasKind.WINDOW and alias.target == SSAValue(call)
            for alias in ctx.storage.answers[SSAValue(call.args[1])].aliases
        ):
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

    def instruction(self, call: Call, op, operands, ctx) -> Expr:
        mesh = ctx.enclosing_mesh()
        plan, types, owner = ctx.issues.plan(call, op, mesh)
        params = plan.params
        reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
        writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
        produced = tuple(param for param in writes if not param.effect & MemoryEffect.READ)
        destination_operand = (
            call.args[params.index(produced[0])]
            if isinstance(call.target, ScheduleOp) and produced
            else None
        )

        written = ctx.result(call) if produced else None
        read_values = {}
        supplied = params if isinstance(call.target, ScheduleOp) else reads
        for index, (param, operand) in enumerate(zip(supplied, call.args, strict=True)):
            value = written if operand is destination_operand else operands[index]
            if (
                not isinstance(call.target, ScheduleOp)
                and operand.type != types[params.index(param)]
            ):
                value = ctx.storage.operand(value, types[params.index(param)], whole_tensor=True)
            if (
                isinstance(op, CopyAsync)
                and is_indexed_copy(call.args)
                and param.name == "src"
                and (
                    value.type.layout is None
                    or any(
                        value is self._memo[id(argument)][1]
                        for argument in ctx.frame.function.params
                    )
                )
            ):
                type_ = replace(
                    value.type,
                    layout=ctx.issues.source_layout(value.type),
                )
                value = ctx.storage.operand(value, type_, stem="source", whole_tensor=True)
            if (
                isinstance(value.type, TensorType)
                and value.type.shape == ()
                and value.type.storage is StorageKind.UMAT
            ):
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
        atom = getattr(op, "atom", None)
        if atom is None:
            if mesh is None:
                raise LoweringError(f"{_label(call)} instruction issue has no lexical mesh")
            with ctx.builder.region(
                mesh=owner, binding=Var(ctx.names.fresh("threads"), type=_BINDING)
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
                    if param.kind == "attribute" and (not param.has_default or param.name == "fill")
                }
                ctx.builder.add(Evaluate(type(op)(**attributes), tuple(issued)))
            result_param = next(param for param in params if param.effect & MemoryEffect.WRITE)
            return (
                read_values[result_param.name]
                if result_param.effect & MemoryEffect.READ
                else written
            )
        for windows in ctx.issues.points(plan, op):
            issued = []
            for (role, value), desired, starts in zip(
                issued_operands, plan.issued, windows, strict=True
            ):
                logical_type = types[
                    next(index for index, param in enumerate(params) if param.name == role)
                ]
                if tuple(logical_type.shape) == tuple(desired.shape):
                    item = ctx.builder.value(
                        ctx.storage.operand(value, desired, stem=f"{role}_view"), ctx=ctx
                    )
                else:
                    base = ctx.builder.value(value, ctx=ctx)
                    starts, physical = ctx.storage.window(base, desired, starts, instruction=True)
                    item = ctx.builder.value(
                        ctx.storage.operand(
                            base, physical, starts=starts, sizes=desired.shape, stem=f"{role}_view"
                        ),
                        ctx=ctx,
                    )
                issued.append(item)
            ctx.builder.add(Evaluate(op, tuple(issued)))
        result_param = next(param for param in params if param.effect & MemoryEffect.WRITE)
        if result_param.effect & MemoryEffect.READ:
            return read_values[result_param.name]
        if written is None:
            raise LoweringError(f"{_label(call)} produced no addressable result")
        return written


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
            TypeInferContext(
                frame=CallFrame(result.module, result.function), owner=result.function
            ),
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
            node.functions = tuple(
                lowered[id(fn)].lowered if id(fn) in lowered else fn for fn in node.functions
            )

        lower_module(result.module)
        return result.module


__all__ = ["ConvertHIRToTIR", "LoweringError"]
