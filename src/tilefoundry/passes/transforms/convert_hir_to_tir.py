"""Lower scheduled HIR to effect-form TIR.

The pass consumes a memory analysis and lowers each retained definition once.
StoragePlan interprets raw allocation decisions and registered access
relations; the visitor writes the resulting declarations and statements.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import product
from math import prod
from types import MappingProxyType
from typing import ClassVar

import isl

from tilefoundry import analysis, services, utils
from tilefoundry import target as tilefoundry_target
from tilefoundry.interfaces import access, schedule, typeinfer, valuation
from tilefoundry.ir import core, expr, hir, tir, types
from tilefoundry.ir import dim as ir_dim
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.hir import math, tensor
from tilefoundry.ir.tir import memory
from tilefoundry.ir.types import layout_algebra
from tilefoundry.passes import options as passes_options
from tilefoundry.passes import pass_base
from tilefoundry.target import cpu

INDEX = types.TensorType.from_umat_scalar()
BINDING = types.TensorType.from_scalar(expr.DType.i64, storage=expr.StorageKind.RMEM)
DIM_BINARY = MappingProxyType({
    expr.BinaryKind.ADD: "add",
    expr.BinaryKind.SUB: "sub",
    expr.BinaryKind.MUL: "mul",
    expr.BinaryKind.FLOOR_DIV: "floordiv",
    expr.BinaryKind.MOD: "mod",
})


class LoweringError(ValueError):
    """A checked HIR construct has no sound TIR representation."""


@dataclass(frozen=True, eq=False)
class View:
    """An address recipe, materialized only in its consuming scope."""

    base: expr.Expr | View
    starts: tuple[expr.Expr, ...]
    sizes: tuple
    type: types.TensorType
    stem: str = "tile"
    whole_tensor: bool = False


def covers(starts, sizes, shape) -> bool:
    """Whether a window at *starts* of *sizes* is the whole operand of *shape*."""
    return tuple(sizes) == tuple(shape) and all(expr.static_dim_value(start) == 0 for start in starts)


def cut(base, starts, sizes, loops=()) -> expr.Call:
    """*base* windowed at *starts* to *sizes*, with the type Slice infers for it."""
    keys = expr.Tuple(starts, type=types.TupleType(tuple(start.type for start in starts)))
    window = expr.Call(
        tensor.Slice(sizes=tuple(sizes), strides=(1,) * len(sizes)), (base, keys), type=base.type
    )
    inferred = typeinfer.registry.require(tensor.Slice)(
        window,
        typeinfer.TypeInferContext(memo={id(arg): (arg, arg.type) for arg in (base, *starts)}, loops=loops),
    )
    return expr.Call(window.target, window.args, type=getattr(inferred, "type", inferred))


def addresses(shape, layout):
    """The isl map from coordinates of *shape* to addresses of *layout*."""
    flat = types.flatten(layout)
    coordinates = ir_dim.layout_to_isl_map(tuple(shape), flat, {}, divided=lambda _: ())
    offset = layout.offset if isinstance(layout, types.ComposedLayout) else 0
    linear = " + ".join(f"{stride} * p{axis}" for axis, stride in enumerate(flat.strides)) or "0"
    address = isl.map(f"{{ [{', '.join(f'p{i}' for i in range(len(flat.shape)))}] -> [{offset} + {linear}] }}")
    return coordinates.apply_range(address)


def lowers_to_dim(call: expr.Call) -> bool:
    """Whether *call* computes a dimension: a dim op, a mesh coordinate or scalar arithmetic."""
    target = call.target
    return (
        expr.is_dim_op_call(call)
        or isinstance(target, ir_dim.MeshCoord)
        or (isinstance(target, math.Binary) and call.type.shape == ())
    )


def selected_op(call: expr.Call, target) -> expr.Op | None:
    """The instruction *call* issues on *target*: its schedule's, else the sole candidate."""
    if isinstance(call.target, hir.schedule.ScheduleOp):
        return call.target.op
    return schedule.sole_candidate(call, target=target)


def operand_types(call: expr.Call, op: expr.Op) -> tuple:
    """The types *op* is issued with, one per parameter it takes for *call*."""
    if isinstance(call.target, hir.schedule.ScheduleOp):
        return tuple(arg.type for arg in call.args)
    return schedule.candidate_operands(call, type(op))[1]


def participants(layouts) -> types.Mesh | None:
    """The first physical participant mesh among *layouts*, if any."""
    for layout in layouts:
        if isinstance(layout, types.ShardLayout):
            physical = ir_utils.physical_layout(layout)
            if isinstance(physical, types.ShardLayout):
                return ir_utils.participant_layout(physical).mesh
    return None


def copy_mesh(type_list) -> types.Mesh | None:
    """The mesh a copy between operands of *types* runs on, None for the enclosing one."""
    return participants(type_.layout for type_ in type_list)


def first_thread(enclosing: types.Mesh | None, topologies) -> types.Mesh | None:
    """A slice holding the first thread of *enclosing*, or of the CTA when there is none.

    None when that level holds one thread already, or when the first thread has
    no constant position: *enclosing* is a slice whose start is not a number, or
    one over several levels whose sizes are not all numbers, so its start cannot
    be split into the start of each level.
    """
    if enclosing is not None and "thread" not in types.topology_names(enclosing):
        enclosing = None
    if enclosing is None:
        topology = next((one for one in topologies or () if one.name == "thread"), None)
        if topology is None or topology.size == 1:
            return None
        start = 0
    else:
        if isinstance(enclosing.layout, types.ComposedLayout):
            sizes = [topology.size for topology in enclosing.topologies]
            if not isinstance(enclosing.layout.offset, int):
                return None
            if len(sizes) > 1 and not all(isinstance(size, int) for size in sizes):
                return None
        level = next(part for part in types.separate(enclosing) if part.topologies[0].name == "thread")
        topology = level.topologies[0]
        if types.size(ir_utils.device_layout(level)) == 1:
            return None
        start = level.layout.offset if isinstance(level.layout, types.ComposedLayout) else 0
    return types.Mesh((topology,), types.ComposedLayout(None, start, types.Layout((1,), (1,))), ("writer",))


def transfer_meshes(type_list, destination, around, topologies) -> tuple[types.Mesh, ...]:
    """The meshes of the regions lowering opens around a copy or fill it adds, outermost first.

    *around* holds the meshes of the regions around it, innermost first. A
    sharded operand gives the mesh that holds it. Unless that mesh splits the
    threads, every thread of the innermost of *around* that names threads would
    write the whole of its share of *destination*. In shared memory, which
    those threads share, the first of them writes it alone.
    """
    mesh = copy_mesh(type_list)
    held = () if mesh is None else (mesh,)
    if getattr(destination, "storage", None) is not expr.StorageKind.SMEM:
        return held
    if mesh is not None and "thread" in types.topology_names(mesh):
        return held
    enclosing = next(
        (one for one in around if one is not None and "thread" in types.topology_names(one)), None
    )
    writer = first_thread(enclosing, topologies)
    return held if writer is None else (*held, writer)


def issue_owner(type_list) -> types.Mesh | None:
    """The mesh that owns an instruction issued with operands of *types*, if one does.

    ``Partial`` attributes count as ``Broadcast`` here.
    """
    return participants(
        replace(
            type_.layout,
            attrs=tuple(
                types.Broadcast() if isinstance(attr, types.Partial) else attr for attr in type_.layout.attrs
            ),
        )
        if isinstance(type_.layout, types.ShardLayout)
        else type_.layout
        for type_ in type_list
    )


def accepted_aliases(value: analysis.AllocationValue) -> tuple[analysis.BufferAlias, ...]:
    """Accepted links that lowering executes; a register IN_PLACE tie shares no storage."""
    return tuple(
        alias
        for alias in value.aliases
        if value.lifetime.memory_level != "rmem" or alias.kind is not analysis.AliasKind.IN_PLACE
    )


@dataclass
class TirScope:
    mesh: types.Mesh | None = None
    region: expr.Expr | None = None
    bound: dict = field(default_factory=dict)
    views: dict = field(default_factory=dict)
    entries: list = field(default_factory=list)
    result: expr.Expr | View | None = None

    def bind_var(self, var) -> None:
        """Mark *var* as bound inside this scope."""
        self.bound[id(var)] = var

    def remember_view(self, view, var) -> None:
        """Keep the Var that materialized *view* in this scope."""
        self.views[id(view)] = (view, var)

    def yield_value(self, value) -> None:
        """Record what this scope's region evaluates to."""
        self.result = value


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
        self.blocks[-1].bind_var(var)
        return var

    def build(self, region=None):
        return core.Sequential(tuple(
            core.LetStmt(*entry) if isinstance(entry, tuple) else entry
            for entry in (region or self.blocks[-1]).entries
        ))

    @contextmanager
    def region(self, *, owner=None, mesh=None, binding=None, loop=None):
        if mesh is not None and owner is None:
            current = self.blocks[-1].mesh
            if current is not None and current == mesh:
                mesh = None
        if mesh is not None or loop is not None:
            region = TirScope(mesh if mesh is not None else self.blocks[-1].mesh, owner)
            if binding is not None:
                region.bind_var(binding)
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
            region.bind_var(loop[0])
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
                if isinstance(value, expr.Var) and id(value) in region.bound:
                    raise LoweringError(
                        f"value {value.name!r} escapes region mesh {mesh or region.mesh!r}"
                    )
                if isinstance(value, View):
                    pending.extend((value.base, *value.starts))
                elif isinstance(value, expr.Expr) and not isinstance(value, expr.Var):
                    pending.extend(core.expr_children(value))
        if mesh is not None:
            self.add(core.MeshScope(mesh, binding, self.build(region)))
        elif loop is not None:
            self.add(core.For(*loop, self.build(region)))

    def declare(self, leaf, *, ctx, values, desired=None):
        """Declare one allocation root in its owning scope."""
        node = leaf.expr
        type_ = desired or ir_utils.leaves_of(node.type)[leaf.index]
        buffers = getattr(node.target, "buffers", 1) if isinstance(node, expr.Call) else 1
        owner = (
            ctx.storage.placement[id(node)][0]
            if buffers > 1
            else ctx.storage.placement.get(id(node), (None, (), None))[2]
            if type_.storage is expr.StorageKind.RMEM
            else None
        )
        with ctx.builder.region(owner=owner):
            layout = type_.layout.layout if isinstance(type_.layout, types.ShardLayout) else type_.layout
            storage_type = replace(type_, layout=layout)
            stem = ctx.names.fresh(
                node.name if isinstance(node, expr.Var) else "value",
                node,
            )
            if type_.storage is expr.StorageKind.RMEM:
                storage_type = type_
                var = expr.Var(stem, type=storage_type)
                ctx.builder.bind(
                    var, expr.Call(memory.AllocTensor(tensor_type=storage_type), (), type=storage_type)
                )
                return var
            if storage_type.storage not in (expr.StorageKind.SMEM, expr.StorageKind.GMEM):
                raise LoweringError(
                    f"{label(node)} has addressable result in unsupported storage {storage_type.storage}"
                )
            offsets = ctx.storage.answers[leaf].offsets
            if not offsets:
                raise LoweringError(
                    f"{label(node)} has an addressable {storage_type.storage} result but no offsets"
                )
            if len(offsets) != buffers:
                raise LoweringError(
                    f"{label(node)} declares {buffers} buffer(s), but memory analysis placed {len(offsets)}"
                )
            if storage_type.storage is expr.StorageKind.GMEM:
                if buffers != 1:
                    raise LoweringError(f"{label(node)} uses staged internal gmem")
                elements = prod(tuple(storage_type.shape))
                scratch = ctx.storage.scratch[1][leaf]
                storage_type = replace(
                    storage_type,
                    layout=types.Layout(tuple(storage_type.shape)),
                )
                view = View(scratch, (ir_dim.i64_const(0),), (elements,), storage_type, stem)
                return view
            if len(offsets) == 1:
                pointer = expr.Constant(offsets[0], type=types.PointerType(storage_type.dtype, storage_type.storage))
                return View(pointer, (), tuple(storage_type.shape), storage_type, stem)
            fields = tuple(
                self.bind(expr.Var(ctx.names.fresh(stem), type=storage_type), ctx.storage.view(offset, None, storage_type))
                for offset in offsets
            )
            stages = expr.Var(ctx.names.fresh(f"{stem}_stages"), type=types.TupleType(tuple(field.type for field in fields)))
            self.bind(stages, expr.Tuple(fields, type=stages.type))
            loops = ctx.storage.placement[id(node)][1]
            if any(
                expr.static_dim_value(bound) is None
                for loop in loops[1:]
                for bound in (loop.start, loop.extent, loop.step)
            ):
                key = tuple(id(loop) for loop in loops)
                if key not in self.stage_counters:
                    counter = expr.Var(ctx.names.fresh("stage_counter"), type=BINDING)
                    one = expr.Var(ctx.names.fresh("stage_one"), type=BINDING)
                    with ctx.builder.region(owner=ctx.storage.placement[id(node)][0]):
                        for value, number in ((counter, 0), (one, 1)):
                            ctx.builder.bind(
                                value,
                                expr.Call(memory.AllocTensor(tensor_type=BINDING), (), type=BINDING),
                            )
                            ctx.builder.add(
                                core.Evaluate(memory.Fill(), (value, expr.Constant(number, type=BINDING)))
                            )
                    self.stage_counters[key] = (counter, one)
                counter, _ = self.stage_counters[key]
                index = services.simplify_dim("mod", (counter, len(offsets)))
            else:
                index = ir_dim.i64_const(0)
                for depth, loop in enumerate(loops):
                    induction = values.get(id(loop.induction_var))
                    if not isinstance(induction, expr.Var):
                        raise LoweringError(f"{label(node)} staged selection has no owning induction")
                    start = expr.static_dim_value(loop.start)
                    stop = expr.static_dim_value(loop.extent)
                    step = expr.static_dim_value(loop.step)
                    if step is None:
                        raise LoweringError(
                            f"{label(node)} staged selection needs a literal step for "
                            f"{loop.induction_var.name!r}"
                        )
                    if depth:
                        trips = max(0, (stop - start + step - 1) // step)
                        index = services.simplify_dim("mul", (index, trips))
                    moved = (
                        induction
                        if start == 0
                        else services.simplify_dim("sub", (induction, values[id(loop.start)]))
                    )
                    ordinal = moved if step == 1 else services.simplify_dim("floordiv", (moved, step))
                    index = ordinal if depth == 0 else services.simplify_dim("add", (index, ordinal))
                if loops:
                    index = services.simplify_dim("mod", (index, len(offsets)))
            selected = expr.Call(tensor.TupleGetItem(), (stages, index), type=storage_type)
            return View(selected, (), tuple(storage_type.shape), storage_type, stem)

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
                if isinstance(base, expr.Constant) and isinstance(base.type, types.PointerType)
                else ctx.names.fresh(value.stem)
            )
            result = self.bind(expr.Var(stem, type=expression.type), expression)
            scope.remember_view(value, result)
        return scope.views[id(value)][1]

    def transfer(
        self,
        source,
        target,
        *,
        starts: tuple[expr.Expr, ...] | None = None,
        sizes: tuple | None = None,
        desired: types.TensorType | None = None,
        ctx,
        op=None,
        source_type=None,
    ) -> None:
        source_type = source.type if source_type is None else source_type
        if not isinstance(source_type, types.TensorType) or not isinstance(target.type, types.TensorType):
            raise LoweringError("copy lowering requires tensor operands")
        sharded = copy_mesh((source_type, target.type))
        meshes = transfer_meshes(
            (source_type, target.type),
            target.type,
            [block.mesh for block in reversed(self.blocks)],
            ctx.topologies,
        )
        with ExitStack() as regions:
            for mesh in meshes or (None,):
                binding = expr.Var(ctx.names.fresh("threads"), type=BINDING)
                regions.enter_context(self.region(mesh=mesh, binding=binding))
            if isinstance(op, memory.Fill):
                self.add(core.Evaluate(op, (self.value(target, ctx=ctx), source)))
                return
            read = (
                self.value(source, ctx=ctx)
                if sharded is None
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
            ctx.builder.add(core.Evaluate(memory.Copy(), (read, written)))


@dataclass
class Names:
    """Stable authored names plus deterministic generated-name suffixes."""

    used: set[str]

    def fresh(self, stem: str, node=None) -> str:
        binding = expr.get_metadata(node, expr.BindingMetadata) if node is not None else None
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


def label(call: expr.Expr) -> str:
    if isinstance(call, expr.Var):
        return call.name
    binding = expr.get_metadata(call, expr.BindingMetadata)
    return binding.name if binding is not None else type(call.target if isinstance(call, expr.Call) else call).__name__


@dataclass(frozen=True)
class StorageAnswer:
    partner: core.SSAValue | int | None
    chain: tuple[core.SSAValue, ...]
    aliases: tuple
    offsets: tuple[int, ...]
    relation: access.Relation | None
    refusal: str | None
    kind: analysis.AliasKind | None = None


@dataclass
class StoragePlan:
    """Interpret raw allocation decisions without publishing derived geometry."""

    function: core.Function
    context: valuation.Context | None = None
    mesh: types.Mesh | None = None

    @cached_property
    def regions(self):
        values = (*self.function.params, *core.collect_exprs(self.function.body))
        memo = {id(node): (node, node.type) for node in values}
        ctx = typeinfer.TypeInferContext(
            module=self.context.module, function=self.function, target=self.context.target, topologies=self.context.topologies, owner=self.function, current_mesh=self.mesh, memo=memo
        )
        return valuation.index_contexts(self.function, ctx)

    @cached_property
    def answers(self):
        allocation = expr.get_metadata(self.function, analysis.AllocationSummaryMetadata)
        values = {value.value: value for value in allocation.values}
        links, operations, outputs, refusals = {}, {}, {}, {}
        for leaf, value in values.items():
            ctx = self.regions.get(id(leaf.expr))
            if ctx is None:
                continue
            forwarded = analysis.forwarded_leaves(leaf.expr, ctx)
            position = access.aliased_operand(leaf.expr) if isinstance(leaf.expr, expr.Call) else None
            if forwarded is not None:
                partner = forwarded[leaf.index]
                kind = analysis.AliasKind.FORWARD
            elif position is not None:
                partner = core.SSAValue(leaf.expr.args[position])
                kind = analysis.AliasKind.VIEW
            else:
                alias = next(
                    (
                        alias
                        for alias in value.aliases
                        if alias.target != leaf
                        and alias.target in values
                        and not (
                            value.lifetime.memory_level == "rmem"
                            and alias.kind is analysis.AliasKind.IN_PLACE
                        )
                    ),
                    None,
                )
                partner = None if alias is None else alias.target
                kind = None if alias is None else alias.kind
            if partner is not None and partner != leaf:
                links[leaf] = partner
                operations[leaf] = (kind, leaf.expr)
        for holder, alias, representative in allocation.address_decisions:
            value = values[representative]
            if value.lifetime.memory_level == "rmem" and alias.kind is analysis.AliasKind.IN_PLACE:
                continue
            if holder != representative:
                chain = self.chain(holder, links)
                refusal = (
                    self.identity_refusal(
                        chain[: chain.index(representative) + 1], operations, values
                    )
                    if representative in chain
                    else f"address for {label(values[holder].value.expr)} requires an inverse relation to an unreachable storage representative: {alias.relation.relation}"
                )
                if refusal is not None:
                    refusals[representative] = refusal
            links[representative] = alias.target
            operations[representative] = (
                alias.kind,
                alias.target.expr if alias.kind is analysis.AliasKind.WINDOW else holder.expr,
            )
        seen = set()
        for index, (path, type_) in enumerate(access.leaf_paths(self.function.return_type)):
            if not isinstance(type_, types.TensorType):
                continue
            leaf = core.SSAValue(self.function.body, index)
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
                    isinstance(item.expr, expr.Call)
                    and self.context.callee_context(item.expr) is not None
                )
                writes = any(
                    operations.get(member, (None, None))[0] is analysis.AliasKind.INSERT_SLICE
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
                    returning and not retained and kind in (analysis.AliasKind.VIEW, analysis.AliasKind.FORWARD)
                )
            if (
                not any(root.expr is param for param in self.function.params)
                and ir_utils.leaves_of(root.expr.type)[root.index] == type_
                and all(
                    operations.get(item, (None, None))[0] is not analysis.AliasKind.VIEW
                    or (
                        (
                            alias := analysis.buffer_alias(
                                item.expr,
                                self.regions[id(item.expr)],
                                access.relations_of(item.expr, self.regions[id(item.expr)]),
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
            if id(leaf.expr) not in self.regions or not isinstance(leaf.expr, expr.Call):
                continue
            callee = self.context.callee_context(leaf.expr)
            if callee is None:
                continue
            child = StoragePlan(callee.function, callee)
            returned = child.answers[core.SSAValue(callee.function.body, leaf.index)].chain[-1]
            formal = next(
                (i for i, param in enumerate(callee.function.params) if returned.expr is param),
                None,
            )
            if (
                formal is not None
                and self.chain(leaf, links)[-1]
                != self.chain(core.SSAValue(leaf.expr.args[formal], returned.index), links)[-1]
            ):
                path = tuple(
                    path
                    for path, type_ in access.leaf_paths(leaf.expr.type)
                    if isinstance(type_, types.TensorType)
                )[leaf.index]
                refusals[leaf] = (
                    f"return leaf {path!r} cannot be placed on its output: {label(leaf.expr)}"
                )
        answers = {}
        for leaf, value in values.items():
            if id(leaf.expr) not in self.regions:
                continue
            relation = access.identity(len(ir_utils.leaves_of(leaf.expr.type)[leaf.index].shape))
            kind, origin = operations.get(leaf, (None, leaf.expr))
            if leaf not in outputs and kind is analysis.AliasKind.VIEW:
                ctx = self.regions[id(origin)]
                alias = analysis.buffer_alias(origin, ctx, access.relations_of(origin, ctx))
                if alias is None:
                    refusals[leaf] = (
                        f"non-EXACT address for {label(value.value.expr)}: {value.aliases[0].relation.relation}"
                    )
                else:
                    relation = alias.relation
            elif leaf not in outputs and kind is analysis.AliasKind.WINDOW:
                offsets = origin.args[2]
                shape = ir_utils.leaves_of(leaf.expr.type)[leaf.index].shape
                offsets = (
                    offsets.elements
                    if isinstance(offsets, expr.Tuple)
                    else (offsets, *(0 for _ in shape[1:]))
                )
                coordinates, images, guards, parameters = [], [], [], {}
                for axis, (offset, extent) in enumerate(zip(offsets, shape, strict=True)):
                    term, bound = access.affine_term(offset, f"o{axis}")
                    parameters.update(bound)
                    coordinates.append(f"i{axis}")
                    images.append(f"i{axis} + ({term})")
                    guards.append(f"0 <= i{axis} < {extent}")
                prefix = f"[{', '.join(parameters)}] -> " if parameters else ""
                relation = access.Relation(
                    isl.map(
                        f"{prefix}{{ [{', '.join(coordinates)}] -> [{', '.join(images)}] : {' and '.join(guards)} }}"
                    ),
                    parameters,
                )
            answers[leaf] = StorageAnswer(
                outputs.get(leaf, links.get(leaf)),
                self.chain(leaf, links),
                accepted_aliases(value),
                getattr(expr.get_metadata(leaf.expr, analysis.MemoryMetadata), "offsets", ()),
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

    @cached_property
    def scratch(self):
        """The GMEM scratch this function declares, decided before emission.

        Returns ``(params, roots, calls)``: every scratch parameter in order, the
        parameter of each GMEM allocation root this function owns, and the
        parameters each call forwards to its callee's scratch. A root is an
        allocation leaf with no storage partner that is neither a parameter nor
        an output; each call is handled once, and two calls of one definition
        forward separate parameters. Without memory analysis there is none.
        """
        if expr.get_metadata(self.function, analysis.AllocationSummaryMetadata) is None:
            return (), {}, {}
        names = Names({param.name for param in self.function.params})
        params, roots, calls = [], {}, {}
        for leaf, answer in self.answers.items():
            held = ir_utils.leaves_of(leaf.expr.type)[leaf.index]
            if (
                answer.partner is not None
                or held.storage is not expr.StorageKind.GMEM
                or any(leaf.expr is param for param in self.function.params)
            ):
                continue
            elements = prod(tuple(held.shape))
            roots[leaf] = expr.Var(
                names.fresh("scratch"),
                type=types.TensorType(
                    (elements,), held.dtype, types.Layout((elements,)), expr.StorageKind.GMEM
                ),
            )
            params.append(roots[leaf])
        for node in core.collect_exprs(self.function.body):
            if not (isinstance(node, expr.Call) and isinstance(node.target, core.FunctionRef)):
                continue
            frame = self.regions[id(node)].callee_context(node)
            child = StoragePlan(frame.function, frame, frame.current_mesh).scratch[0]
            calls[id(node)] = tuple(expr.Var(names.fresh(param.name), type=param.type) for param in child)
            params.extend(calls[id(node)])
        return tuple(params), roots, calls

    def aliases_of(self, leaf):
        return self.answers[leaf].aliases

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
            if kind is analysis.AliasKind.FORWARD:
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
            same_shape = tuple(ir_utils.leaves_of(leaf.expr.type)[leaf.index].shape) == tuple(
                ir_utils.leaves_of(partner.expr.type)[partner.index].shape
            )
            if kind is analysis.AliasKind.VIEW:
                ctx = self.regions[id(origin)]
                declared = analysis.buffer_alias(origin, ctx, access.relations_of(origin, ctx))
                if same_shape and declared is not None and declared.relation.relation.is_identity():
                    continue
            elif accepted is not None:
                if same_shape and kind in (
                    analysis.AliasKind.CARRY_INIT,
                    analysis.AliasKind.CARRY_YIELD,
                    analysis.AliasKind.INSERT_SLICE,
                ):
                    continue
                if kind is analysis.AliasKind.IN_PLACE and value.lifetime.memory_level != "rmem":
                    continue
            relation = declared if kind is analysis.AliasKind.VIEW and declared is not None else accepted
            return (
                f"address for {label(values[chain[0]].value.expr)} requires a non-identity inverse relation "
                f"at {label(value.value.expr)}: {None if relation is None else relation.relation.relation}"
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
            for index, (_, type_) in enumerate(access.leaf_paths(region.owner.type)):
                if not isinstance(type_, types.TensorType) or type_.storage is not expr.StorageKind.RMEM:
                    continue
                leaf = core.SSAValue(region.owner, index)
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
                    if isinstance(scope.owner, core.LoopRegion)
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

    def runtime_values(self, node, memo):
        """List HIR inputs the caller must lower before tensor calculations."""
        dimensions = {}
        leaves = (
            (node,)
            if isinstance(node, core.SSAValue)
            else tuple(core.SSAValue(node, index) for index in range(len(ir_utils.leaves_of(node.type))))
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
                    if expr.static_dim_value(loop.start) != 0:
                        dimensions[id(loop.start)] = loop.start
        return tuple(dimensions.values())

    def hir_view(self, call, operands, values):
        """Keep the view authored by this HIR operation, including identity views."""
        region = self.regions[id(call)]
        alias = analysis.buffer_alias(call, region, access.relations_of(call, region))
        if alias is None:
            raise LoweringError(self.answers[core.SSAValue(call)].refusal)
        return self.view(operands[access.aliased_operand(call)], alias.relation, call.type, values=values)

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
            if answer.kind is analysis.AliasKind.WINDOW:
                windows.append(current)
            current = answer.partner

    def operand(
        self,
        value,
        desired,
        *,
        starts=None,
        sizes=None,
        stem="tile",
        whole_tensor=False,
    ):
        """Choose an existing operand or the window required at its use."""
        held = value.type
        if not isinstance(held, types.TensorType):
            raise LoweringError(f"a tensor window cannot use {held!r} as its base")
        if starts is not None and covers(starts, sizes, held.shape):
            starts = None
        if starts is None:
            def physical_type(type_):
                if whole_tensor or not isinstance(type_.layout, types.ShardLayout):
                    return type_
                layout = ir_utils.physical_layout(type_.layout)
                if isinstance(layout, types.ShardLayout):
                    layout = ir_utils.participant_layout(layout)
                return replace(type_, layout=layout)

            actual, requested = physical_type(held), physical_type(desired)
            if actual == requested or (
                not whole_tensor
                and not isinstance(desired.layout, types.ShardLayout)
                and held.shape == desired.shape
                and held.dtype == desired.dtype
                and held.storage is desired.storage
            ):
                return value
            starts, sizes = tuple(ir_dim.i64_const(0) for _ in held.shape), tuple(held.shape)
        return View(value, starts, sizes, desired, stem, whole_tensor)

    def materialization(self, view, base, loops):
        """Describe one physical TIR view without adding any statements."""
        desired = view.type
        if isinstance(base, expr.Constant) and isinstance(base.type, types.PointerType):
            return self.view(base.value, None, desired)
        if isinstance(base, expr.Call) and isinstance(base.target, tensor.TupleGetItem):
            return base
        held = base.type
        if len(view.starts) != len(held.shape):
            raise LoweringError(
                f"a rank-{len(held.shape)} buffer is windowed by {len(view.starts)} starts"
            )
        source = base
        if not view.whole_tensor and not covers(view.starts, view.sizes, held.shape):
            source = cut(base, view.starts, view.sizes, loops)
        pointer = expr.Call(memory.PtrOf(), (source,), type=types.PointerType(held.dtype, held.storage))
        return expr.Call(
            memory.TensorView(layout=desired.layout, shape=tuple(desired.shape)), (pointer,), type=desired
        )

    def window(self, target, desired, starts):
        """Derive a window at already lowered coordinates.

        The layout offset is relative to the sliced pointer: reduce the
        displacement modulo the swizzle period; the pointer holds its full offset.
        """
        type_ = cut(target, starts, desired.shape).type
        layout = type_.layout.layout if isinstance(type_.layout, types.ShardLayout) else type_.layout
        if isinstance(layout, types.ComposedLayout) and isinstance(layout.inner, types.Swizzle):
            swizzle = layout.inner
            period = 1 << (swizzle.bits + swizzle.base + swizzle.shift)
            layout = replace(layout, offset=layout.offset % period)
        if isinstance(type_.layout, types.ShardLayout):
            layout = replace(type_.layout, layout=layout)
        return starts, replace(desired, layout=layout, storage=target.type.storage)

    def issue_window(self, base, desired, starts, loops):
        """The operand one issue point takes: *base* cut at *starts*, placed as *desired*."""
        window = base if covers(starts, desired.shape, base.type.shape) else cut(
            base, starts, desired.shape, loops
        )
        if window.type == desired:
            return window
        pointer = expr.Call(memory.PtrOf(), (window,), type=types.PointerType(desired.dtype, desired.storage))
        address = (
            window.type.layout.layout
            if isinstance(window.type.layout, types.ShardLayout)
            else window.type.layout
        )
        placement = desired.layout
        layout = replace(placement, layout=address) if isinstance(placement, types.ShardLayout) else address
        type_ = replace(window.type, layout=layout)
        return expr.Call(memory.TensorView(layout=layout, shape=tuple(type_.shape)), (pointer,), type=type_)

    def view(self, target, address, desired, *, values=None):
        """Derive an exact physical view; a proved composition carries its origin."""
        if isinstance(target, int):
            pointer = expr.Constant(target, type=types.PointerType(desired.dtype, desired.storage))
            return expr.Call(
                memory.TensorView(
                    dtype=desired.dtype.name,
                    storage=desired.storage,
                    layout=desired.layout,
                    shape=tuple(desired.shape),
                ),
                (pointer,),
                type=desired,
            )

        relation = address.relation
        if relation.is_identity() and tuple(desired.shape) == tuple(target.type.shape):
            if not isinstance(desired.layout, types.ShardLayout):
                layout = target.type.layout
                desired = replace(
                    desired, layout=layout.layout if isinstance(layout, types.ShardLayout) else layout
                )
            zeros = tuple(ir_dim.i64_const(0) for _ in target.type.shape)
            return View(
                target, zeros, tuple(target.type.shape), desired, "value", whole_tensor=True
            )
        layout = (
            target.type.layout.layout
            if isinstance(target.type.layout, types.ShardLayout)
            else target.type.layout
        )
        placed = relation.apply_range(addresses(tuple(target.type.shape), layout))
        try:
            affine = utils.as_multi_aff(placed).get_at(0)
            if isinstance(layout, types.ComposedLayout) and any(
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
            if not isinstance(layout, types.ComposedLayout):
                raise LoweringError(f"non-affine address {relation}: {error}") from error
            strides = None
        parameters = {name: values[id(value)] for name, value in address.values.items()}
        origin = relation
        for axis in range(relation.dim(isl.dim_type.IN)):
            origin = origin.fix_val(isl.dim_type.IN, axis, isl.val(0))
        point = origin.range()
        starts = tuple(
            ir_dim.isl_to_dimension(point.dim_min(axis).gist_params(point.params()), parameters)
            for axis in range(point.dim(isl.dim_type.SET))
        )
        starts = tuple(ir_dim.i64_const(start) if isinstance(start, int) else start for start in starts)
        whole = all(expr.static_dim_value(start) == 0 for start in starts)
        if isinstance(layout, types.ComposedLayout):
            candidate = desired.layout
            if isinstance(candidate, types.ComposedLayout) or whole:
                if not isinstance(candidate, types.ComposedLayout):
                    candidate = replace(layout, outer=types.Layout(tuple(desired.shape), strides))
                if strides is not None and not affine.dim(isl.dim_type.PARAM):
                    candidate = replace(
                        layout,
                        offset=int(affine.get_constant_val().num_si()),
                        outer=types.Layout(tuple(desired.shape), strides),
                    )
                elif strides is None:
                    linear = relation.apply_range(
                        addresses(
                            tuple(target.type.shape),
                            types.Layout(
                                tuple(target.type.shape),
                                types.compact_row_major(tuple(target.type.shape)),
                            ),
                        )
                    )
                    linear = utils.as_multi_aff(linear).get_at(0)
                    steps = tuple(
                        int(linear.get_coefficient_val(isl.dim_type.IN, axis).num_si())
                        for axis in range(len(desired.shape))
                    )
                    offset = (
                        utils.as_multi_aff(
                            origin.apply_range(addresses(tuple(target.type.shape), layout))
                        )
                        .get_at(0)
                        .get_constant_val()
                        .num_si()
                    )
                    candidate = replace(
                        layout,
                        outer=layout_algebra.composition(
                            layout.outer, types.Layout(tuple(desired.shape), steps), major="row"
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
                starts = tuple(ir_dim.i64_const(0) for _ in starts)
            else:
                desired = self.window(target, desired, starts)[1]
        else:
            offset = isl.map.from_domain_and_range(
                relation.domain(),
                origin.apply_range(addresses(tuple(target.type.shape), layout)).range(),
            )
            declared = addresses(tuple(desired.shape), types.Layout(tuple(desired.shape), strides))
            declared = declared.flat_range_product(offset).apply_range(
                isl.map("{ [a, b] -> [a + b] }")
            )
            if not declared.is_equal(placed):
                raise LoweringError(f"non-affine address {relation}")
            if not isinstance(desired.layout, types.ShardLayout):
                desired = replace(desired, layout=types.Layout(tuple(desired.shape), strides))
        sizes = [1] * len(starts)
        shard = types.shard_layout_of(target.type.layout)
        while shard is not None:
            for axis in types.split_target_axes(shard, tuple(target.type.shape)):
                if axis is None:
                    continue
                if expr.static_dim_value(starts[axis]) != 0:
                    raise LoweringError(f"nonzero pointer origin on split axis {axis}: {relation}")
                sizes[axis] = target.type.shape[axis]
            shard = types.shard_layout_of(shard.layout)
        return View(target, starts, tuple(sizes), desired, "value", whole_tensor=whole)


@dataclass
class IssuePlan:
    """Instruction selection and one shared geometric projection, without admission."""

    def op(self, call, *, target):
        op = selected_op(call, target)
        if op is not None:
            return op
        candidates = schedule.candidate_ops(type(call.target), target=target, call=call)
        if not candidates:
            raise LoweringError(
                f"{label(call)} uses unknown HIR call {type(call.target).__name__}"
            )
        available = ", ".join(expr.get_registered_name(candidate) for candidate in candidates)
        raise LoweringError(
            f"{label(call)} has instruction candidates [{available}] but no automatic "
            "selection; write tf.schedule to choose the instruction and its attributes"
        )

    def plan(self, call, op, mesh):
        type_list = operand_types(call, op)
        if isinstance(call.target, hir.schedule.ScheduleOp):
            repeat, order = call.target.repeat, call.target.order
        else:
            repeat = order = None
        try:
            projection = schedule.project(op, type_list, mesh, repeat=repeat, order=order)
            return projection, type_list, issue_owner(type_list)
        except (schedule.RefusalError, ValueError) as error:
            raise LoweringError(str(error)) from error

    def points(self, plan):
        for ordered in product(*(range(plan.serial[axis]) for axis in plan.order)):
            point = dict(zip(plan.order, ordered, strict=True))
            windows = []
            for logical, masks, axes in zip(
                plan.logical_rests, plan.bind_masks, plan.axes, strict=True
            ):
                starts = []
                for index, (axis, mask) in enumerate(zip(axes, masks, strict=True)):
                    mode = types.get(logical, index)
                    selected = tuple(
                        (width, stride)
                        for width, stride, bound in zip(
                            types.flatten(mode.shape),
                            types.flatten(mode.strides),
                            types.flatten(mask),
                            strict=True,
                        )
                        if not bound
                    )
                    shape, strides = zip(*selected) if selected else ((), ())
                    starts.append(ir_dim.i64_const(types.crd2idx(point.get(axis, 0), shape, strides)))
                windows.append(tuple(starts))
            yield tuple(windows)

@dataclass(kw_only=True)
class LoweringContext(typeinfer.TypeInferContext):
    """Carry storage answers and shared emission state through lexical visits."""

    storage: StoragePlan
    issues: IssuePlan
    builder: TirBuilder
    names: Names
    outputs: list[tuple[tuple[int, ...], expr.Var]] = field(default_factory=list)
    allocations: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)

    def bind_value(self, param, value) -> None:
        """Bind the lowered *value* to the HIR *param* in this context's memo."""
        self.memo[id(param)] = (param, value)

    def value(self, leaf):
        result = self.memo[id(leaf.expr)][1]
        pending = [result]
        index = 0
        while pending:
            value = pending.pop()
            if isinstance(value, expr.Tuple):
                pending.extend(reversed(value.elements))
            elif index == leaf.index:
                return value
            else:
                index += 1
        raise LoweringError(f"missing lowered leaf {leaf.index} of {label(leaf.expr)}")

    def result(self, node):
        """Select result storage using completed Visit results and root declarations."""
        values = {key: value for key, (_, value) in self.memo.items()}
        values.update(
            (id(value), self.dim(value, self.memo))
            for value in self.storage.runtime_values(node, self.memo)
        )
        fields = []
        for index, desired in enumerate(ir_utils.leaves_of(node.type)):
            leaf = core.SSAValue(node, index)
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
            if desired.storage is expr.StorageKind.GMEM and desired.dtype != result.type.dtype:
                raise LoweringError("gmem cross-dtype reuse has no offset-view representation")
            if desired.storage is expr.StorageKind.SMEM and desired.dtype != result.type.dtype:
                root = self.storage.answers[leaf].chain[-1]
                result = self.builder.declare(root, ctx=self, values=values, desired=desired)
            for window in windows:
                result = self.storage.view(
                    result,
                    self.storage.answers[window].relation,
                    ir_utils.leaves_of(window.expr.type)[window.index],
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
                expr.Tuple(tuple(pack(field) for field in type_.fields), type=type_)
                if isinstance(type_, types.TupleType)
                else next(fields)
            )

        return pack(node.type)

    def dim(self, value, values) -> expr.Expr:
        number = expr.static_dim_value(value)
        if number is not None:
            return ir_dim.i64_const(number)
        if isinstance(value, expr.Var):
            known = values.get(id(value))
            if known is None:
                raise LoweringError(f"dimension uses unbound value {value.name!r}")
            return self.builder.value(known[1], ctx=self)
        if isinstance(value, expr.Call) and isinstance(value.target, ir_dim.MeshCoord):
            return expr.Call(
                ir_dim.MeshCoord(mesh=value.target.mesh),
                value.args,
                type=value.type,
            )
        if isinstance(value, expr.Call) and isinstance(value.target, math.Binary):
            dim_kind = DIM_BINARY.get(value.target.kind)
            if dim_kind is None or not isinstance(value.type, types.TensorType) or value.type.shape != ():
                raise LoweringError(
                    f"dimension uses unsupported binary operation {value.target.kind}"
                )
            return services.simplify_dim(dim_kind, tuple(self.dim(arg, values) for arg in value.args))
        if isinstance(value, expr.Call) and expr.is_dim_op_call(value):
            return expr.Call(
                value.target, tuple(self.dim(arg, values) for arg in value.args), type=value.type
            )
        if isinstance(value, expr.Call):
            return self.builder.value(values[id(value)][1], ctx=self)
        if isinstance(value, expr.Expr):
            return values.get(id(value), (value, value))[1]
        raise LoweringError(f"dimension {value!r} is not lowerable")


@dataclass
class Lowering(valuation.Visitor[expr.Expr | View]):
    """Mechanical lowering of one analyzed HIR function.

    ``visit_Call`` dispatches a Call to ``visit_<HIR op>``, else to ``visit_T``,
    which dispatches the selected TIR op to ``visit_T_<Name>``, else to
    ``default_visit_T``. A target lowering overrides the visits it changes. A node
    without a visit goes to ``default_visit``, which refuses it: the visitor keeps
    ``ExprVisitor.default_visit_leaf``, so it never walks such a node's operands.
    """

    context: typeinfer.TypeInferContext
    callees: dict[int, Lowering] = field(default_factory=dict)
    lowered: core.PrimFunction | None = field(default=None, init=False)
    default_visit_leaf = core.ExprVisitor.default_visit_leaf

    def visit(self, node, ctx=None, *, dimension=False):
        if ctx is None:
            ctx = self.locations.get(id(node), self.context)
            if isinstance(node, (core.MeshRegion, core.LoopRegion)):
                ctx = ctx.parent
        if dimension:
            if isinstance(node, expr.Expr):
                for value in core.collect_exprs(node):
                    if isinstance(value, expr.Call) and not lowers_to_dim(value):
                        self.visit(value, ctx=ctx)
            return ctx.dim(node, self.memo)
        known = ctx.memo.get(id(node))
        if known is not None:
            return known[1]
        if isinstance(node, expr.Constant):
            return node
        if isinstance(node, expr.Var) and id(node) not in self.memo:
            raise LoweringError(f"unbound HIR value {node.name!r} reached lowering")
        if isinstance(node, expr.Tuple):
            elements = tuple(self.visit(element, ctx=ctx) for element in node.elements)
            result = expr.Tuple(elements, type=types.TupleType(tuple(element.type for element in elements)))
        else:
            if isinstance(node, (core.LoopRegion, core.MeshRegion)) and id(node) not in self.memo:
                values = tuple(self.visit(arg, ctx=ctx) for arg in node.args)
                for index, (param, value) in enumerate(zip(node.params, values, strict=True)):
                    if not isinstance(node, core.LoopRegion) or index >= len(node.yield_values):
                        ctx.bind_value(param, value)
            result = super().visit(node, ctx)
        self.memo[id(node)] = (node, result)
        return result

    def __post_init__(self) -> None:
        super().__init__()
        physical = {
            id(param): expr.Var(param.name, type=param.type, is_const=param.is_const)
            for param in self.context.function.params
        }
        storage = StoragePlan(self.context.function, self.context, self.context.current_mesh)
        self.context = LoweringContext(
            **{**vars(self.context), "memo": self.memo},
            parameters=physical,
            storage=storage,
            builder=TirBuilder([TirScope(region=self.context.function)]),
            names=Names(
                {param.name for param in (*self.context.function.params, *storage.scratch[0])}
            ),
            issues=IssuePlan(),
        )
        ctx = self.context
        self.locations = valuation.index_contexts(ctx.function, self.context)
        for param in ctx.function.params:
            self.memo[id(param)] = (param, physical[id(param)])

    def visit_ordered(self, owner, ctx) -> None:
        """Lower the children of *owner*'s body first in the order liveness gave them events."""
        liveness = expr.get_metadata(self.context.function, analysis.LivenessMetadata)
        for node in () if liveness is None else liveness.orders.get(id(owner), ()):
            self.visit(node, ctx)

    def on_region_start(self, region, ctx):
        inner = super().on_region_start(region, ctx)
        self.visit_ordered(region, inner)
        return inner

    def run(self) -> core.PrimFunction:
        ctx = self.context
        if ctx.function.body is None:
            raise LoweringError(f"{ctx.function.name!r} has no body to lower")
        returns = tuple(access.leaf_paths(ctx.function.return_type))
        for index, (path, type_) in enumerate(returns):
            if not isinstance(type_, types.TensorType):
                raise LoweringError(f"return leaf {path!r} has non-tensor type {type_!r}")
            leaf = core.SSAValue(ctx.function.body, index)
            if leaf not in ctx.storage.answers:
                raise LoweringError(f"return leaf {path!r} cannot be placed on its output: None")
            if (refusal := ctx.storage.answers[leaf].refusal) is not None:
                raise LoweringError(refusal)
        for path, type_ in returns:
            output_type = (
                replace(type_, storage=expr.StorageKind.GMEM)
                if type_.storage is expr.StorageKind.UMAT
                else type_
            )
            output = expr.Var(ctx.names.fresh("out"), type=output_type)
            ctx.outputs.append((path, output))
        self.visit_ordered(ctx.function, ctx)
        self.visit(ctx.function.body, ctx=ctx)
        statements = ctx.builder.build(ctx.builder.blocks[0])

        params = tuple(ctx.parameters[id(param)] for param in ctx.function.params)
        self.lowered = core.PrimFunction(
            ctx.function.name,
            (*params, *ctx.storage.scratch[0], *(output for _, output in ctx.outputs)),
            statements,
            output_count=len(ctx.outputs),
            target=self.context.target,
        )
        return self.lowered

    def default_visit(self, node, ctx):
        raise LoweringError(f"{type(node).__name__} has no lowering on {tilefoundry_target.summary(ctx.target)}")

    def visit_MeshRegion(self, region: core.MeshRegion, ctx):
        mesh = region.mesh
        binding = expr.Var(
            ctx.names.fresh("cta" if region is ctx.function.body else "scope"), type=BINDING
        )
        with ctx.builder.region(mesh=mesh, binding=binding, owner=region) as scope:
            result = self.visit(region.body, self.on_region_start(region, ctx))
            scope.yield_value(result)
        return result

    def visit_Call(self, call: expr.Call, ctx) -> expr.Expr:
        if lowers_to_dim(call):
            return self.visit(call, ctx=ctx, dimension=True)
        if core.SSAValue(call) in ctx.storage.answers:
            if (refusal := ctx.storage.answers[core.SSAValue(call)].refusal) is not None:
                raise LoweringError(refusal)
            self.runtime_values(call, ctx)
        target = call.target
        if isinstance(target, core.FunctionRef):
            operands = tuple(
                self.visit(arg, ctx=ctx, dimension=True) for arg in target.indices(call.args)
            ) + tuple(self.visit(arg, ctx=ctx) for arg in target.arguments(call.args))
        else:
            operands = tuple(self.visit(arg, ctx=ctx) for arg in call.args)
        if ctx.callee_context(call) is not None:
            return self.visit_callee(call, operands, ctx)
        return getattr(self, f"visit_{type(call.target).__name__}", self.visit_T)(
            call, operands, ctx
        )

    def runtime_values(self, call: expr.Call, ctx) -> dict:
        """Lower the values the dimensions of *call*'s storage read, by HIR value.

        ``visit_Call`` lowers them before the operands; a view asks again for the
        lowered values, which adds no statement.
        """
        if core.SSAValue(call) not in ctx.storage.answers:
            return {}
        return {
            id(value): self.visit(value, ctx=ctx, dimension=True)
            for value in ctx.storage.runtime_values(call, ctx.memo)
        }

    def visit_callee(self, call: expr.Call, operands, ctx) -> expr.Expr | View:
        """Call the retained function *call* names, lowering it once."""
        frame = ctx.callee_context(call)
        lowering = self.callees.get(id(frame.function))
        if lowering is None:
            lowering = type(self)(
                typeinfer.TypeInferContext(module=frame.module, function=frame.function, target=frame.target, topologies=frame.topologies, owner=frame.function, current_mesh=ctx.current_mesh),
                self.callees,
            )
            lowering.run()
            self.callees[id(frame.function)] = lowering
        destination = ctx.result(call)
        scratch = ctx.storage.scratch[2][id(call)]
        target = call.target
        indices = target.indices(operands)
        arguments = target.arguments(operands)
        params = tuple(
            param
            for param in lowering.lowered.params
            if target.name is not None or not param.is_const
        )
        passed = []
        for value, param in zip((*arguments, *scratch, destination), params, strict=True):
            passed.append(
                ctx.builder.value(
                    ctx.storage.operand(value, param.type, whole_tensor=True, stem="argument"),
                    ctx=ctx,
                )
                if isinstance(param.type, types.TensorType)
                else ctx.builder.value(value, ctx=ctx)
            )
        ctx.builder.add(core.Evaluate(target, (*indices, *passed)))
        return destination

    def visit_TupleGetItem(self, call: expr.Call, operands, ctx) -> expr.Expr:
        source, index = operands
        return (
            source.elements[index.value]
            if isinstance(source, expr.Tuple) and isinstance(index, expr.Constant)
            else expr.Call(call.target, (source, index), type=call.type)
        )

    def view_of(self, call: expr.Call, operands, ctx) -> View:
        """The view *call* authors of its aliased operand."""
        return ctx.storage.hir_view(call, operands, self.runtime_values(call, ctx))

    def visit_Slice(self, call: expr.Call, operands, ctx) -> View:
        return self.view_of(call, operands, ctx)

    def visit_Reshape(self, call: expr.Call, operands, ctx) -> View:
        return self.view_of(call, operands, ctx)

    def visit_Bitcast(self, call: expr.Call, operands, ctx) -> View:
        return self.view_of(call, operands, ctx)

    def visit_Reshard(self, call: expr.Call, operands, ctx) -> expr.Expr | View:
        """A view where memory accepted its result's VIEW alias, else a copy."""
        leaf = core.SSAValue(call)
        if leaf in ctx.storage.answers and any(
            alias.kind is analysis.AliasKind.VIEW for alias in ctx.storage.aliases_of(leaf)
        ):
            return self.view_of(call, operands, ctx)
        return self.visit_T(call, operands, ctx)

    def visit_Empty(self, call: expr.Call, operands, ctx) -> expr.Expr | View:
        if call.type.storage is expr.StorageKind.RMEM and call.target.buffers != 1:
            raise LoweringError(f"{label(call)} staged buffer has no owning scope")
        return ctx.result(call)

    def visit_IndexSelect(self, call: expr.Call, operands, ctx) -> expr.Expr:
        raise LoweringError(
            f"{label(call)} is an unscheduled IndexSelect; write "
            "tf.schedule((x, tf.empty(...), idx), op=T.copy_async(fill=...))"
        )

    def visit_T(self, call: expr.Call, operands, ctx) -> expr.Expr | View:
        """Issue the TIR op selected for *call* through ``visit_T_<Name>``."""
        op = ctx.issues.op(call, target=ctx.target)
        return getattr(self, f"visit_T_{type(op).__name__}", self.default_visit_T)(
            call, op, operands, ctx
        )

    def default_visit_T(self, call: expr.Call, op, operands, ctx) -> expr.Expr | View:
        raise LoweringError(
            f"{label(call)} has no lowering for {type(op).__name__} "
            f"on {tilefoundry_target.summary(ctx.target)}"
        )


def scratch_buffers(frame) -> tuple[expr.Var, ...]:
    """The scratch parameters the lowered *frame* function declares, in order.

    A synthesized host entry declares and forwards the same list, so it never
    has to be rewritten after lowering.
    """
    return StoragePlan(frame.function, frame).scratch[0]


def device_root(module):
    function = module.entry_function()
    if isinstance(function, core.PrimFunction) and isinstance(function.target, cpu.Target):
        callees = (
            core.resolve(module, stmt.args[0])[1]
            for stmt in function.body
            if isinstance(stmt, core.Evaluate) and isinstance(stmt.callable, tir.launch.Launch)
        )
        roots = tuple(callee for callee in callees if isinstance(callee, core.Function))
        if len(roots) != 1:
            raise LoweringError(
                f"host {function.name!r} launches {len(roots)} HIR functions; expected exactly one"
            )
        return roots[0]
    return function


@dataclass
class ConvertHIRToTIR(pass_base.ModulePass):
    """Replace one selected HIR entry with its scheduled TIR function."""

    visitor: ClassVar[type[Lowering]] = Lowering
    options: passes_options.Options = passes_options.Options()

    name = "convert-hir-to-tir"
    requires = ()
    analyses = ("memory",)

    @classmethod
    def create(cls, target, options) -> ConvertHIRToTIR:
        """Create the configured pass selected by a Target."""
        return cls(options)

    def run(self, module: core.Module) -> core.Module:
        if not self.options.lower or not any(
            isinstance(frame.function, core.Function) for frame in valuation.collect_functions(module).values()
        ):
            return module
        function = device_root(module)
        if expr.get_metadata(function, analysis.AllocationSummaryMetadata) is None:
            raise LoweringError(
                "finalize needs the result of analyze(..., analysis=('memory',)) on a checked program"
            )
        lowered = {}
        lowering = self.visitor(
            typeinfer.TypeInferContext(
                module=module, function=function, owner=function
            ),
            lowered,
        )
        lowering.run()
        lowered[id(function)] = lowering
        modules = set()

        def lower_module(node):
            if id(node) in modules:
                return
            modules.add(id(node))
            for slot in node.children.values():
                for child in core.instances(slot):
                    lower_module(child)
            node.replace_functions(
                lowered[id(fn)].lowered if id(fn) in lowered else fn for fn in node.functions
            )

        lower_module(module)
        return module


__all__ = [
    "BINDING",
    "ConvertHIRToTIR",
    "INDEX",
    "Lowering",
    "LoweringError",
    "StoragePlan",
    "TirBuilder",
    "TirScope",
    "View",
    "accepted_aliases",
    "copy_mesh",
    "issue_owner",
    "label",
    "lowers_to_dim",
    "operand_types",
    "scratch_buffers",
    "selected_op",
    "transfer_meshes",
]
