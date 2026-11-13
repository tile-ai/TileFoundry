"""The ``sync`` family: where lowering orders asynchronous memory effects.

It walks a Function's actions in lowering's order over the shared
``IterationScope`` and plans the commits, waits, fences and barriers each
access needs before it may follow an earlier one, or refuses the Function. See
[analysis §1](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/analysis.md#1-sync).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from itertools import product
from math import prod
from operator import mul
from typing import ClassVar

from tilefoundry import analysis
from tilefoundry import utils as tf_utils
from tilefoundry.interfaces import schedule
from tilefoundry.ir import core, dim, expr, tir, types, utils
from tilefoundry.ir.tir import memory
from tilefoundry.passes import transforms
from tilefoundry_nvidia.cuda.ir.tir import sync

from .completion import Effect, Group, Issued, Mechanism, Proxy, completion_of
from .ctas import axis_numbering, refuse_cta_overlaps
from .emission import (
    copies_carry_backedge,
    copies_carry_entry,
    copies_insert_slice_base,
    copies_insert_slice_update,
    issue_mesh,
    operand_roles,
    required_execution_mesh,
    result_param,
    tensor_map_load,
)
from .threads import refuse_thread_overlaps

DONE = -1
UNCOMMITTED = -2
LOOP_LIMIT = 64


class StepKind(Enum):
    """The statement a ``SyncStep`` is; under one tag, steps run in the members' order."""

    COMMIT = auto()
    WAIT = auto()
    PROXY_FENCE = auto()
    BARRIER = auto()
    FENCE = auto()


@dataclass(frozen=True)
class SyncStep:
    """One statement lowering emits.

    A commit, a wait or a fence is for one completion ``group``; a barrier runs
    over ``mesh``; a proxy fence publishes accesses to ``storage``, None for
    shared memory.
    """

    kind: StepKind
    mesh: types.Mesh | None = None
    group: Group | None = None
    storage: expr.StorageKind | None = None


@dataclass(frozen=True)
class SyncMetadata(expr.IRMetadata):
    """The steps lowering emits around what it lowers for one HIR node.

    ``steps`` pairs a tag with its steps. The tag None is before the node's own
    statements: a Call's instruction, fill or retained call, or entering a
    region. ``"issue"`` is before each issue of a Call's instruction and
    ``"after"`` right after its issues, in the region that issues them. Any
    other tag names a copy lowering adds for the node: ``"base"`` or
    ``"update"`` of an InsertSlice, ``("entry", slot)`` or ``("backedge", slot)``
    of a loop.
    """

    steps: tuple[tuple[object, tuple[SyncStep, ...]], ...] = ()

    def at(self, tag=None) -> tuple[SyncStep, ...]:
        """The steps lowering emits at the statements *tag* names."""
        return dict(self.steps).get(tag, ())


@dataclass(frozen=True)
class SyncPlanMetadata(expr.IRMetadata):
    """Marks a Function the ``sync`` family planned; the steps sit on its nodes.

    ``shared`` is whether any of its actions touches shared memory.
    ``barriers`` is the offset and count of the per-warp mbarrier words that its
    tensor-map loads, or a callee's, complete on; None when there are none.
    ``global_async`` is whether any of its actions reaches global memory through
    the async proxy. ``errors`` are non-fatal: the words ending above capacity.
    """

    family: ClassVar[str] = "sync"

    shared: bool = False
    barriers: tuple[int, int] | None = None
    global_async: bool = False
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class Use:
    """One effect of an action on shared memory bytes, global memory roots or one register storage.

    ``threads`` is the layout that gives each of those bytes to one issuing
    thread, when there is one. A root of None in ``global_roots`` is unknown.
    """

    effect: Effect
    spans: tuple = ()
    register: core.SSAValue | None = None
    threads: types.ShardLayout | None = None
    global_roots: tuple = ()


@dataclass(frozen=True)
class Action:
    """One instruction or retained call lowering emits for an HIR node.

    ``position`` orders it among its scope's children; ``owner`` and ``tag``
    name it the way lowering reaches it.
    """

    position: tuple[int, int]
    owner: expr.Expr
    tag: object
    uses: tuple[Use, ...]
    name: str
    mesh: types.Mesh | None


@dataclass(frozen=True)
class Access:
    """One access a later one may have to follow.

    ``ordered`` holds ``(mesh chain, published)`` for each barrier that ordered it;
    it orders the access only for threads inside that scope, and ``published``
    says a proxy fence ran before the barrier. ``fenced`` says the issuing threads
    ran the proxy fence after it, and ``group_fenced`` that its group's fence ran
    since this register access. ``issuers`` and ``unit`` are the issuing threads and
    the collective unit whose steps they run together, if any.
    """

    position: tuple[int, int]
    write: bool
    spans: tuple
    register: core.SSAValue | None
    meshes: tuple[int, ...]
    state: int
    group: Group | None = None
    proxy: Proxy = Proxy.GENERIC
    chain: tuple[int, expr.DType] | None = None
    issuers: types.Mesh | None = None
    unit: types.Mesh | None = None
    threads: types.ShardLayout | None = None
    ordered: frozenset = frozenset()
    fenced: bool = False
    group_fenced: bool = False
    global_roots: tuple = ()


class UnplacedError(Exception):
    """A step that an asynchronous action needs has no place; it completes in place instead."""

    def __init__(self, position: tuple[int, int], reason: str):
        super().__init__(reason)
        self.position = position


def event_of(liveness: analysis.LivenessMetadata, node) -> int | None:
    """The liveness event at which lowering emits *expr*'s actions, if it has one."""
    lifetime = liveness.interval_of(core.SSAValue(node, 0))
    return None if lifetime is None else lifetime.defined_at


def region_position(liveness: analysis.LivenessMetadata, region) -> tuple[int, int]:
    """Where an inner region sits among the children of its scope."""
    if isinstance(region, core.LoopRegion):
        return (liveness.loop_windows[id(region)][0], -1)
    return (next(item.entered_at for item in liveness.regions if item.region is region), -1)


def carry_positions(liveness: analysis.LivenessMetadata, loop: core.LoopRegion, slot: int):
    """Where a loop slot's entry copy, just before the loop, and backedge copy sit."""
    start, backedge = liveness.loop_windows[id(loop)]
    return (start, slot - len(loop.yield_values) - 1), (backedge, slot)


def overlap(a, b) -> bool:
    return a is None or b is None or (a[0] < b[1] and b[0] < a[1])


def threads_of(op, type_, owners) -> types.ShardLayout | None:
    """The layout giving each byte of a shared memory operand of *type_* to one issuing thread.

    Each thread issues *op* on its own shard when *op* declares no execution
    mesh. A layout that splits along every mesh axis whose units share shared
    memory, as *owners* states, gives each shard's bytes to that thread alone.
    """
    if required_execution_mesh(op) is not None or not isinstance(type_, types.TensorType):
        return None
    layout = type_.layout
    if type_.storage is not expr.StorageKind.SMEM or not isinstance(layout, types.ShardLayout):
        return None
    axes = zip(layout.attrs, types.axis_keys(layout.mesh), strict=True)
    shared = (level for attr, (level, _) in axes if not isinstance(attr, types.Split))
    return layout if all(owners.private(type_.storage, level) for level in shared) else None


def mesh_chain(scope) -> tuple[int, ...]:
    return tuple(id(s.owner) for s in reversed(list(scope.ancestors())) if isinstance(s.owner, core.MeshRegion))


def alone(mesh: types.Mesh | None) -> bool:
    """Whether *mesh* holds one thread, whose own program order orders its accesses."""
    if mesh is None or any(topology.name != "thread" for topology in mesh.topologies):
        return False
    try:
        return tir.sync.participation(mesh).count == 1
    except expr.VerifyError:
        return False


def region_mesh(scope) -> types.Mesh | None:
    """The mesh of the innermost ``MeshRegion`` around *scope*."""
    return next(
        (node.owner.mesh for node in scope.ancestors() if isinstance(node.owner, core.MeshRegion)), None
    )


def one_coordinate(mesh: types.Mesh, axis: int, size: int, bases: range) -> bool:
    """Whether each group of *size* threads from *bases* reads one coordinate on *mesh*'s *axis*.

    Every thread of a group must be one of *mesh*'s. A thread reads
    ``((thread - start) // stride) % extent`` on an axis of its level, as the
    runtime's ``mesh_coords`` does.
    """
    numbering = axis_numbering(mesh, axis)
    if numbering is None:
        return False
    start, extents, strides, index = numbering
    extent, stride = extents[index], strides[index]
    if extent > 1 and stride <= 0:
        return False
    held = {start + sum(map(mul, point, strides)) for point in product(*map(range, extents))}
    for base in bases:
        threads = range(base, base + size)
        if not set(threads) <= held:
            return False
        if extent > 1 and len({(thread - start) // stride % extent for thread in threads}) > 1:
            return False
    return True


class Planner:
    """One walk of a function in lowering's order, repeated for loops until it settles.

    The actions at the positions in *in_place* complete right after they issue.
    """

    def __init__(self, function: core.Function, context: analysis.AnalyzeContext, in_place: frozenset):
        self.function = function
        self.context = context
        self.in_place = in_place
        self.liveness = expr.get_metadata(function, analysis.LivenessMetadata)
        self.summary = expr.get_metadata(function, analysis.AllocationSummaryMetadata)
        if self.liveness is None or self.summary is None:
            raise analysis.Error(f"sync: {function.name!r} has no memory analysis")
        self.values = {value.value: value for value in self.summary.values}
        self.returned = set(self.roots(core.ssa_leaves(function.body)))
        self.root = context.scope
        self.locations = self.root.snapshot.definition.locations
        self.nodes = {}
        pending = [self.root]
        while pending:
            node = pending.pop()
            self.nodes[id(node.owner)] = node
            pending.extend(node.children)
        self.by_id = {id(node): node for node in self.nodes.values()}
        self.kids: dict[int, list] = {id(node): [] for node in self.nodes.values()}
        self.actions: dict[tuple[int, int], Action] = {}
        self.names: dict[tuple[int, int], tuple[expr.Expr, object]] = {}
        self.paths: dict[tuple[int, int], tuple] = {}
        self.planned: dict[tuple[int, int], list[SyncStep]] = {}
        self.issues: dict[tuple[int, int], list[SyncStep]] = {}
        self.after: dict[tuple[int, int], list[SyncStep]] = {}
        self.entered: dict[tuple[int, int], int] = {}
        self.seq = 0
        self.loads = False
        self.build()

    def aliases_of(self, leaf):
        value = self.values.get(leaf)
        return () if value is None else transforms.accepted_aliases(value)

    def build(self) -> None:
        for node2 in core.collect_exprs(self.function.body):
            if isinstance(node2, (core.MeshRegion, core.LoopRegion)) and id(node2) in self.nodes:
                node = self.nodes[id(node2)]
                if not any(child is node for _, child in self.kids[id(node.parent)]):
                    position = region_position(self.liveness, node2)
                    self.names[position] = (node2, None)
                    self.kids[id(node.parent)].append((position, node))
                if isinstance(node2, core.LoopRegion):
                    self.loop_actions(node2, node)
            elif isinstance(node2, expr.Call):
                event = event_of(self.liveness, node2)
                if event is not None:
                    scope = self.locations[id(node2)]
                    for action in self.call_actions(node2, event, scope):
                        self.add(scope, action)
        for children in self.kids.values():
            children.sort(key=lambda item: item[0])

    def add(self, scope, action: Action) -> None:
        self.actions[action.position] = action
        self.names[action.position] = (action.owner, action.tag)
        self.kids[id(scope)].append((action.position, action))

    def loop_actions(self, loop: core.LoopRegion, inner) -> None:
        outer = inner.parent
        for slot in range(len(loop.yield_values)):
            entry, backedge = carry_positions(self.liveness, loop, slot)
            target = loop.params[slot]
            if copies_carry_entry(loop, slot, self.aliases_of):
                self.add(outer, self.copy(
                    entry, loop, ("entry", slot), loop.args[slot], target, outer, "carry entry copy"
                ))
            if copies_carry_backedge(loop, slot, self.aliases_of):
                source = loop.yield_values[slot]
                self.add(inner, self.copy(
                    backedge, loop, ("backedge", slot), source, target, inner, "carry backedge copy"
                ))

    def copy(self, position, owner, tag, source, target, scope, name: str) -> Action:
        """A ``Copy`` lowering adds for *owner* from *source* into *target*'s storage."""
        type_list = (source.type, target.type)
        mesh = self.transferred(type_list, target.type, scope)
        return self.action(
            position, owner, tag, memory.Copy(), (source, target), type_list, name, scope, mesh
        )

    def transferred(self, type_list, destination, scope) -> types.Mesh | None:
        """The mesh of the innermost region around a copy or fill that lowering adds in *scope*."""
        around = [scope.mesh, *(node.owner.mesh for node in scope.ancestors() if isinstance(node.owner, core.MeshRegion))]
        meshes = transforms.transfer_meshes(type_list, destination, around, self.context.topologies)
        return meshes[-1] if meshes else None

    def action(
        self, position, owner, tag, op, args, type_list, name: str, scope, mesh, *, filled=(), result=()
    ) -> Action:
        """*op* issued for *owner* on *mesh* with *args* of *types* in its parameters' order.

        A parameter named in *filled* is issued with the call's *result* storage,
        and also with its argument when one is given.
        """
        if mesh is not None and (scope.mesh is None or not types.within_scope(mesh, scope.mesh)):
            raise analysis.Error(f"sync: {name} is issued on a mesh not within its scope's")
        params = schedule.call_params(op, len(args))
        leaves = {
            param.name: (() if arg is None else core.ssa_leaves(arg))
            + (tuple(result) if param.name in filled else ())
            for param, arg in zip(params, args, strict=True)
        }
        owners = self.context.target.storage_owners
        threads = {
            param.name: threads_of(op, type_, owners)
            for param, type_ in zip(params, type_list, strict=True)
        }
        effects = completion_of(Issued(op, params, tuple(type_list)))
        uses = self.uses(effects, leaves, threads)
        return Action(position, owner, tag, uses, name, mesh)

    def uses(self, effects, leaves, threads=None) -> tuple[Use, ...]:
        """Each effect on the shared memory bytes, the global memory roots and each register."""
        threads = threads or {}
        found = []
        for effect in effects:
            touched = leaves[effect.param]
            spans, roots = self.spans(touched), self.global_roots(touched)
            if spans:
                found.append(Use(effect, spans, threads=threads.get(effect.param)))
            if roots:
                found.append(Use(effect, global_roots=roots))
            found.extend(Use(effect, register=register) for register in self.registers(touched))
        return tuple(found)

    def global_roots(self, leaves) -> tuple:
        """The global memory allocation roots these values live in; None stands for unknown.

        A Function other than the entry may return a value in storage its caller
        also passes it. A global value the memory family did not record has an
        unknown root.
        """
        entry = self.context.entry is None or self.function is self.context.entry
        roots = []
        for leaf in leaves:
            if leaf not in self.values:
                leaf_types = utils.leaves_of(leaf.expr.type)
                stored = leaf_types[leaf.index] if leaf.index < len(leaf_types) else None
                if getattr(stored, "storage", None) is expr.StorageKind.GMEM:
                    roots.append(None)
                continue
            root = self.summary.root(leaf)
            if root.lifetime.memory_level == "gmem":
                roots.append(None if root.value in self.returned and not entry else root.value)
        return tuple(dict.fromkeys(roots))

    def registers(self, leaves) -> tuple[core.SSAValue, ...]:
        """The register storages lowering gives these values."""
        return tuple(
            dict.fromkeys(
                self.register_storage(leaf)
                for leaf in leaves
                if leaf in self.values and self.values[leaf].lifetime.memory_level == "rmem"
            )
        )

    def register_storage(self, leaf: core.SSAValue) -> core.SSAValue:
        """The value whose register storage lowering gives *leaf*.

        An instruction written in place returns its argument's storage; any other
        value follows its first accepted alias.
        """
        seen = set()
        while leaf not in seen:
            seen.add(leaf)
            following = self.following_in_place(leaf)
            if following is None:
                following = next(
                    (alias.target for alias in self.aliases_of(leaf) if alias.target != leaf), None
                )
            if following is None:
                return leaf
            leaf = following
        raise analysis.Error(f"sync: the register storage links of {leaf} form a cycle")

    def following_in_place(self, leaf: core.SSAValue) -> core.SSAValue | None:
        """The argument an instruction call writes in place, when *leaf* is its result.

        A Call issues an instruction when its visit is ``visit_T``, or
        ``visit_Reshard`` of a result that is not a view.
        """
        call = leaf.expr
        if not isinstance(call, expr.Call) or leaf.index != 0 or transforms.lowers_to_dim(call):
            return None
        visit = self.visit(call)
        if visit != self.visit_T and (visit != self.visit_Reshard or self.viewed(call)):
            return None
        op = transforms.selected_op(call, self.context.target)
        if op is None:
            return None
        params = schedule.call_params(op, len(transforms.operand_types(call, op)))
        result = result_param(params)
        supplied, _ = operand_roles(call, params)
        if not result.effect & expr.MemoryEffect.READ or result not in supplied:
            return None
        return core.SSAValue(call.args[supplied.index(result)], 0)

    def call_actions(self, call: expr.Call, event: int, scope) -> list[Action]:
        """Each instruction or retained call lowering emits for *call* at *event*, in order.

        A Call is dispatched as ``CudaLowering`` dispatches it.
        """
        if transforms.lowers_to_dim(call):
            return []
        callee = scope.attachment(call)
        if callee is not None:
            return self.call(call, event, callee.owner)
        return self.visit(call)(call, event, scope)

    def visit(self, call: expr.Call):
        """The visit of *call*: ``visit_<HIR op>``, else ``visit_T``."""
        return getattr(self, f"visit_{type(call.target).__name__}", self.visit_T)

    def viewed(self, call: expr.Call) -> bool:
        """Whether memory accepted a VIEW alias on *call*'s result."""
        return any(alias.kind is analysis.AliasKind.VIEW for alias in self.aliases_of(core.SSAValue(call)))

    def no_action(self) -> list[Action]:
        """Lowering emits no access for a tuple item, a view, an allocation or a refused Call."""
        return []

    def visit_TupleGetItem(self, call: expr.Call, event: int, scope) -> list[Action]:
        return self.no_action()

    def visit_Slice(self, call: expr.Call, event: int, scope) -> list[Action]:
        return self.no_action()

    def visit_Reshape(self, call: expr.Call, event: int, scope) -> list[Action]:
        return self.no_action()

    def visit_Bitcast(self, call: expr.Call, event: int, scope) -> list[Action]:
        return self.no_action()

    def visit_Empty(self, call: expr.Call, event: int, scope) -> list[Action]:
        return self.no_action()

    def visit_IndexSelect(self, call: expr.Call, event: int, scope) -> list[Action]:
        return self.no_action()

    def visit_Reshard(self, call: expr.Call, event: int, scope) -> list[Action]:
        return [] if self.viewed(call) else self.visit_T(call, event, scope)

    def fill(self, call: expr.Call, event: int, scope) -> list[Action]:
        """The one ``Fill`` lowering emits into *call*'s result."""
        type_list = (call.type, None)
        mesh = self.transferred((call.type,), call.type, scope)
        return [self.action(
            (event, 0), call, None, memory.Fill(), (call, None), type_list, "Fill", scope, mesh,
        )]

    def visit_Zeros(self, call: expr.Call, event: int, scope) -> list[Action]:
        return self.fill(call, event, scope)

    def visit_FullLike(self, call: expr.Call, event: int, scope) -> list[Action]:
        return self.fill(call, event, scope)

    def visit_InsertSlice(self, call: expr.Call, event: int, scope) -> list[Action]:
        actions = []
        for tag, source, copies in (
            ("base", call.args[0], copies_insert_slice_base),
            ("update", call.args[1], copies_insert_slice_update),
        ):
            if copies(call, self.aliases_of):
                position = (event, len(actions))
                name = f"InsertSlice {tag}"
                actions.append(self.copy(position, call, tag, source, call, scope, name))
        return actions

    def visit_T(self, call: expr.Call, event: int, scope) -> list[Action]:
        """The action that issues the TIR op selected for *call*."""
        op = transforms.selected_op(call, self.context.target)
        if op is None:
            return []
        type_list = transforms.operand_types(call, op)
        if tensor_map_load(op, type_list):
            self.loads = True
        params = schedule.call_params(op, len(type_list))
        supplied, produced = operand_roles(call, params)
        given = dict(zip((param.name for param in supplied), call.args, strict=True))
        return [
            self.action(
                (event, 0),
                call,
                None,
                op,
                tuple(given.get(param.name) for param in params),
                type_list,
                type(op).__name__,
                scope,
                issue_mesh(op, transforms.issue_owner(type_list), scope.mesh),
                filled=tuple(param.name for param in produced),
                result=core.ssa_leaves(call),
            )
        ]

    def call(self, call: expr.Call, event: int, callee: core.Function) -> list[Action]:
        """A retained call, which reads its arguments and writes its results.

        Callees are planned before their callers, order their own accesses and
        complete them before returning. Unless the callee's plan touches no
        shared memory, the call also reads and writes unknown shared memory bytes
        and may leave ``cp.async`` groups in flight, so whatever conflicts with it
        waits for all of them. Unless the callee's plan reaches global memory only
        through the generic proxy, the call also reads and writes unknown global
        memory through the async proxy, its own temporaries included.
        """
        leaves = {
            **{f"argument {index}": core.ssa_leaves(arg) for index, arg in enumerate(call.args)},
            "result": core.ssa_leaves(call),
        }
        effects = tuple(Effect(name, name == "result", Mechanism.ON_RETURN) for name in leaves)
        uses = [
            use
            for use in self.uses(effects, leaves)
            if use.register is not None or use.global_roots
        ]
        plan = expr.get_metadata(callee, SyncPlanMetadata)
        if plan is not None and plan.barriers is not None:
            self.loads = True
        if plan is None or plan.shared:
            uses += [
                Use(Effect("callee", write, Mechanism.GROUP, Group.CP_ASYNC), (None,))
                for write in (False, True)
            ]
        if plan is None or plan.global_async:
            uses += [
                Use(
                    Effect("callee", write, Mechanism.ON_RETURN, proxy=Proxy.ASYNC),
                    global_roots=(None,),
                )
                for write in (False, True)
            ]
        if not uses:
            return []
        return [Action((event, 0), call, None, tuple(uses), f"the call to {callee.name!r}", None)]

    def roots(self, leaves) -> tuple:
        """The allocation roots these values live in."""
        return tuple(self.summary.root(leaf).value for leaf in leaves if leaf in self.values)

    def spans(self, leaves) -> tuple:
        """The shared memory byte ranges these values own; None stands for unknown.

        A returned allocation may live in the output its caller passes.
        """
        return tuple(
            None if self.summary.root(leaf).value in self.returned else analysis.owned_bytes(self.summary, leaf)
            for leaf in leaves
            if leaf in self.values and self.summary.root(leaf).lifetime.memory_level == "smem"
        )

    def path(self, scope, position) -> tuple:
        """The (scope, child) pairs from the function body down to *position*."""
        steps = []
        child = position
        while scope is not None:
            steps.append((id(scope), child))
            if scope.parent is None:
                break
            child = region_position(self.liveness, scope.owner)
            scope = scope.parent
        return tuple(reversed(steps))

    def run(self) -> None:
        """Plan the Function and attach each node's steps to it as ``SyncMetadata``.

        A collective group still in flight when the Function returns makes its
        action complete in place.
        """
        state = self.walk(self.root, frozenset())
        for access, _ in sorted(state, key=lambda item: item[1]):
            if access.unit is not None and access.state != DONE:
                name = self.actions[access.position].name
                raise UnplacedError(access.position, f"{name} may still be in flight at the return")
        for node in core.collect_exprs(self.function.body):
            node.detach_metadata(SyncMetadata)
        steps: dict[int, tuple[expr.Expr, list]] = {}
        for position, planned in sorted(self.planned.items()):
            owner, tag = self.names[position]
            steps.setdefault(id(owner), (owner, []))[1].append((tag, tuple(planned)))
        for tag, placed in (("issue", self.issues), ("after", self.after)):
            for position, planned in sorted(placed.items()):
                owner, _ = self.names[position]
                steps.setdefault(id(owner), (owner, []))[1].append((tag, tuple(planned)))
        for owner, tagged in steps.values():
            owner.attach_metadata(SyncMetadata(tuple(tagged)))
        uses = [use for action in self.actions.values() for use in action.uses]
        shared = any(use.spans for use in uses)
        global_async = any(use.global_roots and use.effect.proxy is Proxy.ASYNC for use in uses)
        words = self.barrier_words() if self.loads else None
        errors = () if words is None else self.word_errors(*words)
        self.function.attach_metadata(SyncPlanMetadata(shared, words, global_async, errors))

    def entry_peak(self):
        """The shared memory peak the memory family placed for the entry, if any.

        Only the entry places shared memory, so every Function it reaches shares it.
        """
        entry = self.context.entry or self.function
        region = expr.get_metadata(entry, analysis.RegionMemoryMetadata)
        peaks = () if region is None else region.peaks
        return next((level for level in peaks if level.memory_level == "smem"), None)

    def barrier_words(self) -> tuple[int, int]:
        """The offset and count of one mbarrier word per warp, above the entry's shared peak."""
        peak = self.entry_peak()
        topologies = self.context.topologies or ()
        threads = prod(topology.size for topology in topologies if topology.name == "thread")
        warp = self.context.target.architecture.max_threads_per_warp
        end = 0 if peak is None else peak.peak_bytes
        return -(-end // sync.MBARRIER_BYTES) * sync.MBARRIER_BYTES, -(-threads // warp)

    def word_errors(self, offset: int, count: int) -> tuple[str, ...]:
        """A non-fatal error when the words end above a capacity the placed peak fits in.

        A peak already above capacity is the memory family's error, and an unknown
        capacity bounds nothing; neither adds one.
        """
        peak = self.entry_peak()
        if peak is None or peak.capacity_bytes is None or peak.exceeds_capacity:
            return ()
        end = offset + sync.MBARRIER_BYTES * count
        if end <= peak.capacity_bytes:
            return ()
        return (
            f"the mbarrier words end smem at {tf_utils.format_bytes(end)}, above capacity "
            f"{tf_utils.format_bytes(peak.capacity_bytes)}",
        )

    def walk(self, scope, state):
        for position, child in self.kids[id(scope)]:
            self.seq += 1
            self.entered[position] = self.seq
            self.paths.setdefault(position, self.path(scope, position))
            action = child if isinstance(child, Action) else None
            for step in (*self.planned.get(position, ()), *self.issues.get(position, ())):
                state = self.effect(step, scope, state, self.entered[position], action)
            if action is not None:
                state = self.evaluate(action, scope, state)
            elif isinstance(child.owner, core.LoopRegion):
                state = self.loop(child, state)
            else:
                state = self.walk(child, state)
        return state

    def loop(self, scope, state):
        """Join the head with every body end until it settles; the head is also the exit."""
        start = self.seq
        self.seq += 1
        head = state
        for _ in range(LOOP_LIMIT):
            end = self.walk(scope, head)
            joined = frozenset((access, min(seq, start)) for access, seq in head | end)
            if joined == head:
                return head
            head = joined
        raise analysis.Error("sync: the analysis of a loop did not settle within its limit")

    def evaluate(self, action: Action, scope, state):
        units = {use.effect.unit for use in action.uses if use.effect.unit is not None}
        for unit in units:
            self.collective(scope, unit, action.mesh)
        for use in action.uses:
            state = self.order(action, use, scope, state)
        state = self.group_fence(action, scope, state)
        self.seq += 1
        made = frozenset((self.access(action, use, scope), self.seq) for use in action.uses)
        state = state | made
        if action.position in self.in_place:
            state = self.complete_in_place(action, state)
        return state

    def complete_in_place(self, action: Action, state):
        """Commit and wait for *action*'s groups right after it issues, in its issuing region."""
        steps = self.after.setdefault(action.position, [])
        for group in dict.fromkeys(
            use.effect.group for use in action.uses if use.effect.mechanism is Mechanism.GROUP
        ):
            for kind in (StepKind.COMMIT, StepKind.WAIT):
                if SyncStep(kind, group=group) not in steps:
                    steps.append(SyncStep(kind, group=group))
        return frozenset(
            (replace(access, state=DONE) if access.position == action.position else access, seq)
            for access, seq in state
        )

    def access(self, action: Action, use: Use, scope) -> Access:
        """What *use* leaves for later accesses once *action* is issued in *scope*.

        An effect no protocol states is complete when its statement returns.
        """
        effect = use.effect
        status = UNCOMMITTED if effect.mechanism is Mechanism.GROUP else DONE
        return Access(
            action.position,
            effect.write,
            use.spans,
            use.register,
            mesh_chain(scope),
            status,
            effect.group,
            effect.proxy,
            effect.chain,
            action.mesh or region_mesh(scope),
            effect.unit,
            use.threads,
            global_roots=use.global_roots,
        )

    def order(self, action: Action, use: Use, scope, state):
        """Plan what must run before this use, and apply it to *state*."""
        meshes = mesh_chain(scope)
        while True:
            pending = [
                (access, seq)
                for access, seq in sorted(state, key=lambda item: (item[1], item[0].position))
                if self.conflicts(access, use, meshes)
            ]
            if not pending:
                return state
            access, seq = pending[0]
            meeting, position = self.meeting(access.position, action.position)
            if access.state != DONE:
                if access.unit is not None:
                    try:
                        self.collective(meeting, access.unit, access.issuers)
                    except analysis.Error as error:
                        raise UnplacedError(access.position, str(error)) from None
                group = access.group
                if access.state == UNCOMMITTED:
                    commit = SyncStep(StepKind.COMMIT, group=group)
                    state = self.plan(commit, meeting, position, state)
                state = self.plan(SyncStep(StepKind.WAIT, group=group), meeting, position, state)
            if access.register is None:
                if use.effect.proxy is Proxy.ASYNC and access.proxy is Proxy.GENERIC:
                    storage = expr.StorageKind.GMEM if access.global_roots else None
                    fence = SyncStep(StepKind.PROXY_FENCE, storage=storage)
                    state = self.plan(fence, meeting, position, state)
                mesh = region_mesh(meeting)
                if alone(mesh):
                    barrier = SyncStep(StepKind.BARRIER, mesh)
                    state = self.effect(barrier, meeting, state, self.entered[position])
                else:
                    barrier = SyncStep(StepKind.BARRIER, self.barrier_mesh(meeting))
                    state = self.plan(barrier, meeting, position, state)
            same = (access.position, access.write, access.meshes, seq)
            if any(
                (other.position, other.write, other.meshes, later) == same
                and self.conflicts(other, use, meshes)
                for other, later in state
            ):
                raise analysis.Error(
                    f"sync: {action.name} cannot be ordered after "
                    f"{self.actions[access.position].name}"
                )

    def conflicts(self, access, use: Use, meshes) -> bool:
        """Whether *use* must follow *access*.

        A register still in flight in a group takes no other access before the
        group's wait (PTX wgmma.wait_group); later effects of the same group only
        conflict when one writes and their chains differ. A completed access
        whose bytes the same threads touched, by the same layout, is behind *use*
        in each thread's own program order. Global memory roots only conflict
        when one of the two accesses goes through the async proxy.
        """
        if use.register is not None or access.register is not None:
            if use.register != access.register or access.state == DONE:
                return False
            if access.group is None or use.effect.group != access.group:
                return True
            chained = use.effect.chain is not None and use.effect.chain == access.chain
            return (use.effect.write or access.write) and not chained
        if not (use.effect.write or access.write):
            return False
        if use.global_roots or access.global_roots:
            if Proxy.ASYNC not in (use.effect.proxy, access.proxy) or not any(
                a is None or b is None or a == b
                for a in use.global_roots
                for b in access.global_roots
            ):
                return False
        elif not any(overlap(a, b) for a in use.spans for b in access.spans):
            return False
        if use.threads is not None and use.threads == access.threads and access.state == DONE:
            return False
        visible = use.effect.proxy is Proxy.GENERIC or access.proxy is not Proxy.GENERIC
        return not any(
            meshes[: len(tag)] == tag and (visible or published)
            for tag, published in access.ordered
        )

    def group_fence(self, action: Action, scope, state):
        """Plan the group's fence before an action whose registers were touched since the last.

        An action reading registers no chain links to takes the fence before each issue.
        """
        fenced = [use for use in action.uses if use.register is not None and use.effect.fence]
        if not fenced:
            return state
        every = any(use.effect.chain is None for use in fenced)

        def needed(use) -> bool:
            seen = [access for access, _ in state if access.register == use.register]
            return not seen or any(
                not access.group_fenced
                and (access.group is not use.effect.group or access.chain != use.effect.chain)
                for access in seen
            )

        if not every and not any(needed(use) for use in fenced):
            return state
        step = SyncStep(StepKind.FENCE, group=fenced[0].effect.group)
        placed = self.issues if every else self.planned
        steps = placed.setdefault(action.position, [])
        if step not in steps:
            steps.append(step)
        return self.effect(step, scope, state, self.entered[action.position], action)

    def meeting(self, earlier, later):
        """The scope both positions sit in, and the child of it that holds *later*."""
        first, second = self.paths[earlier], self.paths[later]
        scope_id, child = second[-1]
        for a, b in zip(first, second, strict=False):
            if a[0] != b[0]:
                break
            scope_id, child = b
            if a[1] != b[1]:
                break
        return self.by_id[scope_id], child

    def plan(self, step: SyncStep, scope, position, state):
        planned = self.planned.setdefault(position, [])
        if step not in planned:
            planned.append(step)
            planned.sort(
                key=lambda item: (
                    item.kind.value,
                    0 if item.group is None else item.group.value,
                    0 if item.storage is None else item.storage.value,
                )
            )
        return self.effect(step, scope, state, self.entered[position])

    def barrier_mesh(self, scope) -> types.Mesh:
        """The thread mesh a barrier at *scope* runs over, once every thread reaches it.

        It is the mesh of the innermost ``MeshRegion`` around *scope*.
        """
        mesh = region_mesh(scope)
        try:
            barrier = None if mesh is None else tir.sync.classify(mesh)
        except expr.VerifyError as error:
            raise analysis.Error(f"sync: {error}") from None
        if barrier is None or barrier is tir.sync.SyncBarrier.GRID:
            raise analysis.Error("sync: a barrier is needed in a scope that no thread mesh binds")
        self.uniform_loops(scope, mesh, "a barrier")
        return mesh

    def collective(self, scope, unit: types.Mesh, threads: types.Mesh | None) -> None:
        """Require whole *unit*s of threads, *threads* among them, to reach *scope* together."""
        mesh = region_mesh(scope)
        if mesh is None:
            raise analysis.Error(
                "sync: a collective step is needed in a scope that no thread mesh binds"
            )
        try:
            size = tir.sync.participation(unit).count
            held, needed = tir.sync.participation(mesh), tir.sync.participation(threads or mesh)
            whole = sync.whole_groups(mesh, size)
        except expr.VerifyError as error:
            raise analysis.Error(f"sync: {error}") from None
        if not whole:
            raise analysis.Error(
                f"sync: a collective step is needed over {held.count} threads from {held.base}, "
                f"which are not whole groups of {size}"
            )
        if needed.base < held.base or needed.base + needed.count > held.base + held.count:
            raise analysis.Error(
                "sync: a collective step is needed outside the threads that issued it"
            )
        self.uniform_loops(scope, mesh, "a collective step", size)

    def uniform_loops(self, scope, mesh: types.Mesh, what: str, unit: int | None = None) -> None:
        """Require every loop around *scope* to have bounds the threads of *mesh* agree on.

        A barrier needs all of them to agree. A collective step of *unit* threads
        only needs the threads of each aligned group of *unit* to agree (PTX
        ``.aligned``), so the trip count may differ from one group to the next.
        """
        spanned = {topology.name for topology in mesh.topologies}
        groups = None
        if unit is not None and len(mesh.topologies) == 1:
            held = tir.sync.participation(mesh)
            groups = range(held.base, held.base + held.count, unit)

        def fixed(coordinates: types.Mesh, axis: int) -> bool:
            if types.axis_keys(coordinates)[axis][0] not in spanned:
                return True
            return groups is not None and one_coordinate(coordinates, axis, unit, groups)

        entry = self.context.entry is None or self.function is self.context.entry
        uniform = {id(param) for param in self.function.params} if entry else set()
        for node in reversed(list(scope.ancestors())):
            loop = node.owner
            if not isinstance(loop, core.LoopRegion):
                continue
            bounds = (loop.start, loop.extent, loop.step)
            if not all(self.uniform(bound, uniform, fixed, node) for bound in bounds):
                raise analysis.Error(
                    f"sync: {what} is needed inside loop {loop.induction_var.name!r}, "
                    "whose bounds may differ between the threads that must reach it"
                )
            uniform.add(id(loop.induction_var))

    def uniform(self, node, uniform: set[int], fixed, scope) -> bool:
        """Whether the threads that must agree see the same value of a loop bound.

        ``fixed(mesh, axis)`` says whether they read one coordinate on that mesh axis.
        """
        if isinstance(node, expr.Constant) or expr.static_dim_value(node) is not None:
            return True
        if isinstance(node, expr.Var):
            return id(node) in uniform or id(scope.capture_root(node)) in uniform
        if isinstance(node, expr.Call) and isinstance(node.target, dim.MeshCoord):
            axes = dim.literal_axes(node.args[0])
            count = len(types.axis_keys(node.target.mesh))
            return axes is not None and all(
                0 <= axis < count and fixed(node.target.mesh, axis) for axis in axes
            )
        if isinstance(node, expr.Call) and expr.is_dim_op_call(node):
            return all(self.uniform(arg, uniform, fixed, scope) for arg in node.args)
        return False

    def effect(self, step: SyncStep, scope, state, limit, action=None):
        meshes = mesh_chain(scope)
        registers = (
            {use.register for use in action.uses if use.register is not None}
            if step.kind is StepKind.FENCE and action is not None
            else set()
        )
        out = set()
        for access, seq in state:
            issued = seq < limit and access.meshes[: len(meshes)] == meshes
            grouped = access.group == step.group
            if issued and step.kind is StepKind.COMMIT and grouped and access.state == UNCOMMITTED:
                access = replace(access, state=0)
            elif issued and step.kind is StepKind.WAIT and grouped and access.state >= 0:
                access = replace(access, state=DONE)
            elif (
                issued
                and step.kind is StepKind.PROXY_FENCE
                and access.state == DONE
                and (step.storage is expr.StorageKind.GMEM) == bool(access.global_roots)
            ):
                access = replace(access, fenced=True)
            elif issued and step.kind is StepKind.BARRIER and access.state == DONE:
                access = replace(access, ordered=access.ordered | {(meshes, access.fenced)})
            elif step.kind is StepKind.FENCE and seq < limit and access.register in registers:
                access = replace(access, group_fenced=True)
            out.add((access, seq))
        return frozenset(out)


def analyze_sync(function: core.Function, context: analysis.AnalyzeContext) -> None:
    """Attach the synchronisation plan lowering emits for *function*.

    An asynchronous action one of whose steps has no place completes right after
    it issues instead, and the Function is planned again. The entry is refused
    when two of its CTAs may reach one global element unordered, and a Function
    when two threads may write one shared element in one issue of an action; a
    retained call's reaches count as unknown.
    """
    in_place: frozenset = frozenset()
    planner = Planner(function, context, in_place)
    if context.entry is None or function is context.entry:
        refuse_cta_overlaps(planner)
    refuse_thread_overlaps(planner)
    while True:
        try:
            planner.run()
            return
        except UnplacedError as unplaced:
            if unplaced.position in in_place:
                raise analysis.Error(f"sync: {unplaced}") from None
            in_place |= {unplaced.position}
            planner = Planner(function, context, in_place)


__all__ = [
    "StepKind",
    "SyncMetadata",
    "SyncPlanMetadata",
    "SyncStep",
    "analyze_sync",
]
