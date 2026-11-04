"""The ``sync`` family: where lowering orders shared memory accesses.

It walks a Function's actions in lowering's order over the shared
``IterationScope`` and plans a commit, a wait and a barrier before each access
that must follow an earlier one, or refuses the Function. See
[analysis §1.2.5](docs/spec/analysis.md#125-sync).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    IRMetadata,
    SSAValue,
    Var,
    VerifyError,
    detach_metadata,
    get_metadata,
)
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord, literal_axes
from tilefoundry.ir.tir.memory import Copy, Fill
from tilefoundry.ir.tir.sync import SyncBarrier, classify
from tilefoundry.ir.types import Mesh, ShardLayout, Split, StorageKind, TensorType
from tilefoundry.ir.types.dim import is_dim_op_call
from tilefoundry.ir.types.mesh import axis_keys, within_scope
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.completion import Completion, completion_of
from tilefoundry.visitor_registry.instruction import call_params

from .allocation import AllocationSummaryMetadata
from .emission import (
    Emission,
    accepted_aliases,
    copies_carry_backedge,
    copies_carry_entry,
    copies_insert_slice_base,
    copies_insert_slice_update,
    copy_mesh,
    emission,
    issue_mesh,
    issue_owner,
    operand_roles,
    operand_types,
    required_execution_mesh,
    selected_op,
)
from .errors import AnalysisError
from .liveness import LivenessMetadata
from .memory import owned_bytes
from .visitor import AnalyzeContext

_DONE = -1
_UNCOMMITTED = -2
_LOOP_LIMIT = 64


@dataclass(frozen=True)
class SyncStep:
    """One statement lowering emits: ``commit``, ``wait`` or ``barrier``."""

    kind: str
    mesh: Mesh | None = None


@dataclass(frozen=True)
class SyncRole:
    """One parameter of an action's instruction and the shared memory its values own.

    ``spans`` are byte ranges ``(start, end)``; None stands for unknown bytes.
    ``threads`` is the layout that gives each byte the parameter touches to one
    issuing thread, when there is one.
    """

    name: str
    read: bool
    write: bool
    spans: tuple
    threads: ShardLayout | None = None


@dataclass(frozen=True)
class SyncMetadata(IRMetadata):
    """The steps lowering emits before what it lowers for one HIR node.

    ``steps`` pairs a tag with its steps. The tag None is the node's own
    statements: a Call's instruction or Fill, or entering a region. Any other
    tag names a copy lowering adds for the node: ``"base"`` or ``"update"`` of
    an InsertSlice, ``("entry", slot)`` or ``("backedge", slot)`` of a loop.
    """

    steps: tuple[tuple[object, tuple[SyncStep, ...]], ...] = ()

    def at(self, tag=None) -> tuple[SyncStep, ...]:
        """The steps lowering emits before the statements *tag* names."""
        return dict(self.steps).get(tag, ())


@dataclass(frozen=True)
class SyncPlanMetadata(IRMetadata):
    """Marks a Function the ``sync`` family planned; the steps sit on its nodes.

    ``shared`` is whether any of its actions touches shared memory.
    """

    shared: bool = False


@dataclass(frozen=True)
class _Action:
    """One instruction lowering issues for an HIR node, with what it touches.

    ``position`` orders it among its scope's children; ``owner`` and ``tag``
    name it the way lowering reaches it.
    """

    position: tuple[int, int]
    owner: Expr
    tag: object
    roles: tuple[SyncRole, ...]
    completion: Completion
    name: str

    @property
    def accesses(self) -> tuple[tuple[bool, tuple, ShardLayout | None], ...]:
        """Each read and write as ``(write, spans, threads)``, reads of a parameter first."""
        return tuple(
            (write, role.spans, role.threads)
            for role in self.roles
            for write in (False, True)
            if (role.write if write else role.read)
        )


@dataclass(frozen=True)
class _Access:
    """One access a later one may have to follow.

    ``ordered`` names the barriers that ordered it, each as the mesh chain of the
    scope it ran in; it orders the access only for threads inside that scope.
    ``threads`` is its role's layout giving each byte to one issuing thread.
    """

    position: tuple[int, int]
    write: bool
    spans: tuple
    meshes: tuple[int, ...]
    state: int
    ordered: frozenset = frozenset()
    threads: ShardLayout | None = None


def _event(liveness: LivenessMetadata, expr) -> int | None:
    """The liveness event at which lowering emits *expr*'s actions, if it has one."""
    lifetime = liveness.interval_of(SSAValue(expr, 0))
    return None if lifetime is None else lifetime.defined_at


def _region_position(liveness: LivenessMetadata, region) -> tuple[int, int]:
    """Where an inner region sits among the children of its scope."""
    if isinstance(region, LoopRegion):
        return (liveness.loop_windows[id(region)][0], -1)
    return (next(item.entered_at for item in liveness.regions if item.region is region), -1)


def _carry_positions(liveness: LivenessMetadata, loop: LoopRegion, slot: int):
    """Where a loop slot's entry copy, just before the loop, and backedge copy sit."""
    start, backedge = liveness.loop_windows[id(loop)]
    return (start, slot - len(loop.yield_values) - 1), (backedge, slot)


def _overlap(a, b) -> bool:
    return a is None or b is None or (a[0] < b[1] and b[0] < a[1])


def _threads(op, type_, owners) -> ShardLayout | None:
    """The layout giving each byte of a shared memory operand of *type_* to one issuing thread.

    Each thread issues *op* on its own shard when *op* declares no execution
    mesh. A layout that splits along every mesh axis whose units share shared
    memory, as *owners* states, gives each shard's bytes to that thread alone.
    """
    if required_execution_mesh(op) is not None or not isinstance(type_, TensorType):
        return None
    layout = type_.layout
    if type_.storage is not StorageKind.SMEM or not isinstance(layout, ShardLayout):
        return None
    axes = zip(layout.attrs, axis_keys(layout.mesh), strict=True)
    shared = (level for attr, (level, _) in axes if not isinstance(attr, Split))
    return layout if all(owners.private(type_.storage, level) for level in shared) else None


def _mesh_chain(scope) -> tuple[int, ...]:
    return tuple(id(s.owner) for s in reversed(list(scope.ancestors())) if isinstance(s.owner, MeshRegion))


class _Planner:
    """One walk of a function in lowering's order, repeated for loops until it settles."""

    def __init__(self, function: Function, context: AnalyzeContext):
        self.function = function
        self.context = context
        self.liveness = get_metadata(function, LivenessMetadata)
        self.summary = get_metadata(function, AllocationSummaryMetadata)
        if self.liveness is None or self.summary is None:
            raise AnalysisError(f"sync: {function.name!r} has no memory analysis")
        self.values = {value.value: value for value in self.summary.values}
        self.returned = set(self.roots(SSAValue.leaves(function.body)))
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
        self.actions: dict[tuple[int, int], _Action] = {}
        self.names: dict[tuple[int, int], tuple[Expr, object]] = {}
        self.paths: dict[tuple[int, int], tuple] = {}
        self.planned: dict[tuple[int, int], list[SyncStep]] = {}
        self.entered: dict[tuple[int, int], int] = {}
        self.seq = 0
        self.build()

    def aliases_of(self, leaf):
        value = self.values.get(leaf)
        return () if value is None else accepted_aliases(value)

    def build(self) -> None:
        for expr in collect_exprs(self.function.body):
            if isinstance(expr, (MeshRegion, LoopRegion)) and id(expr) in self.nodes:
                node = self.nodes[id(expr)]
                if not any(child is node for _position, child in self.kids[id(node.parent)]):
                    position = _region_position(self.liveness, expr)
                    self.names[position] = (expr, None)
                    self.kids[id(node.parent)].append((position, node))
                if isinstance(expr, LoopRegion):
                    self.loop_actions(expr, node)
            elif isinstance(expr, Call):
                event = _event(self.liveness, expr)
                if event is not None:
                    scope = self.locations[id(expr)]
                    for action in self.call_actions(expr, event, scope):
                        self.add(scope, action)
        for children in self.kids.values():
            children.sort(key=lambda item: item[0])

    def add(self, scope, action: _Action) -> None:
        self.actions[action.position] = action
        self.names[action.position] = (action.owner, action.tag)
        self.kids[id(scope)].append((action.position, action))

    def loop_actions(self, loop: LoopRegion, inner) -> None:
        outer = inner.parent
        for slot in range(len(loop.yield_values)):
            entry, backedge = _carry_positions(self.liveness, loop, slot)
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

    def copy(self, position, owner, tag, source, target, scope, name: str) -> _Action:
        """A ``Copy`` lowering adds for *owner* from *source* into *target*'s storage."""
        mesh = copy_mesh((source.type, target.type))
        return self.action(position, owner, tag, Copy(), (source, target), name, scope, mesh)

    def action(
        self,
        position,
        owner,
        tag,
        op,
        args,
        name: str,
        scope,
        mesh,
        *,
        types=None,
        filled=(),
        result=(),
    ) -> _Action:
        """*op* issued for *owner* on *mesh* with *args* in its parameters' order.

        *types* are the operand types it is issued with, the arguments' by
        default. A parameter named in *filled* is issued with the call's
        *result* storage, and also with its argument when one is given.
        """
        if mesh is not None and (scope.mesh is None or not within_scope(mesh, scope.mesh)):
            raise AnalysisError(f"sync: {name} is issued on a mesh not within its scope's")
        params = call_params(op, len(args))
        if types is None:
            types = tuple(None if arg is None else arg.type for arg in args)
        roles = tuple(
            SyncRole(
                param.name,
                bool(param.effect & MemoryEffect.READ),
                bool(param.effect & MemoryEffect.WRITE),
                (() if arg is None else self.spans(SSAValue.leaves(arg)))
                + (self.spans(result) if param.name in filled else ()),
                _threads(op, type_, self.context.target.storage_owners),
            )
            for param, arg, type_ in zip(params, args, types, strict=True)
        )
        return _Action(position, owner, tag, roles, completion_of(op), name)

    def call_actions(self, call: Call, event: int, scope) -> list[_Action]:
        """Each instruction lowering issues for *call* at *event*, in order."""
        callee = scope.attachment(call)
        kind = emission(call, retained=callee is not None)
        if kind is Emission.CALL:
            return self.call(call, event, callee.owner)
        if kind is Emission.FILL:
            mesh = copy_mesh((call.type,))
            return [self.action((event, 0), call, None, Fill(), (call, None), "Fill", scope, mesh)]
        if kind is Emission.INSERT_SLICE:
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
        op = selected_op(call, self.context.target) if kind is Emission.INSTRUCTION else None
        if op is None:
            return []
        types = operand_types(call, op)
        params = call_params(op, len(types))
        supplied, produced = operand_roles(call, params)
        given = dict(zip((param.name for param in supplied), call.args, strict=True))
        return [
            self.action(
                (event, 0),
                call,
                None,
                op,
                tuple(given.get(param.name) for param in params),
                type(op).__name__,
                scope,
                issue_mesh(op, issue_owner(types), scope.mesh),
                types=types,
                filled=tuple(param.name for param in produced),
                result=SSAValue.leaves(call),
            )
        ]

    def call(self, call: Call, event: int, callee: Function) -> list[_Action]:
        """A retained call, as one action unless its callee's plan touches no shared memory.

        Callees are planned before their callers and order their own accesses.
        Without a summary of what a callee touches, the call reads and writes
        unknown bytes and may leave ``cp.async`` groups in flight, so whatever
        conflicts with it waits for all of them.
        """
        plan = get_metadata(callee, SyncPlanMetadata)
        if plan is not None and not plan.shared:
            return []
        role = SyncRole("callee", True, True, (None,))
        name = f"the call to {callee.name!r}"
        return [_Action((event, 0), call, None, (role,), Completion.CP_ASYNC_GROUP, name)]

    def roots(self, leaves) -> tuple:
        """The allocation roots these values live in."""
        return tuple(self.summary.root(leaf).value for leaf in leaves if leaf in self.values)

    def spans(self, leaves) -> tuple:
        """The shared memory byte ranges these values own; None stands for unknown.

        A returned allocation may live in the output its caller passes.
        """
        return tuple(
            None if self.summary.root(leaf).value in self.returned else owned_bytes(self.summary, leaf)
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
            child = _region_position(self.liveness, scope.owner)
            scope = scope.parent
        return tuple(reversed(steps))

    def run(self) -> None:
        """Plan the Function and attach each node's steps to it as ``SyncMetadata``."""
        self.walk(self.root, frozenset())
        for expr in collect_exprs(self.function.body):
            detach_metadata(expr, SyncMetadata)
        steps: dict[int, tuple[Expr, list]] = {}
        for position, planned in sorted(self.planned.items()):
            owner, tag = self.names[position]
            steps.setdefault(id(owner), (owner, []))[1].append((tag, tuple(planned)))
        for owner, tagged in steps.values():
            attach(owner, SyncMetadata(tuple(tagged)))
        shared = any(
            spans for action in self.actions.values() for _write, spans, _threads in action.accesses
        )
        attach(self.function, SyncPlanMetadata(shared))

    def walk(self, scope, state):
        for position, child in self.kids[id(scope)]:
            self.seq += 1
            self.entered[position] = self.seq
            self.paths.setdefault(position, self.path(scope, position))
            for step in self.planned.get(position, ()):
                state = self.effect(step, scope, state, self.entered[position])
            if isinstance(child, _Action):
                state = self.evaluate(child, scope, state)
            elif isinstance(child.owner, LoopRegion):
                state = self.loop(child, state)
            else:
                state = self.walk(child, state)
        return state

    def loop(self, scope, state):
        """Join the head with every body end until it settles; the head is also the exit."""
        start = self.seq
        self.seq += 1
        head = state
        for _ in range(_LOOP_LIMIT):
            end = self.walk(scope, head)
            joined = frozenset((access, min(seq, start)) for access, seq in head | end)
            if joined == head:
                return head
            head = joined
        raise AnalysisError("sync: the analysis of a loop did not settle within its limit")

    def evaluate(self, action: _Action, scope, state):
        accesses = [access for access in action.accesses if access[1]]
        for write, spans, threads in accesses:
            state = self.order(action, write, spans, threads, scope, state)
        status = _UNCOMMITTED if action.completion is Completion.CP_ASYNC_GROUP else _DONE
        self.seq += 1
        meshes = _mesh_chain(scope)
        made = frozenset(
            (_Access(action.position, write, spans, meshes, status, threads=threads), self.seq)
            for write, spans, threads in accesses
        )
        return state | made

    def order(self, action: _Action, write, spans, threads, scope, state):
        """Plan what must run before this access, and apply it to *state*."""
        meshes = _mesh_chain(scope)
        while True:
            pending = [
                (access, seq)
                for access, seq in sorted(state, key=lambda item: (item[1], item[0].position))
                if self.conflicts(access, write, spans, threads, meshes)
            ]
            if not pending:
                return state
            access, seq = pending[0]
            meeting, position = self.meeting(access.position, action.position)
            if access.state != _DONE:
                if access.state == _UNCOMMITTED:
                    state = self.plan(SyncStep("commit"), meeting, position, state)
                state = self.plan(SyncStep("wait"), meeting, position, state)
            barrier = SyncStep("barrier", self.barrier_mesh(meeting))
            state = self.plan(barrier, meeting, position, state)
            same = (access.position, access.write, access.meshes, seq)
            if any(
                (other.position, other.write, other.meshes, later) == same
                and self.conflicts(other, write, spans, threads, meshes)
                for other, later in state
            ):
                raise AnalysisError(
                    f"sync: {action.name} cannot be ordered after "
                    f"{self.actions[access.position].name}"
                )

    def conflicts(self, access, write, spans, threads, meshes) -> bool:
        """Whether this access must follow *access* across threads or its completion.

        A completed access whose bytes the same threads touched, by the same layout,
        is behind this one in each thread's own program order.
        """
        if not (write or access.write):
            return False
        if not any(_overlap(a, b) for a in spans for b in access.spans):
            return False
        if threads is not None and threads == access.threads and access.state == _DONE:
            return False
        return not any(meshes[: len(tag)] == tag for tag in access.ordered)

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
        if all(existing.kind != step.kind for existing in planned):
            planned.append(step)
            planned.sort(key=lambda item: ("commit", "wait", "barrier").index(item.kind))
        return self.effect(step, scope, state, self.entered[position])

    def barrier_mesh(self, scope) -> Mesh:
        """The thread mesh a barrier at *scope* runs over, once every thread reaches it.

        It is the mesh of the innermost ``MeshRegion`` around *scope*.
        """
        mesh = next(
            (node.owner.mesh for node in scope.ancestors() if isinstance(node.owner, MeshRegion)),
            None,
        )
        try:
            barrier = None if mesh is None else classify(mesh)
        except VerifyError as error:
            raise AnalysisError(f"sync: {error}") from None
        if barrier is None or barrier is SyncBarrier.GRID:
            raise AnalysisError("sync: a barrier is needed in a scope that no thread mesh binds")
        spanned = {topology.name for topology in mesh.topologies}
        entry = self.context.entry is None or self.function is self.context.entry
        uniform = {id(param) for param in self.function.params} if entry else set()
        for node in reversed(list(scope.ancestors())):
            loop = node.owner
            if not isinstance(loop, LoopRegion):
                continue
            bounds = (loop.start, loop.extent, loop.step)
            if not all(self.uniform(bound, uniform, spanned, node) for bound in bounds):
                raise AnalysisError(
                    f"sync: a barrier is needed inside loop {loop.induction_var.name!r}, "
                    "whose bounds may differ between the threads that must reach it"
                )
            uniform.add(id(loop.induction_var))
        return mesh

    def uniform(self, expr, uniform: set[int], spanned: set[str], scope) -> bool:
        """Whether every thread of a barrier's mesh sees the same value of a loop bound."""
        if isinstance(expr, Constant) or static_dim_value(expr) is not None:
            return True
        if isinstance(expr, Var):
            return id(expr) in uniform or id(scope.capture_root(expr)) in uniform
        if isinstance(expr, Call) and isinstance(expr.target, MeshCoord):
            axes = literal_axes(expr.args[0])
            keys = axis_keys(expr.target.mesh)
            return axes is not None and all(
                0 <= axis < len(keys) and keys[axis][0] not in spanned for axis in axes
            )
        if isinstance(expr, Call) and is_dim_op_call(expr):
            return all(self.uniform(arg, uniform, spanned, scope) for arg in expr.args)
        return False

    def effect(self, step: SyncStep, scope, state, limit):
        meshes = _mesh_chain(scope)
        out = set()
        for access, seq in state:
            issued = seq < limit and access.meshes[: len(meshes)] == meshes
            if issued and step.kind == "commit" and access.state != _DONE:
                access = replace(access, state=0 if access.state == _UNCOMMITTED else access.state)
            elif issued and step.kind == "wait" and access.state >= 0:
                access = replace(access, state=_DONE)
            elif issued and step.kind == "barrier" and access.state == _DONE:
                access = replace(access, ordered=access.ordered | {meshes})
            out.add((access, seq))
        return frozenset(out)


def analyze_sync(function: Function, context: AnalyzeContext) -> None:
    """Attach the synchronisation plan lowering emits for *function*."""
    _Planner(function, context).run()


__all__ = [
    "SyncMetadata",
    "SyncPlanMetadata",
    "SyncStep",
    "analyze_sync",
]
