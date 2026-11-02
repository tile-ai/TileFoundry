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
    IRMetadata,
    SSAValue,
    Var,
    VerifyError,
    get_metadata,
)
from tilefoundry.ir.core import attach_metadata as attach
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.tir.memory import Copy, Fill
from tilefoundry.ir.tir.sync import SyncBarrier, classify
from tilefoundry.ir.types import Mesh
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
    selected_op,
)
from .errors import AnalysisError
from .liveness import LivenessMetadata
from .memory import owned_bytes
from .visitor import AnalyzeContext

_DONE = -1
_UNCOMMITTED = -2
_UNKNOWN = -3
_LOOP_LIMIT = 64


@dataclass(frozen=True)
class SyncStep:
    """One statement lowering emits at a position: ``commit``, ``wait`` or ``barrier``."""

    kind: str
    mesh: Mesh | None = None


@dataclass(frozen=True)
class SyncRole:
    """One parameter of an action's instruction and the shared memory its values own.

    ``spans`` are byte ranges ``(start, end)``; None stands for unknown bytes.
    """

    name: str
    read: bool
    write: bool
    spans: tuple


@dataclass(frozen=True)
class SyncPlanMetadata(IRMetadata):
    """The steps lowering emits before direct children of lexical scopes."""

    steps: tuple[tuple[tuple[int, int], tuple[SyncStep, ...]], ...] = ()

    def at(self, position: tuple[int, int]) -> tuple[SyncStep, ...]:
        """The steps lowering emits before the child at *position*."""
        return dict(self.steps).get(position, ())


@dataclass(frozen=True)
class _Action:
    """One instruction lowering issues for an HIR event, with what it touches."""

    position: tuple[int, int]
    roles: tuple[SyncRole, ...]
    completion: Completion
    name: str

    @property
    def accesses(self) -> tuple[tuple[bool, tuple], ...]:
        """Each read and write as ``(write, spans)``, reads of a parameter first."""
        return tuple(
            (write, role.spans)
            for role in self.roles
            for write in (False, True)
            if (role.write if write else role.read)
        )


@dataclass(frozen=True)
class _Access:
    """One access a later one may have to follow.

    ``ordered`` names the barriers that ordered it, each as the mesh chain of the
    scope it ran in; it orders the access only for threads inside that scope.
    """

    position: tuple[int, int]
    write: bool
    spans: tuple
    meshes: tuple[int, ...]
    state: int
    ordered: frozenset = frozenset()


def event_of(liveness: LivenessMetadata, expr) -> int | None:
    """The liveness event at which lowering emits *expr*'s actions, if it has one."""
    lifetime = liveness.interval_of(SSAValue(expr, 0))
    return None if lifetime is None else lifetime.defined_at


def region_position(liveness: LivenessMetadata, region) -> tuple[int, int]:
    """The position of an inner region among the children of its scope."""
    if isinstance(region, LoopRegion):
        return (liveness.loop_windows[id(region)][0], -1)
    return (next(item.entered_at for item in liveness.regions if item.region is region), -1)


def carry_positions(liveness: LivenessMetadata, loop: LoopRegion, slot: int):
    """The positions of a loop slot's entry copy, just before the loop, and backedge copy."""
    start, backedge = liveness.loop_windows[id(loop)]
    return (start, slot - len(loop.yield_values) - 1), (backedge, slot)


def _overlap(a, b) -> bool:
    return a is None or b is None or (a[0] < b[1] and b[0] < a[1])


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
                    self.kids[id(node.parent)].append((region_position(self.liveness, expr), node))
                if isinstance(expr, LoopRegion):
                    self.loop_actions(expr, node)
            elif isinstance(expr, Call):
                event = event_of(self.liveness, expr)
                if event is not None:
                    scope = self.locations[id(expr)]
                    for action in self.call_actions(expr, event, scope):
                        self.add(scope, action)
        for children in self.kids.values():
            children.sort(key=lambda item: item[0])

    def add(self, scope, action: _Action) -> None:
        self.actions[action.position] = action
        self.kids[id(scope)].append((action.position, action))

    def loop_actions(self, loop: LoopRegion, inner) -> None:
        outer = inner.parent
        for slot in range(len(loop.yield_values)):
            entry, backedge = carry_positions(self.liveness, loop, slot)
            target = loop.params[slot]
            if copies_carry_entry(loop, slot, self.aliases_of):
                self.add(outer, self.copy(entry, loop.args[slot], target, outer, "carry entry copy"))
            if copies_carry_backedge(loop, slot, self.aliases_of):
                source = loop.yield_values[slot]
                self.add(inner, self.copy(backedge, source, target, inner, "carry backedge copy"))

    def copy(self, position, source, target, scope, name: str) -> _Action:
        """A ``Copy`` lowering adds from *source* into *target*'s storage."""
        mesh = copy_mesh((source.type, target.type))
        return self.action(position, Copy(), (source, target), name, scope, mesh)

    def action(self, position, op, args, name: str, scope, mesh, *, filled=(), result=()) -> _Action:
        """*op* issued on *mesh* at *position* with *args* in its parameters' order.

        A parameter named in *filled* is issued with the call's *result* storage,
        and also with its argument when one is given.
        """
        if mesh is not None and (scope.mesh is None or not within_scope(mesh, scope.mesh)):
            raise AnalysisError(
                f"sync: {name} at {position} is issued on a mesh not within its scope's"
            )
        params = call_params(op, len(args))
        roles = tuple(
            SyncRole(
                param.name,
                bool(param.effect & MemoryEffect.READ),
                bool(param.effect & MemoryEffect.WRITE),
                (() if arg is None else self.spans(SSAValue.leaves(arg)))
                + (self.spans(result) if param.name in filled else ()),
            )
            for param, arg in zip(params, args, strict=True)
        )
        return _Action(position, roles, completion_of(op), name)

    def call_actions(self, call: Call, event: int, scope) -> list[_Action]:
        """Each instruction lowering issues for *call* at *event*, in order."""
        callee = scope.attachment(call)
        kind = emission(call, retained=callee is not None)
        if kind is Emission.CALL:
            self.admit(call, callee.owner)
            return []
        if kind is Emission.FILL:
            mesh = copy_mesh((call.type,))
            return [self.action((event, 0), Fill(), (call, None), "Fill", scope, mesh)]
        if kind is Emission.INSERT_SLICE:
            actions = []
            if copies_insert_slice_base(call, self.aliases_of):
                position = (event, len(actions))
                actions.append(self.copy(position, call.args[0], call, scope, "InsertSlice base"))
            if copies_insert_slice_update(call, self.aliases_of):
                position = (event, len(actions))
                actions.append(self.copy(position, call.args[1], call, scope, "InsertSlice update"))
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
                op,
                tuple(given.get(param.name) for param in params),
                type(op).__name__,
                scope,
                issue_mesh(op, issue_owner(types), scope.mesh),
                filled=tuple(param.name for param in produced),
                result=SSAValue.leaves(call),
            )
        ]

    def admit(self, call: Call, callee: Function) -> None:
        """Refuse a retained call whose callee's own summary holds shared memory."""
        summary = get_metadata(callee, AllocationSummaryMetadata)
        if summary is None or any(
            value.lifetime.memory_level == "smem" for value in summary.values
        ):
            raise AnalysisError(
                f"sync: {self.function.name!r} calls {callee.name!r}, which "
                "touches shared memory; call summaries are not supported yet"
            )

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
            child = region_position(self.liveness, scope.owner)
            scope = scope.parent
        return tuple(reversed(steps))

    def run(self) -> SyncPlanMetadata:
        state = self.walk(self.root, frozenset())
        for access, _seq in sorted(state, key=lambda item: item[1]):
            if access.state == _UNKNOWN:
                raise AnalysisError(
                    f"sync: {self.actions[access.position].name} may still touch shared memory "
                    f"when {self.function.name!r} returns, and no protocol states its completion"
                )
        return SyncPlanMetadata(
            tuple((position, tuple(planned)) for position, planned in sorted(self.planned.items()))
        )

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
        accesses = [(write, spans) for write, spans in action.accesses if spans]
        for write, spans in accesses:
            state = self.order(action, write, spans, scope, state)
        status = {
            Completion.ON_RETURN: _DONE,
            Completion.CP_ASYNC_GROUP: _UNCOMMITTED,
        }.get(action.completion, _UNKNOWN)
        self.seq += 1
        made = frozenset(
            (
                _Access(
                    action.position,
                    write,
                    spans,
                    _mesh_chain(scope),
                    status if write else (_DONE if status == _DONE else _UNKNOWN),
                ),
                self.seq,
            )
            for write, spans in accesses
        )
        return state | made

    def order(self, action: _Action, write, spans, scope, state):
        """Plan what must run before this access, and apply it to *state*."""
        meshes = _mesh_chain(scope)
        while True:
            pending = [
                (access, seq)
                for access, seq in sorted(state, key=lambda item: (item[1], item[0].position))
                if self.conflicts(access, write, spans, meshes)
            ]
            if not pending:
                return state
            access, seq = pending[0]
            if access.state == _UNKNOWN:
                raise AnalysisError(
                    f"sync: {action.name} must follow {self.actions[access.position].name}, "
                    "whose completion no protocol states"
                )
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
                and self.conflicts(other, write, spans, meshes)
                for other, later in state
            ):
                raise AnalysisError(
                    f"sync: {action.name} cannot be ordered after "
                    f"{self.actions[access.position].name}"
                )

    def conflicts(self, access, write, spans, meshes) -> bool:
        if not (write or access.write):
            return False
        if not any(_overlap(a, b) for a in spans for b in access.spans):
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
            axis = static_dim_value(expr.args[0])
            keys = axis_keys(expr.target.mesh)
            return axis is not None and 0 <= axis < len(keys) and keys[axis][0] not in spanned
        if isinstance(expr, Call) and is_dim_op_call(expr):
            return all(self.uniform(arg, uniform, spanned, scope) for arg in expr.args)
        return False

    def effect(self, step: SyncStep, scope, state, limit):
        meshes = _mesh_chain(scope)
        out = set()
        for access, seq in state:
            issued = seq < limit and access.meshes[: len(meshes)] == meshes
            if issued and step.kind == "commit" and access.state not in (_DONE, _UNKNOWN):
                access = replace(access, state=0 if access.state == _UNCOMMITTED else access.state)
            elif issued and step.kind == "wait" and access.state >= 0:
                access = replace(access, state=_DONE)
            elif issued and step.kind == "barrier" and access.state == _DONE:
                access = replace(access, ordered=access.ordered | {meshes})
            out.add((access, seq))
        return frozenset(out)


def analyze_sync(function: Function, context: AnalyzeContext) -> None:
    """Attach the synchronisation plan lowering emits for *function*."""
    attach(function, _Planner(function, context).run())


__all__ = [
    "SyncPlanMetadata",
    "SyncStep",
    "analyze_sync",
]
