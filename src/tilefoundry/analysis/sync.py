"""The ``sync`` family: where lowering orders asynchronous memory effects.

It walks a Function's actions in lowering's order over the shared
``IterationScope`` and plans the commits, waits, fences and barriers each
access needs before it may follow an earlier one, or refuses the Function. See
[analysis §1.2.5](docs/spec/analysis.md#125-sync).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from operator import mul

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
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord, literal_axes
from tilefoundry.ir.mesh_scope import device_layout
from tilefoundry.ir.tir.memory import Copy, Fill
from tilefoundry.ir.tir.sync import SyncBarrier, classify, participation, whole_groups
from tilefoundry.ir.types import Mesh, ShardLayout, Split, StorageKind, TensorType
from tilefoundry.ir.types.dim import is_dim_op_call
from tilefoundry.ir.types.layout import ComposedLayout
from tilefoundry.ir.types.mesh import axis_keys, separate, within_scope
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.completion import Effect, Issued, completion_of
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
    register_storage,
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
_ORDER = ("commit", "wait", "proxy_fence", "barrier", "fence")


@dataclass(frozen=True)
class SyncStep:
    """One statement lowering emits.

    ``kind`` is ``commit``, ``wait``, ``proxy_fence``, ``barrier`` or ``fence``. A
    commit, a wait or a fence is for one completion ``group``; a barrier runs over
    ``mesh``.
    """

    kind: str
    mesh: Mesh | None = None
    group: str | None = None


@dataclass(frozen=True)
class SyncMetadata(IRMetadata):
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
class SyncPlanMetadata(IRMetadata):
    """Marks a Function the ``sync`` family planned; the steps sit on its nodes.

    ``shared`` is whether any of its actions touches shared memory.
    """

    shared: bool = False


@dataclass(frozen=True)
class _Use:
    """One effect of an action on shared memory bytes or on one register storage.

    ``threads`` is the layout that gives each of those bytes to one issuing
    thread, when there is one.
    """

    effect: Effect
    spans: tuple = ()
    register: SSAValue | None = None
    threads: ShardLayout | None = None


@dataclass(frozen=True)
class _Action:
    """One instruction or retained call lowering emits for an HIR node.

    ``position`` orders it among its scope's children; ``owner`` and ``tag``
    name it the way lowering reaches it.
    """

    position: tuple[int, int]
    owner: Expr
    tag: object
    uses: tuple[_Use, ...]
    name: str
    mesh: Mesh | None


@dataclass(frozen=True)
class _Access:
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
    register: SSAValue | None
    meshes: tuple[int, ...]
    state: int
    group: str | None = None
    proxy: str = "generic"
    chain: object = None
    issuers: Mesh | None = None
    unit: Mesh | None = None
    threads: ShardLayout | None = None
    ordered: frozenset = frozenset()
    fenced: bool = False
    group_fenced: bool = False


class _Unplaced(Exception):
    """A step that an asynchronous action needs has no place; it completes in place instead."""

    def __init__(self, position: tuple[int, int], reason: str):
        super().__init__(reason)
        self.position = position


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


def _region_mesh(scope) -> Mesh | None:
    """The mesh of the innermost ``MeshRegion`` around *scope*."""
    return next(
        (node.owner.mesh for node in scope.ancestors() if isinstance(node.owner, MeshRegion)), None
    )


def _one_coordinate(mesh: Mesh, axis: int, size: int, bases: range) -> bool:
    """Whether each group of *size* threads from *bases* reads one coordinate on *mesh*'s *axis*.

    Every thread of a group must be one of *mesh*'s. A thread reads
    ``((thread - start) // stride) % extent`` on an axis of its level, as the
    runtime's ``mesh_coords`` does.
    """
    first = 0
    for level in separate(mesh):
        layout = device_layout(level)
        if axis < first + len(layout.shape):
            break
        first += len(layout.shape)
    else:
        return False
    start = level.layout.offset if isinstance(level.layout, ComposedLayout) else 0
    extents, strides = layout.shape, layout.strides
    if not all(isinstance(value, int) for value in (start, *extents, *strides)):
        return False
    extent, stride = extents[axis - first], strides[axis - first]
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


class _Planner:
    """One walk of a function in lowering's order, repeated for loops until it settles.

    The actions at the positions in *in_place* complete right after they issue.
    """

    def __init__(self, function: Function, context: AnalyzeContext, in_place: frozenset):
        self.function = function
        self.context = context
        self.in_place = in_place
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
        self.issues: dict[tuple[int, int], list[SyncStep]] = {}
        self.after: dict[tuple[int, int], list[SyncStep]] = {}
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
        types = (source.type, target.type)
        return self.action(
            position, owner, tag, Copy(), (source, target), types, name, scope, copy_mesh(types)
        )

    def action(
        self, position, owner, tag, op, args, types, name: str, scope, mesh, *, filled=(), result=()
    ) -> _Action:
        """*op* issued for *owner* on *mesh* with *args* of *types* in its parameters' order.

        A parameter named in *filled* is issued with the call's *result* storage,
        and also with its argument when one is given.
        """
        if mesh is not None and (scope.mesh is None or not within_scope(mesh, scope.mesh)):
            raise AnalysisError(f"sync: {name} is issued on a mesh not within its scope's")
        params = call_params(op, len(args))
        leaves = {
            param.name: (() if arg is None else SSAValue.leaves(arg))
            + (tuple(result) if param.name in filled else ())
            for param, arg in zip(params, args, strict=True)
        }
        owners = self.context.target.storage_owners
        threads = {
            param.name: _threads(op, type_, owners)
            for param, type_ in zip(params, types, strict=True)
        }
        effects = completion_of(Issued(op, params, tuple(types)))
        uses = self.uses(effects, leaves, threads)
        return _Action(position, owner, tag, uses, name, mesh)

    def uses(self, effects, leaves, threads=None) -> tuple[_Use, ...]:
        """Each effect on the shared memory bytes and on each register storage it touches."""
        threads = threads or {}
        return tuple(
            use
            for effect in effects
            for spans in (self.spans(leaves[effect.param]),)
            for use in (
                *((_Use(effect, spans, threads=threads.get(effect.param)),) if spans else ()),
                *(
                    _Use(effect, register=register)
                    for register in self.registers(leaves[effect.param])
                ),
            )
        )

    def registers(self, leaves) -> tuple[SSAValue, ...]:
        """The register storages lowering gives these values."""
        return tuple(
            dict.fromkeys(
                register_storage(leaf, self.aliases_of, self.context.target)
                for leaf in leaves
                if leaf in self.values and self.values[leaf].lifetime.memory_level == "rmem"
            )
        )

    def call_actions(self, call: Call, event: int, scope) -> list[_Action]:
        """Each instruction or retained call lowering emits for *call* at *event*, in order."""
        callee = scope.attachment(call)
        kind = emission(call, retained=callee is not None)
        if kind is Emission.CALL:
            return self.call(call, event, callee.owner)
        if kind is Emission.FILL:
            types = (call.type, None)
            return [self.action(
                (event, 0), call, None, Fill(), (call, None), types, "Fill", scope,
                copy_mesh((call.type,)),
            )]
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
                types,
                type(op).__name__,
                scope,
                issue_mesh(op, issue_owner(types), scope.mesh),
                filled=tuple(param.name for param in produced),
                result=SSAValue.leaves(call),
            )
        ]

    def call(self, call: Call, event: int, callee: Function) -> list[_Action]:
        """A retained call, which reads its arguments and writes its results.

        Callees are planned before their callers, order their own accesses and
        complete them before returning. Unless the callee's plan touches no
        shared memory, the call also reads and writes unknown shared memory bytes
        and may leave ``cp.async`` groups in flight, so whatever conflicts with it
        waits for all of them.
        """
        leaves = {
            **{f"argument {index}": SSAValue.leaves(arg) for index, arg in enumerate(call.args)},
            "result": SSAValue.leaves(call),
        }
        effects = tuple(Effect(name, name == "result", "on_return") for name in leaves)
        uses = [use for use in self.uses(effects, leaves) if use.register is not None]
        plan = get_metadata(callee, SyncPlanMetadata)
        if plan is None or plan.shared:
            uses += [
                _Use(Effect("callee", write, "group", "cp_async"), (None,)) for write in (False, True)
            ]
        if not uses:
            return []
        return [_Action((event, 0), call, None, tuple(uses), f"the call to {callee.name!r}", None)]

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
        """Plan the Function and attach each node's steps to it as ``SyncMetadata``.

        A collective group still in flight when the Function returns makes its
        action complete in place.
        """
        state = self.walk(self.root, frozenset())
        for access, _seq in sorted(state, key=lambda item: item[1]):
            if access.unit is not None and access.state != _DONE:
                name = self.actions[access.position].name
                raise _Unplaced(access.position, f"{name} may still be in flight at the return")
        for expr in collect_exprs(self.function.body):
            detach_metadata(expr, SyncMetadata)
        steps: dict[int, tuple[Expr, list]] = {}
        for position, planned in sorted(self.planned.items()):
            owner, tag = self.names[position]
            steps.setdefault(id(owner), (owner, []))[1].append((tag, tuple(planned)))
        for tag, placed in (("issue", self.issues), ("after", self.after)):
            for position, planned in sorted(placed.items()):
                owner, _tag = self.names[position]
                steps.setdefault(id(owner), (owner, []))[1].append((tag, tuple(planned)))
        for owner, tagged in steps.values():
            attach(owner, SyncMetadata(tuple(tagged)))
        shared = any(use.spans for action in self.actions.values() for use in action.uses)
        attach(self.function, SyncPlanMetadata(shared))

    def walk(self, scope, state):
        for position, child in self.kids[id(scope)]:
            self.seq += 1
            self.entered[position] = self.seq
            self.paths.setdefault(position, self.path(scope, position))
            action = child if isinstance(child, _Action) else None
            for step in (*self.planned.get(position, ()), *self.issues.get(position, ())):
                state = self.effect(step, scope, state, self.entered[position], action)
            if action is not None:
                state = self.evaluate(action, scope, state)
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

    def complete_in_place(self, action: _Action, state):
        """Commit and wait for *action*'s groups right after it issues, in its issuing region."""
        steps = self.after.setdefault(action.position, [])
        for group in dict.fromkeys(
            use.effect.group for use in action.uses if use.effect.mechanism == "group"
        ):
            for kind in ("commit", "wait"):
                if SyncStep(kind, group=group) not in steps:
                    steps.append(SyncStep(kind, group=group))
        return frozenset(
            (replace(access, state=_DONE) if access.position == action.position else access, seq)
            for access, seq in state
        )

    def access(self, action: _Action, use: _Use, scope) -> _Access:
        """What *use* leaves for later accesses once *action* is issued in *scope*.

        An effect no protocol states is complete when its statement returns.
        """
        effect = use.effect
        status = _UNCOMMITTED if effect.mechanism == "group" else _DONE
        return _Access(
            action.position,
            effect.write,
            use.spans,
            use.register,
            _mesh_chain(scope),
            status,
            effect.group,
            effect.proxy,
            effect.chain,
            action.mesh or _region_mesh(scope),
            effect.unit,
            use.threads,
        )

    def order(self, action: _Action, use: _Use, scope, state):
        """Plan what must run before this use, and apply it to *state*."""
        meshes = _mesh_chain(scope)
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
            if access.state != _DONE:
                if access.unit is not None:
                    try:
                        self.collective(meeting, access.unit, access.issuers)
                    except AnalysisError as error:
                        raise _Unplaced(access.position, str(error)) from None
                group = access.group
                if access.state == _UNCOMMITTED:
                    state = self.plan(SyncStep("commit", group=group), meeting, position, state)
                state = self.plan(SyncStep("wait", group=group), meeting, position, state)
            if access.register is None:
                if use.effect.proxy == "async" and access.proxy == "generic":
                    state = self.plan(SyncStep("proxy_fence"), meeting, position, state)
                barrier = SyncStep("barrier", self.barrier_mesh(meeting))
                state = self.plan(barrier, meeting, position, state)
            same = (access.position, access.write, access.meshes, seq)
            if any(
                (other.position, other.write, other.meshes, later) == same
                and self.conflicts(other, use, meshes)
                for other, later in state
            ):
                raise AnalysisError(
                    f"sync: {action.name} cannot be ordered after "
                    f"{self.actions[access.position].name}"
                )

    def conflicts(self, access, use: _Use, meshes) -> bool:
        """Whether *use* must follow *access*.

        A register still in flight in a group takes no other access before the
        group's wait (PTX wgmma.wait_group); later effects of the same group only
        conflict when one writes and their chains differ. A completed access
        whose bytes the same threads touched, by the same layout, is behind *use*
        in each thread's own program order.
        """
        if use.register is not None or access.register is not None:
            if use.register != access.register or access.state == _DONE:
                return False
            if access.group is None or use.effect.group != access.group:
                return True
            chained = use.effect.chain is not None and use.effect.chain == access.chain
            return (use.effect.write or access.write) and not chained
        if not (use.effect.write or access.write):
            return False
        if not any(_overlap(a, b) for a in use.spans for b in access.spans):
            return False
        if use.threads is not None and use.threads == access.threads and access.state == _DONE:
            return False
        visible = use.effect.proxy == "generic" or access.proxy != "generic"
        return not any(
            meshes[: len(tag)] == tag and (visible or published)
            for tag, published in access.ordered
        )

    def group_fence(self, action: _Action, scope, state):
        """Plan the group's fence before an action whose registers were touched since the last.

        An action reading registers no chain links to takes the fence before each issue.
        """
        fenced = [use for use in action.uses if use.register is not None and use.effect.fence]
        if not fenced:
            return state
        every = any(use.effect.chain is None for use in fenced)

        def needed(use) -> bool:
            seen = [access for access, _seq in state if access.register == use.register]
            return not seen or any(
                not access.group_fenced and access.chain != use.effect.chain for access in seen
            )

        if not every and not any(needed(use) for use in fenced):
            return state
        step = SyncStep("fence", group=fenced[0].effect.group)
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
            planned.sort(key=lambda item: (_ORDER.index(item.kind), item.group or ""))
        return self.effect(step, scope, state, self.entered[position])

    def barrier_mesh(self, scope) -> Mesh:
        """The thread mesh a barrier at *scope* runs over, once every thread reaches it.

        It is the mesh of the innermost ``MeshRegion`` around *scope*.
        """
        mesh = _region_mesh(scope)
        try:
            barrier = None if mesh is None else classify(mesh)
        except VerifyError as error:
            raise AnalysisError(f"sync: {error}") from None
        if barrier is None or barrier is SyncBarrier.GRID:
            raise AnalysisError("sync: a barrier is needed in a scope that no thread mesh binds")
        self.uniform_loops(scope, mesh, "a barrier")
        return mesh

    def collective(self, scope, unit: Mesh, threads: Mesh | None) -> None:
        """Require whole *unit*s of threads, *threads* among them, to reach *scope* together."""
        mesh = _region_mesh(scope)
        if mesh is None:
            raise AnalysisError(
                "sync: a collective step is needed in a scope that no thread mesh binds"
            )
        try:
            size = participation(unit).count
            held, needed = participation(mesh), participation(threads or mesh)
            whole = whole_groups(mesh, size)
        except VerifyError as error:
            raise AnalysisError(f"sync: {error}") from None
        if not whole:
            raise AnalysisError(
                f"sync: a collective step is needed over {held.count} threads from {held.base}, "
                f"which are not whole groups of {size}"
            )
        if needed.base < held.base or needed.base + needed.count > held.base + held.count:
            raise AnalysisError(
                "sync: a collective step is needed outside the threads that issued it"
            )
        self.uniform_loops(scope, mesh, "a collective step", size)

    def uniform_loops(self, scope, mesh: Mesh, what: str, unit: int | None = None) -> None:
        """Require every loop around *scope* to have bounds the threads of *mesh* agree on.

        A barrier needs all of them to agree. A collective step of *unit* threads
        only needs the threads of each aligned group of *unit* to agree (PTX
        ``.aligned``), so the trip count may differ from one group to the next.
        """
        spanned = {topology.name for topology in mesh.topologies}
        groups = None
        if unit is not None and len(mesh.topologies) == 1:
            held = participation(mesh)
            groups = range(held.base, held.base + held.count, unit)

        def fixed(coordinates: Mesh, axis: int) -> bool:
            if axis_keys(coordinates)[axis][0] not in spanned:
                return True
            return groups is not None and _one_coordinate(coordinates, axis, unit, groups)

        entry = self.context.entry is None or self.function is self.context.entry
        uniform = {id(param) for param in self.function.params} if entry else set()
        for node in reversed(list(scope.ancestors())):
            loop = node.owner
            if not isinstance(loop, LoopRegion):
                continue
            bounds = (loop.start, loop.extent, loop.step)
            if not all(self.uniform(bound, uniform, fixed, node) for bound in bounds):
                raise AnalysisError(
                    f"sync: {what} is needed inside loop {loop.induction_var.name!r}, "
                    "whose bounds may differ between the threads that must reach it"
                )
            uniform.add(id(loop.induction_var))

    def uniform(self, expr, uniform: set[int], fixed, scope) -> bool:
        """Whether the threads that must agree see the same value of a loop bound.

        ``fixed(mesh, axis)`` says whether they read one coordinate on that mesh axis.
        """
        if isinstance(expr, Constant) or static_dim_value(expr) is not None:
            return True
        if isinstance(expr, Var):
            return id(expr) in uniform or id(scope.capture_root(expr)) in uniform
        if isinstance(expr, Call) and isinstance(expr.target, MeshCoord):
            axes = literal_axes(expr.args[0])
            count = len(axis_keys(expr.target.mesh))
            return axes is not None and all(
                0 <= axis < count and fixed(expr.target.mesh, axis) for axis in axes
            )
        if isinstance(expr, Call) and is_dim_op_call(expr):
            return all(self.uniform(arg, uniform, fixed, scope) for arg in expr.args)
        return False

    def effect(self, step: SyncStep, scope, state, limit, action=None):
        meshes = _mesh_chain(scope)
        registers = (
            {use.register for use in action.uses if use.register is not None}
            if step.kind == "fence" and action is not None
            else set()
        )
        out = set()
        for access, seq in state:
            issued = seq < limit and access.meshes[: len(meshes)] == meshes
            grouped = access.group == step.group
            if issued and step.kind == "commit" and grouped and access.state == _UNCOMMITTED:
                access = replace(access, state=0)
            elif issued and step.kind == "wait" and grouped and access.state >= 0:
                access = replace(access, state=_DONE)
            elif issued and step.kind == "proxy_fence" and access.state == _DONE:
                access = replace(access, fenced=True)
            elif issued and step.kind == "barrier" and access.state == _DONE:
                access = replace(access, ordered=access.ordered | {(meshes, access.fenced)})
            elif step.kind == "fence" and seq < limit and access.register in registers:
                access = replace(access, group_fenced=True)
            out.add((access, seq))
        return frozenset(out)


def analyze_sync(function: Function, context: AnalyzeContext) -> None:
    """Attach the synchronisation plan lowering emits for *function*.

    An asynchronous action one of whose steps has no place completes right after
    it issues instead, and the Function is planned again.
    """
    in_place: frozenset = frozenset()
    while True:
        try:
            _Planner(function, context, in_place).run()
            return
        except _Unplaced as unplaced:
            if unplaced.position in in_place:
                raise AnalysisError(f"sync: {unplaced}") from None
            in_place |= {unplaced.position}


__all__ = [
    "SyncMetadata",
    "SyncPlanMetadata",
    "SyncStep",
    "analyze_sync",
]
