"""The structure of one training step: nodes on ranks, their data edges, and the values they hold.

The step is read the way the kernels run (D62). A call of a kernel Module is one node
on each rank of its scope; an Op outside every kernel is one node on each rank; a
Reshard that moves data is one communication node on each rank that takes part. Loops
with a static trip count are unrolled and calls of other Modules are entered, so a node
is one execution on one rank.

A value is one storage on one rank: a kernel's arguments and results, and the results
of the Ops outside kernels. What a kernel computes inside never reaches gmem and is not
a value. A view shares the storage of what it views, by the rule the allocator uses
(D10); a kernel result that is (a view of) one of its arguments is that argument.

Nothing here runs the step: the structure comes from the types, and the flop counts
from the core's compute-cost analysis.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from tilefoundry import analysis, target
from tilefoundry.interfaces.access import aliased_operand
from tilefoundry.ir import core, expr, types, utils
from tilefoundry.ir.hir.sharding import Reshard
from tilefoundry.ir.hir.tensor import TupleGetItem

from .configs import AXES, GPUS, Config
from .model import TARGET, TOPOLOGIES

KERNELS = (
    "AttnIn",
    "FlashAttention",
    "AttnOut",
    "Router",
    "Experts",
    "Combine",
    "MoE",
    "Embed",
    "Head",
    "NewtonSchulz",
)
COLLECTIVES = ("all_gather", "reduce_scatter", "all_reduce", "all_to_all")
KINDS = ("compute",) + COLLECTIVES + ("send", "recv")
SMALL = 64
"""Bytes at or under which a value computed only from indices and constants is free."""


@dataclass(frozen=True)
class Node:
    """One execution on one rank: a kernel call, an Op outside kernels, or one rank's part of a communication.

    ``bytes`` is, for compute, what the node reads and writes at its boundary; for a
    collective, the rank's local input (the local output for ``reduce_scatter``); for
    ``send`` and ``recv``, the bytes moved. ``group`` is the participants of a
    collective, ``(src, dst)`` of a send or receive, and empty for compute.
    """

    id: int
    rank: int
    name: str
    kind: str
    op: str
    phase: str
    flops: int
    bytes: int
    group: tuple[int, ...]
    deps: tuple[int, ...]
    tag: int = 0


@dataclass(frozen=True)
class Value:
    """One storage on one rank: its bytes, the node that writes it and the nodes that read it.

    A step input has no producer (-1) and is held from the start; a step result is held
    to the end.
    """

    id: int
    rank: int
    name: str
    bytes: int
    producer: int
    consumers: tuple[int, ...]
    persistent: bool


@dataclass(frozen=True)
class Facts:
    config: str
    recompute: str
    ranks: int
    nodes: tuple[Node, ...]
    values: tuple[Value, ...]
    peak: int
    peak_rank: int
    capacity: int
    communication_bytes: int
    seconds: float = 0.0


def facts(step: core.Module, config: Config, recompute: str) -> Facts:
    """Read *step* (a ``build_step`` Module) into nodes and values."""
    started = time.time()
    checked, function = analysis.check_program(step, step.entry_function(), budget=10000)
    analysis.analyze(checked, function, analysis="compute-cost")
    reader = Reader(checked, config)
    result = reader.step(function)
    nodes, values = reader.finish(result)
    peaks = program_peaks(nodes, values)
    peak_rank = max(peaks, key=lambda rank: peaks[rank])
    memory = TARGET.get_facts(target.MemoryHierarchyFacts).explicit("gmem")
    if memory is None or memory.capacity_bytes is None:
        raise ValueError("target states no gmem capacity")
    communication = sum(node.bytes for node in nodes if node.kind in COLLECTIVES + ("send",))
    return Facts(
        config=config.name,
        recompute=recompute,
        ranks=GPUS,
        nodes=nodes,
        values=values,
        peak=peaks[peak_rank],
        peak_rank=peak_rank,
        capacity=memory.capacity_bytes,
        communication_bytes=communication,
        seconds=round(time.time() - started, 1),
    )


def program_peaks(nodes, values) -> dict[int, int]:
    """Each rank's peak when its nodes run one at a time in id order (D11).

    A value is held from the start of its producer (from the start of the step for an
    input) to the end of its last consumer (to the end of the step when persistent).
    """
    order = {node.id: index for index, node in enumerate(nodes)}
    events: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for value in values:
        start = -1 if value.producer < 0 else order[value.producer]
        ends = [order[consumer] for consumer in value.consumers]
        end = len(nodes) if value.persistent else max(ends, default=start)
        end = max(end, start)
        events[value.rank].append((start, value.bytes))
        events[value.rank].append((end, -value.bytes))
    peaks = {}
    for rank, items in events.items():
        items.sort(key=lambda item: (item[0], item[1] < 0))
        held = peak = 0
        for _, change in items:
            held += change
            peak = max(peak, held)
        peaks[rank] = peak
    return peaks


@dataclass(eq=False)
class Inst:
    """A tensor of the step on some ranks: where each rank's bytes are and which node wrote them."""

    type: types.Type
    storage: dict[int, int]
    producers: dict[int, int]


@dataclass(eq=False)
class Inside:
    """A value a kernel computes inside; it becomes a fresh result of the kernel's node.

    A view of one has the other's storage: *root* is the Inside it views.
    """

    type: types.Type
    root: "Inside | None" = None


FREE = object()
"""A value that holds no bytes: a constant, an index, a shape."""


@dataclass
class Frame:
    module: core.Module
    ranks: tuple[int, ...]
    path: tuple[str, ...]
    phase: str
    backward: bool
    inside: bool = False


@dataclass
class Builder:
    rank: int
    name: str
    bytes: int
    producer: int
    consumers: list[int] = field(default_factory=list)
    persistent: bool = False


class Reader:
    """An abstract run of the checked step: nodes instead of numbers."""

    def __init__(self, module: core.Module, config: Config):
        self.module = module
        self.config = config
        self.owners = TARGET.storage_owners
        self.nodes: list[Node] = []
        self.values: list[Builder] = []
        self.names: Counter = Counter()
        self.tags = itertools.count(1)
        self.cache: dict[tuple, object] = {}
        self.kept: list[object] = []

    # ---------------------------------------------------------------- entry
    def step(self, function: core.Function):
        env = {}
        for param in function.params:
            env[id(param)] = self.input(param)
        frame = Frame(self.module, tuple(range(GPUS)), (), "", False)
        return self.evaluate(function.body, env, frame)

    def input(self, param: expr.Var):
        if isinstance(param.type, types.TupleType):
            raise ValueError(f"step input {param.name!r} is a tuple")
        if not isinstance(param.type, types.TensorType):
            return FREE
        inst = Inst(param.type, {}, {})
        for rank in self.units_of(param.type):
            inst.storage[rank] = self.value(
                rank, param.name, self.local_bytes(param.type), -1, persistent=True
            )
        return inst

    def finish(self, result):
        for leaf in leaves(result):
            if isinstance(leaf, Inst):
                for vid in leaf.storage.values():
                    self.values[vid].persistent = True
        values = tuple(
            Value(
                i,
                item.rank,
                item.name,
                item.bytes,
                item.producer,
                tuple(item.consumers),
                item.persistent,
            )
            for i, item in enumerate(self.values)
        )
        return tuple(self.nodes), values

    # ------------------------------------------------------------ evaluation
    def evaluate(self, root: expr.Expr, env: dict, frame: Frame):
        """The abstract value of *root*: operands before consumers, regions entered on their own."""
        return self.evaluate_all((root,), env, frame)[0]

    def evaluate_all(self, roots: tuple, env: dict, frame: Frame) -> tuple:
        """The abstract values of *roots*, sharing one evaluation of what they share."""
        memo: dict[int, object] = {}
        stack = [(root, False) for root in reversed(roots)]
        while stack:
            node, ready = stack.pop()
            if id(node) in memo:
                continue
            if isinstance(node, expr.Var):
                if id(node) not in env:
                    raise ValueError(f"{location(node)}unbound variable {node.name!r}")
                memo[id(node)] = env[id(node)]
                continue
            operands = children(node)
            if not ready:
                stack.append((node, True))
                stack.extend(
                    (child, False) for child in reversed(operands) if id(child) not in memo
                )
                continue
            args = tuple(memo[id(child)] for child in operands)
            memo[id(node)] = self.visit(node, args, frame)
        return tuple(memo[id(root)] for root in roots)

    def visit(self, node: expr.Expr, args: tuple, frame: Frame):
        if isinstance(node, (expr.Constant, core.ShapeOf)):
            return FREE
        if isinstance(node, expr.Tuple):
            return args
        if isinstance(node, core.MeshRegion):
            env = {id(param): arg for param, arg in zip(node.params, args, strict=True)}
            ranks = frame.ranks if frame.inside else units(node.mesh)
            return self.evaluate(node.body, env, Frame(**{**frame.__dict__, "ranks": ranks}))
        if isinstance(node, core.LoopRegion):
            return self.loop(node, args, frame)
        if not isinstance(node, expr.Call):
            raise ValueError(f"{location(node)}cannot read {type(node).__name__}")
        if isinstance(node.target, core.FunctionRef):
            return self.call(node, args, frame)
        if isinstance(node.target, TupleGetItem):
            return args[0][int(node.args[1].value)]
        if frame.inside:
            position = self.alias(node)
            if position is not None and isinstance(args[position], Inst):
                return view(args[position], node.type)
            if position is not None and isinstance(args[position], Inside):
                return Inside(node.type, args[position].root or args[position])
            return inside(node.type)
        if isinstance(node.target, Reshard):
            return self.reshard(node, args[0], frame)
        return self.op(node, args, frame)

    def loop(self, node: core.LoopRegion, args: tuple, frame: Frame):
        if frame.inside:
            return inside(node.type)
        trips = range(int(node.start), int(node.extent), int(node.step))
        if not trips:
            raise ValueError(f"{location(node)}loop has no iterations")
        carried = len(node.yield_values)
        invariant = {
            id(param): arg for param, arg in zip(node.params[carried:], args[carried:], strict=True)
        }
        state = list(args[:carried])
        last = None
        count = len(trips)
        for ordinal, _ in enumerate(trips):
            index = count - 1 - ordinal if frame.backward else ordinal
            env = {**invariant, id(node.induction_var): FREE}
            env.update(
                {
                    id(param): value
                    for param, value in zip(node.params[:carried], state, strict=True)
                }
            )
            label = "mb" if not frame.path else "layer"
            inner = Frame(**{**frame.__dict__, "path": frame.path + (f"{label}{index}",)})
            if carried:
                state = list(self.evaluate_all(tuple(node.yield_values), env, inner))
            else:
                last = self.evaluate(node.body, env, inner)
        if not carried:
            return last
        return state[0] if carried == 1 else tuple(state)

    # ----------------------------------------------------------------- calls
    def call(self, node: expr.Call, args: tuple, frame: Frame):
        ref = node.target
        module, function = core.resolve(frame.module, ref)
        given = ref.arguments(args)
        supplied = [param for param in function.params if not (ref.name is None and param.is_const)]
        if len(supplied) != len(given):
            raise ValueError(
                f"{location(node)}call of {function.name!r} gives {len(given)} of {len(supplied)} args"
            )
        env = {id(param): arg for param, arg in zip(supplied, given, strict=True)}
        kernel = kernel_of(module.name)
        if frame.inside:
            return self.evaluate(function.body, env, Frame(**{**frame.__dict__, "module": module}))
        if kernel is None:
            segment, phase, backward = stage_segment(module.name, frame)
            inner = Frame(module, frame.ranks, frame.path + segment, phase, backward)
            return self.evaluate(function.body, env, inner)
        phase = (
            "b"
            if module.name.endswith("Backward")
            else ("f" if module.name.endswith("Forward") else frame.phase)
        )
        shape = self.evaluate(
            function.body, env, Frame(module, frame.ranks, frame.path, phase, False, inside=True)
        )
        flops = flops_of(node)
        name = frame.path + (kernel + ("" if phase != "b" else "_b"),)
        inputs = [leaf for leaf in leaves(given) if isinstance(leaf, Inst)]
        made: dict[int, Inst] = {}
        outputs = [leaf for leaf in leaves(shape) if isinstance(leaf, Inside)]
        for leaf in outputs:
            root = leaf.root or leaf
            made.setdefault(id(root), Inst(root.type, {}, {}))
        fresh = list(made.values())
        for rank in frame.ranks:
            moved = sum(self.local_bytes(inst.type) for inst in inputs + fresh)
            nid = self.node(
                rank, name, "compute", kernel, phase, flops, moved, (), self.deps(inputs, rank)
            )
            self.consume(inputs, rank, nid)
            for inst in fresh:
                inst.producers[rank] = nid
                inst.storage[rank] = self.value(
                    rank, "/".join(name), self.local_bytes(inst.type), nid
                )
        return replace_leaves(shape, made)

    # ------------------------------------------------------------------- ops
    def op(self, node: expr.Call, args: tuple, frame: Frame):
        tensors = [leaf for leaf in leaves(args) if isinstance(leaf, Inst)]
        if not isinstance(node.type, types.TensorType):
            if tensors:
                raise ValueError(
                    f"{location(node)}an Op outside kernels gives a {type(node.type).__name__}"
                )
            return FREE
        size = self.local_bytes(node.type)
        if not tensors and size <= SMALL:
            return FREE
        position = self.alias(node)
        if position is not None and isinstance(args[position], Inst):
            return view(args[position], node.type)
        name = frame.path + (expr.binding_name(node) or type(node.target).__name__.lower(),)
        flops = flops_of(node)
        made = Inst(node.type, {}, {})
        for rank in frame.ranks:
            moved = sum(self.local_bytes(inst.type) for inst in tensors) + size
            nid = self.node(
                rank,
                name,
                "compute",
                type(node.target).__name__,
                frame.phase,
                flops,
                moved,
                (),
                self.deps(tensors, rank),
            )
            self.consume(tensors, rank, nid)
            made.producers[rank] = nid
            made.storage[rank] = self.value(rank, "/".join(name), size, nid)
        return made

    def reshard(self, node: expr.Call, source, frame: Frame):
        if not isinstance(source, Inst):
            return FREE
        name = frame.path + (expr.binding_name(node) or "reshard",)
        before, after = self.units_of(source.type), self.units_of(node.type)
        if before != after:
            return self.transfer(node, source, before, after, name, frame)
        states = changes(source.type, node.type, self.owners)
        kinds = {}
        for axis, (old, new) in states.items():
            kind = collective_kind(old, new)
            if kind is None:
                raise ValueError(
                    f"{location(node)}reshard from {type(old).__name__} to {type(new).__name__} "
                    f"on axis {AXES[axis]!r} has no collective kind"
                )
            if kind != "local":
                kinds.setdefault(kind, []).append(axis)
        kinds = {
            kind: axes
            for kind, axes in kinds.items()
            if prod(self.config.size_of(AXES[a]) for a in axes) > 1
        }
        if not kinds:
            position = self.alias(node)
            if position is not None:
                return view(source, node.type)
            return self.op(node, (source,), frame)
        if set(kinds) == {"reduce_scatter", "all_reduce"}:
            scattered = self.collective(
                "reduce_scatter", kinds["reduce_scatter"], source, node.type, name, frame
            )
            return self.collective(
                "all_reduce", kinds["all_reduce"], scattered, node.type, name, frame
            )
        if len(kinds) > 1:
            named = ", ".join(
                f"{kind} on {tuple(AXES[a] for a in axes)}" for kind, axes in kinds.items()
            )
            raise ValueError(f"{location(node)}reshard mixes {named}; write one reshard for each")
        ((kind, axes),) = kinds.items()
        return self.collective(kind, axes, source, node.type, name, frame)

    def collective(
        self, kind: str, axes, source: Inst, result: types.Type, name, frame: Frame
    ) -> Inst:
        made = Inst(result, {}, {})
        ranks = sorted(source.storage)
        for rank in ranks:
            group = tuple(peer for peer in ranks if same_except(peer, rank, axes, self.config))
            moved = self.local_bytes(result if kind == "reduce_scatter" else source.type)
            nid = self.node(
                rank,
                name + (kind,),
                kind,
                "",
                frame.phase,
                0,
                moved,
                group,
                self.deps([source], rank),
            )
            self.consume([source], rank, nid)
            made.producers[rank] = nid
            made.storage[rank] = self.value(rank, "/".join(name), self.local_bytes(result), nid)
        return made

    def transfer(self, node, source: Inst, before, after, name, frame: Frame) -> Inst:
        """A move between unit sets: kept where the unit already holds its piece, else sent point to point."""
        states = changes(source.type, node.type, self.owners)
        if states:
            moved = {
                AXES[a]: f"{type(old).__name__} -> {type(new).__name__}"
                for a, (old, new) in states.items()
            }
            raise ValueError(
                f"{location(node)}reshard changes the units and the axes {moved}; write one reshard for each"
            )
        split = split_axes_of(source.type, self.owners)
        made = Inst(node.type, {}, {})
        size = self.local_bytes(node.type)
        for rank in after:
            if rank in before:
                made.storage[rank] = source.storage[rank]
                if rank in source.producers:
                    made.producers[rank] = source.producers[rank]
                continue
            sender = next(
                (
                    peer
                    for peer in before
                    if all(
                        coords(peer, self.config)[a] == coords(rank, self.config)[a] for a in split
                    )
                ),
                None,
            )
            if sender is None:
                raise ValueError(
                    f"{location(node)}no unit of {sorted(before)} holds the piece unit {rank} needs"
                )
            tag = next(self.tags)
            sent = self.node(
                sender,
                name + (f"send{rank}",),
                "send",
                "",
                frame.phase,
                0,
                size,
                (sender, rank),
                self.deps([source], sender),
                tag,
            )
            self.consume([source], sender, sent)
            got = self.node(
                rank,
                name + (f"recv{sender}",),
                "recv",
                "",
                frame.phase,
                0,
                size,
                (sender, rank),
                (),
                tag,
            )
            made.producers[rank] = got
            made.storage[rank] = self.value(rank, "/".join(name), size, got)
        return made

    # -------------------------------------------------------------- records
    def node(self, rank, name, kind, op, phase, flops, moved, group, deps, tag=0) -> int:
        text = "/".join(name)
        self.names[(rank, text)] += 1
        if self.names[(rank, text)] > 1:
            text = f"{text}#{self.names[(rank, text)]}"
        nid = len(self.nodes)
        self.nodes.append(
            Node(
                nid,
                rank,
                text,
                kind,
                op,
                phase,
                int(flops),
                int(moved),
                tuple(group),
                tuple(deps),
                tag,
            )
        )
        return nid

    def value(self, rank, name, size, producer, *, persistent=False) -> int:
        self.values.append(Builder(rank, name, size, producer, persistent=persistent))
        return len(self.values) - 1

    def deps(self, inputs, rank) -> tuple[int, ...]:
        return tuple(sorted({inst.producers[rank] for inst in inputs if rank in inst.producers}))

    def consume(self, inputs, rank, nid) -> None:
        for vid in {inst.storage[rank] for inst in inputs if rank in inst.storage}:
            self.values[vid].consumers.append(nid)

    def cached(self, kind: str, key: object, compute):
        """*compute()* once per IR object *key*; the object is kept so its id stays its own."""
        slot = (kind, id(key))
        if slot not in self.cache:
            self.kept.append(key)
            self.cache[slot] = compute()
        return self.cache[slot]

    def local_bytes(self, type_: types.Type) -> int:
        return self.cached(
            "bytes",
            type_,
            lambda: utils.tensor_bytes(
                utils.local_type_of(type_, topology="gpu", topologies=TOPOLOGIES)
            ),
        )

    def alias(self, node: expr.Call):
        return self.cached("alias", node, lambda: aliased_operand(node, self.owners))

    def units_of(self, type_):
        return self.cached("units", type_, lambda: units_of_type(type_))


# ------------------------------------------------------------------ helpers
def children(node: expr.Expr) -> tuple[expr.Expr, ...]:
    if isinstance(node, expr.Call):
        return tuple(node.args)
    if isinstance(node, expr.Tuple):
        return tuple(node.elements)
    if isinstance(node, (core.MeshRegion, core.LoopRegion)):
        return tuple(node.args)
    return ()


def leaves(value):
    if isinstance(value, tuple):
        for item in value:
            yield from leaves(item)
    else:
        yield value


def replace_leaves(value, made: dict[int, Inst]):
    if isinstance(value, tuple):
        return tuple(replace_leaves(item, made) for item in value)
    if isinstance(value, Inside):
        root = made[id(value.root or value)]
        return root if value.root is None else view(root, value.type)
    return value


def inside(type_: types.Type):
    if isinstance(type_, types.TupleType):
        return tuple(inside(item) for item in type_.fields)
    if isinstance(type_, types.TensorType):
        return Inside(type_)
    return FREE


def view(source: Inst, type_: types.Type) -> Inst:
    return Inst(type_, source.storage, source.producers)


def kernel_of(name: str) -> str | None:
    base = name.removesuffix("Forward").removesuffix("Backward").rstrip("0123456789")
    for kernel in KERNELS:
        if base == kernel:
            return kernel
        if base.startswith(kernel) and base[len(kernel) :].startswith("W"):
            return base
    return None


def stage_segment(name: str, frame: Frame) -> tuple[tuple[str, ...], str, bool]:
    """The name segment, phase and direction a call of Module *name* adds."""
    for suffix, phase in (("Forward", "f"), ("Backward", "b")):
        if name.endswith(suffix):
            base = name.removesuffix(suffix)
            if base.startswith("Stage"):
                return (f"stage{base.removeprefix('Stage')}", phase), phase, phase == "b"
            return (), phase, phase == "b"
    if name.startswith("Muon"):
        return (name[:-1].lower(),), "", False
    return (), frame.phase, frame.backward


def flops_of(node: expr.Call) -> int:
    cost = expr.get_metadata(node, analysis.ComputeCostMetadata)
    if cost is None:
        return 0
    return sum(
        spread.per_unit[0] if spread.per_unit else spread.total for _, spread in cost.flops.kinds
    )


def units(mesh: types.Mesh) -> tuple[int, ...]:
    run = utils.runs(mesh, ("gpu",))["gpu"]
    ids = utils.unit_ids(run)
    if ids is None:
        raise ValueError(f"mesh {mesh} selects symbolic units")
    return tuple(sorted(ids))


def units_of_type(type_: types.Type) -> tuple[int, ...]:
    layout = types.shard_layout_of(type_.layout)
    if layout is None:
        return tuple(range(GPUS))
    return units(layout.mesh)


def states(type_: types.TensorType, owners) -> dict[int, object]:
    """Each mesh axis's state; a Split names the logical tensor axis it divides, not a layout mode."""
    layout = types.shard_layout_of(type_.layout)
    logical = types.split_target_axes(layout, type_.shape) if layout is not None else ()
    out = {}
    for key, attr in utils.attrs_by_axis(type_, owners=owners).items():
        out[key[1]] = types.Split(logical[key[1]]) if isinstance(attr, types.Split) else attr
    return out


def changes(source: types.TensorType, result: types.TensorType, owners) -> dict[int, tuple]:
    before, after = states(source, owners), states(result, owners)
    found = {}
    for axis in sorted(before.keys() | after.keys()):
        old, new = before.get(axis, types.Broadcast()), after.get(axis, types.Broadcast())
        if old != new:
            found[axis] = (old, new)
    return found


def split_axes_of(type_: types.TensorType, owners) -> tuple[int, ...]:
    return tuple(
        key[1]
        for key, attr in utils.attrs_by_axis(type_, owners=owners).items()
        if isinstance(attr, types.Split)
    )


def collective_kind(old, new) -> str | None:
    split, broadcast, partial = types.Split, types.Broadcast, types.Partial
    if isinstance(old, split) and isinstance(new, broadcast):
        return "all_gather"
    if isinstance(old, partial) and isinstance(new, split):
        return "reduce_scatter"
    if isinstance(old, partial) and isinstance(new, broadcast):
        return "all_reduce"
    if isinstance(old, split) and isinstance(new, split):
        return "all_to_all"
    if isinstance(old, broadcast) and isinstance(new, (split, partial)):
        return "local"
    return None


def coords(rank: int, config: Config) -> tuple[int, ...]:
    sizes = tuple(config.size_of(axis) for axis in AXES)
    out = []
    for size in reversed(sizes):
        rank, rest = divmod(rank, size)
        out.append(rest)
    return tuple(reversed(out))


def same_except(peer: int, rank: int, axes, config: Config) -> bool:
    a, b = coords(peer, config), coords(rank, config)
    return all(a[i] == b[i] for i in range(len(AXES)) if i not in axes)


def prod(values) -> int:
    return math.prod(values)


def location(node: expr.Expr) -> str:
    suffix = expr.diagnostic_suffix(node)
    return f"{suffix.strip()}: " if suffix else ""


def summary(result: Facts) -> dict:
    """Counts for the gate: nodes by kind, values, edges, and the peak."""
    kinds = Counter(node.kind for node in result.nodes)
    return {
        "config": result.config,
        "recompute": result.recompute,
        "nodes": len(result.nodes),
        "edges": sum(len(node.deps) for node in result.nodes),
        "values": len(result.values),
        "kinds": {kind: kinds.get(kind, 0) for kind in KINDS},
        "peak": result.peak,
        "peak_rank": result.peak_rank,
        "capacity": result.capacity,
        "communication_bytes": result.communication_bytes,
        "seconds": result.seconds,
    }


__all__ = ["KERNELS", "Facts", "Node", "Value", "facts", "program_peaks", "summary"]
