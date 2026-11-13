"""Check host communication ordering with the shared physical lifetime checker."""

from dataclasses import dataclass, replace
from itertools import combinations
from math import prod

import isl

from tilefoundry.interfaces import access, execution
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.tir import communication as comm
from tilefoundry.ir.tir import launch
from tilefoundry.target import cpu

from .buffer_lifetime import BufferAccess, BufferHazard, BufferLifetime, check_buffer_lifetimes
from .kernel_effects import summarize

__all__ = ["BufferBindings", "BufferRegion", "CommunicationReport", "Operation", "buffer_regions", "check_communication", "host_statements"]


@dataclass(frozen=True)
class BufferRegion:
    """A parameter's byte range in a caller-declared physical allocation."""

    allocation: str
    offset: int = 0


@dataclass(frozen=True)
class BufferBindings(expr.IRMetadata):
    """Selected host parameter aliases checked again against native addresses."""

    regions: tuple[tuple[str, BufferRegion], ...]
    uninitialized: tuple[str, ...] = ()


def buffer_regions(function):
    """The explicit physical bindings selected for one host entry."""
    bindings = expr.get_metadata(function, BufferBindings)
    if bindings is None:
        return {}
    regions = dict(bindings.regions)
    if len(regions) != len(bindings.regions):
        raise ValueError("buffer bindings must name each parameter only once")
    return regions


@dataclass(frozen=True)
class Operation:
    """One atomic execution occurrence and its dependency endpoints."""

    name: str
    statement: core.Evaluate
    stream: str
    read: str
    complete: str
    effects: tuple[execution.Effect[str], ...]

    @property
    def accesses(self):
        """The whole-parameter projection used by existing report consumers."""
        modes = {}
        for effect in self.effects:
            modes[effect.parameter] = modes.get(effect.parameter, expr.MemoryEffect(0)) | effect.mode
        return tuple(modes.items())


@dataclass(frozen=True)
class CommunicationReport:
    """Ordering proof, physical hazards, and obligations at an asynchronous return."""

    lifetimes: tuple[BufferLifetime, ...]
    predecessors: dict[str, frozenset[str]]
    hazards: tuple[BufferHazard, ...]
    exported: tuple[str, ...]
    operations: tuple[Operation, ...] = ()
    live_out: tuple[str, ...] = ()
    exported_frontiers: tuple[tuple[str, frozenset[str]], ...] = ()


def host_statements(body):
    """Flatten sequences and static repeated host protocols without erasing carries.

    Communication handles and buffers must be fixed parameters. A loop induction
    value cannot be an operand; dynamic host control flow needs an explicit proof.
    """
    if isinstance(body, core.Sequential):
        for statement in body.body:
            yield from host_statements(statement)
    elif isinstance(body, core.For):
        bounds = tuple(expr.static_dim_value(value) for value in (body.start, body.stop, body.step))
        if any(value is None for value in bounds) or bounds[2] == 0:
            raise ValueError("communication host loops require static nonzero-step bounds")
        for _ in range(*bounds):
            yield from host_statements(body.body)
    elif isinstance(body, core.Evaluate):
        yield body
    else:
        raise ValueError(f"unsupported communication host control flow: {type(body).__name__}")


def check_communication(function, regions=None, *, module=None, uninitialized=None) -> CommunicationReport:
    """Prove stream/event ordering and buffer reuse, including repeated protocols.

    Parameters are disjoint unless regions declares aliases. All device work must
    complete or be covered by an exported event; exporting obligates the caller to
    retain buffers. Unknown host effects are refused, never assumed synchronous.
    """
    regions = buffer_regions(function) if regions is None else regions
    if uninitialized is None:
        bindings = expr.get_metadata(function, BufferBindings)
        uninitialized = () if bindings is None else bindings.uninitialized
    statements = tuple(host_statements(function.body))
    parameters = {id(value): value for value in function.params}
    names = [value.name for value in function.params]
    tensor_names = {value.name for value in function.params if isinstance(value.type, types.TensorType)}
    if len(set(names)) != len(names) or set(regions) - tensor_names:
        raise ValueError("communication parameters must be unique and regions must name tensor parameters")
    if set(uninitialized) - tensor_names:
        raise ValueError("uninitialized buffers must name tensor parameters")
    resources = [(value.type.kind, value.type.name) for value in function.params
                 if isinstance(value.type, types.CommunicationHandleType)]
    if len(set(resources)) != len(resources):
        raise ValueError("resource identities must have exactly one parameter")
    buffers = {}
    for value in function.params:
        if not isinstance(value.type, types.TensorType):
            continue
        region = regions.get(value.name, BufferRegion(value.name))
        if not expr.is_literal_shape(value.type.shape) or value.type.dtype.bit_width % 8:
            raise ValueError("execution buffers require static shapes and byte-addressable elements")
        layout = value.type.layout.layout if isinstance(value.type.layout, types.ShardLayout) else value.type.layout
        if layout != types.Layout(value.type.shape):
            raise ValueError("execution buffers require compact parameter layouts")
        size = prod(value.type.shape) * value.type.dtype.bit_width // 8
        if not isinstance(region.offset, int) or isinstance(region.offset, bool) or region.offset < 0 or not region.allocation:
            raise ValueError("buffer regions require an allocation and nonnegative byte offset")
        buffers[value.name] = (region.allocation, region.offset, size)
    predecessors = {"entry": frozenset()}
    streams, events = {}, {}
    operations, operand_hazards = [], []
    completed, exported, jobs = {"entry"}, {}, set()
    depth, pending = 0, []

    def node(name, parents):
        predecessors[name] = frozenset(parents).union(*(predecessors[p] for p in parents))
        return name

    def frontier(stream):
        return streams.get(stream, {"entry"}) | completed

    def physical(effect):
        _, offset, _ = buffers[effect.parameter]
        return effect.region.relation.range().apply(isl.map(f"{{ [b] -> [c] : c = b + {offset} }}"))

    summaries = {}

    def descriptor(index, stmt):
        if isinstance(stmt.callable, comm.CommEnqueue):
            semantic = comm.check_enqueue(stmt.callable, stmt.args)
            accesses = [(stmt.args[0], expr.MemoryEffect.READ), (stmt.args[1], expr.MemoryEffect.WRITE)]
            accesses.extend((arg, expr.MemoryEffect.READ) for arg in stmt.args[4:])
            result = types.shard_layout_of(semantic.type.layout)
            recipients = () if result is None else types.Placement.from_mesh(result.mesh)
            definite = all(member in recipients for member in types.Placement.from_mesh(semantic.target.participants))
            return f"comm.{index}", stmt.args[3], tuple(execution.Effect(
                value.name, replace(access.byte_relation(value.type), may_access=mode == expr.MemoryEffect.WRITE and not definite), mode
            ) for value, mode in accesses)
        if module is None:
            raise ValueError("kernel completion checking requires the owning module")
        stream = stmt.callable.stream
        if not isinstance(stream, expr.Var) or id(stream) not in parameters:
            raise ValueError("mixed protocols require an explicit declared launch stream")
        comm.handle_type(stream, "stream")
        ref = stmt.args[0]
        if not isinstance(ref, core.FunctionRef):
            raise ValueError("kernel launch requires a symbolic callee")
        _, callee = core.resolve(module, ref)
        if (not isinstance(callee, core.PrimFunction) or isinstance(callee.target, cpu.Target)
                or len(stmt.args[7:]) != len(callee.params)):
            raise ValueError("kernel launch requires a concrete kernel and matching operands")
        if any(arg.type != param.type for arg, param in zip(stmt.args[7:], callee.params, strict=True)):
            raise ValueError("kernel launch operand types must match its parameter types")
        if id(callee) not in summaries:
            summaries[id(callee)] = summarize(callee)
        summary = summaries[id(callee)]
        for left, right in combinations(summary.effects, 2):
            if (left.parameter != right.parameter and stmt.args[7 + left.parameter] is stmt.args[7 + right.parameter]
                    and (left.mode | right.mode) & expr.MemoryEffect.WRITE
                    and not left.region.relation.range().intersect(right.region.relation.range()).is_empty()):
                raise ValueError("kernel launch aliases distinct parameters with a write")
        return f"kernel.{index}", stream, tuple(replace(effect, parameter=stmt.args[7 + effect.parameter].name)
                                                for effect in summary.effects)

    descriptions = {}
    cells = {name: (isl.set(f"{{ [b] : 0 <= b < {size} }}"),) if size else () for name, (_, _, size) in buffers.items()}
    for index, stmt in enumerate(statements):
        arguments = stmt.args[7:] if isinstance(stmt.callable, launch.Launch) else stmt.args
        if any(id(arg) not in parameters for arg in arguments):
            raise ValueError("communication operands must be declared host parameters")
        if isinstance(stmt.callable, (comm.CommEnqueue, launch.Launch)):
            descriptions[index] = descriptor(index, stmt)
            for effect in descriptions[index][2]:
                reached = effect.region.relation.range()
                cells[effect.parameter] = tuple(piece for cell in cells[effect.parameter]
                                               for piece in (cell.intersect(reached), cell.subtract(reached)) if not piece.is_empty())
    initial = BufferAccess("entry", 0, 0)
    values, current, initialized = [], {}, {}
    for parameter, parts in cells.items():
        allocation, offset, size = buffers[parameter]
        for index, cell in enumerate(parts):
            current[parameter, index] = len(values)
            initialized[parameter, index] = parameter not in uninitialized
            values.append(BufferLifetime(f"input.{len(values)}", allocation, offset, size, initial, region=cell))

    def touched(effect):
        reached = effect.region.relation.range()
        return tuple((effect.parameter, index) for index, cell in enumerate(cells[effect.parameter]) if cell.is_subset(reached))

    def issue(items, grouped=False):
        described = tuple((stmt, descriptions[index]) for index, stmt in items)
        incoming = set().union(*(frontier(stream.name) for _, (_, stream, _) in described))
        finished = set()
        for stmt, (name, handle, effects) in described:
            for left, right in combinations(effects, 2):
                if left.parameter == right.parameter or not (left.mode | right.mode) & expr.MemoryEffect.WRITE:
                    continue
                allocation, _, _ = buffers[left.parameter]
                other, _, _ = buffers[right.parameter]
                if allocation == other and not physical(left).intersect(physical(right)).is_empty():
                    operand_hazards.append(BufferHazard("overlapping_operands", allocation, (left.parameter, right.parameter), (name,)))
            stream = handle.name
            parents = frontier(stream) | (incoming if grouped else set())
            read = node(f"{name}.read", parents)
            write = node(f"{name}.write", {read})
            ordinal = len(predecessors)
            reader = BufferAccess(read, ordinal, ordinal + 1)
            writer = BufferAccess(write, ordinal + 1, ordinal + 2)
            reads = {cell for effect in effects if effect.mode & expr.MemoryEffect.READ for cell in touched(effect)}
            writes = {}
            for effect in effects:
                if effect.mode & expr.MemoryEffect.WRITE:
                    for cell in touched(effect):
                        writes[cell] = writes.get(cell, False) or not effect.region.may_access
            for cell in reads:
                position = current[cell]
                values[position] = replace(values[position], consumers=(*values[position].consumers, reader))
                if not initialized[cell]:
                    operand_hazards.append(BufferHazard("uninitialized_read", buffers[cell[0]][0], (cell[0],), (name,)))
            for cell, definite in writes.items():
                if not definite and cell not in reads:
                    position = current[cell]
                    values[position] = replace(values[position], consumers=(*values[position].consumers, reader))
                parameter, index = cell
                allocation, offset, size = buffers[parameter]
                current[cell] = len(values)
                initialized[cell] |= definite
                values.append(BufferLifetime(f"{name}.{len(values)}", allocation, offset, size, writer, region=cells[parameter][index]))
            operations.append(Operation(name, stmt, stream, read, write, effects))
            streams[stream] = {write}
            finished.add(write)
            jobs.add(write)
        if grouped:
            for _, (_, stream, _) in described:
                streams[stream.name] = set(finished)

    for index, stmt in enumerate(statements):
        arguments = stmt.args[7:] if isinstance(stmt.callable, launch.Launch) else stmt.args
        if any(id(arg) not in parameters for arg in arguments):
            raise ValueError("communication operands must be declared host parameters")
        op = stmt.callable
        if isinstance(op, comm.CommEnqueue):
            if len(stmt.args) == 5 and depth:
                raise ValueError("runtime counts must be read outside an open communication group")
            if depth:
                pending.append((index, stmt))
            else:
                issue(((index, stmt),))
        elif isinstance(op, launch.Launch):
            if depth:
                raise ValueError("kernel launches are invalid before the outermost group end")
            issue(((index, stmt),))
        elif isinstance(op, comm.CommGroupStart):
            if stmt.args:
                raise ValueError("group start takes no operands")
            depth += 1
        elif isinstance(op, comm.CommGroupEnd):
            if stmt.args or not depth:
                raise ValueError("unmatched group end")
            depth -= 1
            if not depth:
                issue(pending, grouped=True)
                pending = []
        elif type(op) in comm.HANDLES:
            comm.verify_handles(expr.Call(op, stmt.args, type=types.UnitType()), None)
            if depth:
                raise ValueError("stream/event operations are invalid before the outermost group end")
            keys = [arg.name for arg in stmt.args]
            if isinstance(op, comm.EventRecord):
                events[keys[0]] = set(frontier(keys[1]))
            elif isinstance(op, comm.StreamSynchronize):
                completed |= frontier(keys[0])
            else:
                event = keys[-1]
                if event not in events:
                    raise ValueError(f"event {event!r} has not been recorded")
                if isinstance(op, comm.StreamWaitEvent):
                    streams[keys[0]] = frontier(keys[0]) | events[event]
                elif isinstance(op, comm.EventSynchronize):
                    completed |= events[event]
                elif isinstance(op, comm.ExportEvent):
                    exported[event] = set(events[event])
            completed |= set().union(*(predecessors[p] for p in tuple(completed)))
        else:
            raise ValueError(f"unmodeled host effect in communication protocol: {type(op).__name__}")
    if depth:
        raise ValueError("unclosed communication group")
    for event, captured in exported.items():
        if events[event] != captured:
            raise ValueError("an exported event was recorded again")
    covered = completed | set().union(*exported.values())
    covered |= set().union(*(predecessors[p] for p in tuple(covered)))
    if jobs - covered:
        raise ValueError("pending communication at return requires synchronization or an exported completion event")
    outputs = function.params[-function.output_count:] if function.output_count else ()
    if outputs:
        boundary = node("outputs.ready", jobs | completed)
        use = BufferAccess(boundary, len(predecessors), len(predecessors))
        output_names = {parameter.name for parameter in outputs}
        for cell, position in current.items():
            if cell[0] in output_names:
                values[position] = replace(values[position], consumers=(*values[position].consumers, use))
                if not initialized[cell]:
                    operand_hazards.append(BufferHazard("uninitialized_read", buffers[cell[0]][0], (cell[0],), (boundary,)))
    lifetimes = tuple(values)
    hazards = (*operand_hazards, *check_buffer_lifetimes(lifetimes, predecessors=predecessors))
    return CommunicationReport(lifetimes, predecessors, hazards, tuple(exported), tuple(operations),
                               tuple(values[index].value for index in dict.fromkeys(current.values())),
                               tuple((event, frozenset(frontier)) for event, frontier in exported.items()))
