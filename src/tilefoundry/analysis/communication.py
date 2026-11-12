"""Check host communication ordering with the shared physical lifetime checker."""

from dataclasses import dataclass, replace
from itertools import combinations
from math import prod

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
    accesses: tuple[tuple[str, expr.MemoryEffect], ...]


@dataclass(frozen=True)
class CommunicationReport:
    """Ordering proof, physical hazards, and obligations at an asynchronous return."""

    lifetimes: tuple[BufferLifetime, ...]
    predecessors: dict[str, frozenset[str]]
    hazards: tuple[BufferHazard, ...]
    exported: tuple[str, ...]
    operations: tuple[Operation, ...] = ()


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


def check_communication(function, regions=None, *, module=None) -> CommunicationReport:
    """Prove stream/event ordering and buffer reuse, including repeated protocols.

    Parameters are disjoint unless regions declares aliases. All device work must
    complete or be covered by an exported event; exporting obligates the caller to
    retain buffers. Unknown host effects are refused, never assumed synchronous.
    """
    regions = buffer_regions(function) if regions is None else regions
    statements = tuple(host_statements(function.body))
    parameters = {id(value): value for value in function.params}
    names = [value.name for value in function.params]
    tensor_names = {value.name for value in function.params if isinstance(value.type, types.TensorType)}
    if len(set(names)) != len(names) or set(regions) - tensor_names:
        raise ValueError("communication parameters must be unique and regions must name tensor parameters")
    resources = [(value.type.kind, value.type.name) for value in function.params
                 if isinstance(value.type, types.CommunicationHandleType)]
    if len(set(resources)) != len(resources):
        raise ValueError("resource identities must have exactly one parameter")
    bounds, buffers = {}, {}
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
        buffers[id(value)] = (region.allocation, region.offset, region.offset + size)
        bounds.setdefault(region.allocation, set()).update((region.offset, region.offset + size))
    segments = tuple((allocation, a, b) for allocation, points in bounds.items()
                     for a, b in zip(sorted(points), sorted(points)[1:]) if a != b)
    initial = BufferAccess("entry", 0, 0)
    values = [BufferLifetime(f"input.{i}", allocation, a, b - a, initial)
              for i, (allocation, a, b) in enumerate(segments)]
    current = dict(zip(segments, range(len(values))))
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

    def touched(value):
        allocation, start, end = buffers[id(value)]
        return tuple(segment for segment in segments
                     if segment[0] == allocation and start <= segment[1] and segment[2] <= end)

    def descriptor(index, stmt):
        if isinstance(stmt.callable, comm.CommEnqueue):
            accesses = [(stmt.args[0], expr.MemoryEffect.READ), (stmt.args[1], expr.MemoryEffect.WRITE)]
            accesses.extend((arg, expr.MemoryEffect.READ) for arg in stmt.args[4:])
            return f"comm.{index}", stmt.args[3], tuple(accesses)
        if module is None:
            raise ValueError("kernel completion checking requires the owning module")
        stream = stmt.callable.stream
        if not isinstance(stream, expr.Var) or id(stream) not in parameters:
            raise ValueError("mixed protocols require an explicit declared launch stream")
        comm.handle_type(stream, "stream")
        ref = stmt.args[0]
        if not isinstance(ref, core.SymbolRef):
            raise ValueError("kernel launch requires a symbolic callee")
        callee = core.resolve_symbol(module, ref.name, nested=ref.nested)
        if (not isinstance(callee, core.PrimFunction) or isinstance(callee.target, cpu.Target)
                or len(stmt.args[7:]) != len(callee.params)):
            raise ValueError("kernel launch requires a concrete kernel and matching operands")
        if any(arg.type != param.type for arg, param in zip(stmt.args[7:], callee.params, strict=True)):
            raise ValueError("kernel launch operand types must match its parameter types")
        summary = summarize(callee)
        forwarded = [(stmt.args[7 + i], mode) for i, mode in summary.accesses]
        for (a, left), (b, right) in combinations(forwarded, 2):
            if a is b and (left | right) & expr.MemoryEffect.WRITE:
                raise ValueError("kernel launch aliases distinct parameters with a write")
        return f"kernel.{index}", stream, tuple(forwarded)

    def issue(items, grouped=False):
        descriptions = tuple((stmt, descriptor(index, stmt)) for index, stmt in items)
        incoming = set().union(*(frontier(stream.name) for _, (_, stream, _) in descriptions))
        finished = set()
        for stmt, (name, handle, accesses) in descriptions:
            for (a, left), (b, right) in combinations(accesses, 2):
                if a is b or not (left | right) & expr.MemoryEffect.WRITE:
                    continue
                allocation, start, end = buffers[id(a)]
                other, low, high = buffers[id(b)]
                if allocation == other and max(start, low) < min(end, high):
                    operand_hazards.append(BufferHazard("overlapping_operands", allocation, (a.name, b.name), (name,)))
            stream = handle.name
            parents = incoming if grouped else frontier(stream)
            read = node(f"{name}.read", parents)
            write = node(f"{name}.write", {read})
            ordinal = len(predecessors)
            reader = BufferAccess(read, ordinal, ordinal + 1)
            writer = BufferAccess(write, ordinal + 1, ordinal + 2)
            for source, mode in accesses:
                if not mode & expr.MemoryEffect.READ:
                    continue
                for segment in touched(source):
                    position = current[segment]
                    values[position] = replace(values[position], consumers=(*values[position].consumers, reader))
            for destination, mode in accesses:
                if not mode & expr.MemoryEffect.WRITE:
                    continue
                for segment in touched(destination):
                    allocation, start, end = segment
                    current[segment] = len(values)
                    values.append(BufferLifetime(f"{name}.{len(values)}", allocation, start, end - start, writer))
            operations.append(Operation(name, stmt, stream, read, write, tuple((value.name, mode) for value, mode in accesses)))
            streams[stream] = {write}
            finished.add(write)
            jobs.add(write)
        if grouped:
            for _, (_, stream, _) in descriptions:
                streams[stream.name] = set(finished)

    for index, stmt in enumerate(statements):
        arguments = stmt.args[7:] if isinstance(stmt.callable, launch.Launch) else stmt.args
        if any(id(arg) not in parameters for arg in arguments):
            raise ValueError("communication operands must be declared host parameters")
        op = stmt.callable
        if isinstance(op, comm.CommEnqueue):
            comm.check_enqueue(op, stmt.args)
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
    lifetimes = tuple(values)
    hazards = (*operand_hazards, *check_buffer_lifetimes(lifetimes, predecessors=predecessors))
    return CommunicationReport(lifetimes, predecessors, hazards, tuple(exported), tuple(operations))
