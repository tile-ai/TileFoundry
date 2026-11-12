"""Check host communication ordering with the shared physical lifetime checker."""

from dataclasses import dataclass, replace
from math import prod

from tilefoundry.ir import core, expr, types
from tilefoundry.ir.tir import communication as comm

from .buffer_lifetime import BufferAccess, BufferHazard, BufferLifetime, check_buffer_lifetimes

__all__ = ["BufferRegion", "CommunicationReport", "check_communication", "host_statements"]


@dataclass(frozen=True)
class BufferRegion:
    """A parameter's byte range in a caller-declared physical allocation."""

    allocation: str
    offset: int = 0


@dataclass(frozen=True)
class CommunicationReport:
    """Ordering proof, physical hazards, and obligations at an asynchronous return."""

    lifetimes: tuple[BufferLifetime, ...]
    predecessors: dict[str, frozenset[str]]
    hazards: tuple[BufferHazard, ...]
    exported: tuple[str, ...]


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


def check_communication(function, regions=None) -> CommunicationReport:
    """Prove stream/event ordering and buffer reuse, including repeated protocols.

    Parameters are disjoint unless regions declares aliases. All device work must
    complete or be covered by an exported event; exporting obligates the caller to
    retain buffers. Unknown host effects are refused, never assumed synchronous.
    """
    regions = regions or {}
    statements = tuple(host_statements(function.body))
    parameters = {id(value): value for value in function.params}
    names = [value.name for value in function.params]
    if len(set(names)) != len(names) or set(regions) - set(names):
        raise ValueError("communication parameters must be unique and regions must name parameters")
    resources = [(value.type.kind, value.type.name) for value in function.params
                 if isinstance(value.type, types.CommunicationHandleType)]
    if len(set(resources)) != len(resources):
        raise ValueError("resource identities must have exactly one parameter")
    bounds, buffers = {}, {}
    for value in function.params:
        if not isinstance(value.type, types.TensorType):
            continue
        region = regions.get(value.name, BufferRegion(value.name))
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

    def issue(items, grouped=False):
        incoming = set().union(*(frontier(stmt.args[3].name) for _, stmt in items))
        finished = set()
        for index, stmt in items:
            stream = stmt.args[3].name
            parents = incoming if grouped else frontier(stream)
            read = node(f"comm.{index}.read", parents)
            write = node(f"comm.{index}.write", {read})
            ordinal = len(predecessors)
            reader = BufferAccess(read, ordinal, ordinal + 1)
            writer = BufferAccess(write, ordinal + 1, ordinal + 2)
            for source in (stmt.args[0], *stmt.args[4:]):
                for segment in touched(source):
                    position = current[segment]
                    values[position] = replace(values[position], consumers=(*values[position].consumers, reader))
            for segment in touched(stmt.args[1]):
                allocation, start, end = segment
                current[segment] = len(values)
                values.append(BufferLifetime(f"comm.{index}.{len(values)}", allocation, start, end - start, writer))
            streams[stream] = {write}
            finished.add(write)
            jobs.add(write)
        if grouped:
            for _, stmt in items:
                streams[stmt.args[3].name] = set(finished)

    for index, stmt in enumerate(statements):
        if any(id(arg) not in parameters for arg in stmt.args):
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
    return CommunicationReport(lifetimes, predecessors, check_buffer_lifetimes(lifetimes, predecessors=predecessors), tuple(exported))
