"""Place explicit host workspace using the protocol's completion dependencies."""

from dataclasses import dataclass
from itertools import combinations
from math import prod

from tilefoundry.ir import core, expr, types

from .allocation import AllocationSummaryMetadata, AllocationValue, solve_allocation
from .buffer_lifetime import interfere
from .communication import BufferBindings, BufferRegion, check_communication
from .memory import MemoryOptions
from .metadata import ValueLifetime


@dataclass(frozen=True)
class WorkspacePlan:
    """A feasible arena layout, its constraints, and the binding checked at emission."""

    bindings: BufferBindings
    peak_bytes: int
    conflicts: tuple[tuple[str, str], ...]
    solver_status: str


def plan_workspace(function, parameters, *, module=None, capacity_bytes=None, options=None) -> WorkspacePlan:
    """Assign explicitly disposable scratch parameters to one shared arena.

    Caller inputs and outputs are never implicitly reclaimed. Scratch must be
    written before being read. The existing solver consumes completion-derived
    conflicts; its result is rechecked by the same physical lifetime checker.
    Attach the returned bindings to the host function before code generation.
    """
    names = tuple(parameters)
    params = {param.name: param for param in function.params}
    if not names or len(names) != len(set(names)) or any(name not in params for name in names):
        raise ValueError("workspace requires unique declared parameter names")
    outputs = function.params[-function.output_count:] if function.output_count else ()
    if any(name in {param.name for param in outputs} for name in names):
        raise ValueError("caller outputs cannot be reclaimed as workspace")
    held = [params[name].type for name in names]
    if any(not isinstance(ty, types.TensorType) or ty.storage not in {expr.StorageKind.GMEM, expr.StorageKind.HOST}
           for ty in held) or len({ty.storage for ty in held}) != 1:
        raise ValueError("workspace parameters must share one host or device memory space")
    report = check_communication(function, regions={}, module=module)
    if report.hazards:
        raise ValueError(f"cannot allocate an unsafe execution: {report.hazards}")
    lifetimes = {name: tuple(value for value in report.lifetimes if value.allocation == name
                            and (value.producer.operation != "entry" or value.consumers)) for name in names}
    for name, values in lifetimes.items():
        if not values or any(value.producer.operation == "entry" for value in values):
            raise ValueError(f"workspace {name!r} must be written before use and cannot hold an initial input")
    conflicts = set()
    for first, second in combinations(names, 2):
        simultaneous = any(
            first in {name for name, _ in operation.accesses}
            and second in {name for name, _ in operation.accesses}
            and any(mode & expr.MemoryEffect.WRITE for name, mode in operation.accesses if name in {first, second})
            for operation in report.operations
        )
        if simultaneous or any(interfere(a, b, predecessors=report.predecessors)
                               for a in lifetimes[first] for b in lifetimes[second]):
            conflicts.add((first, second))
    values = []
    memory_level = str(held[0].storage)
    for name in names:
        accesses = lifetimes[name]
        size = prod(params[name].type.shape) * params[name].type.dtype.bit_width // 8
        lifetime = ValueLifetime(core.SSAValue(params[name]), min(v.producer.start for v in accesses),
                                 max(v.end for v in accesses), memory_level=memory_level, bytes=size)
        values.append(AllocationValue(lifetime))
    summary = AllocationSummaryMetadata(tuple(values), windows=tuple(v.lifetime for v in values),
                                        sizes={v: v.bytes for v in values})
    result = solve_allocation(
        summary, None, options or MemoryOptions(), memory_level=memory_level,
        completion_conflicts=frozenset((core.SSAValue(params[a]), core.SSAValue(params[b])) for a, b in conflicts),
        capacity_bytes=capacity_bytes,
    )
    regions = tuple((value.value.expr.name, BufferRegion(f"{function.name}.workspace", offset))
                    for value, offset in result.offsets.items())
    checked = check_communication(function, regions=dict(regions), module=module)
    if checked.hazards:
        raise ValueError(f"allocated workspace failed completion checking: {checked.hazards}")
    return WorkspacePlan(BufferBindings(regions), result.peak_bytes, tuple(sorted(conflicts)), result.solver_status)


__all__ = ["WorkspacePlan", "plan_workspace"]
