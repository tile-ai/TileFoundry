"""What CUDA scheduling is, at each level CUDA schedules.

Two levels, two decisions. At `thread` the question is how the warps of one CTA
overlap their work asynchronously; the tile they cooperate on is the CTA's, which
is why the capacity fact projected there is per-CTA shared memory. At `cta` the
question is how work and its tensors spread across the device.

Each entry does the same four things in the same order -- build the program view,
ask the hardware once, close and solve the problem, export the plan -- and the
steps below that boundary stay private to the algorithm that owns them.
"""

from __future__ import annotations

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.specialize import origin_of
from tilefoundry.schedule import ScheduleError, ScheduleOptions
from tilefoundry.target.services import Scheduler

from .target import CudaTarget

PIPELINE_TOPOLOGY = "thread"
PARTITION_TOPOLOGY = "cta"


def _options(topology: str, options: object | None) -> ScheduleOptions:
    """The solver controls to run under, defaulted rather than inferred."""
    if options is None:
        return ScheduleOptions()
    if not isinstance(options, ScheduleOptions):
        raise ScheduleError(
            f"{topology} schedule options must be ScheduleOptions, got "
            f"{type(options).__name__}"
        )
    return options


def _entry_function(topology: str, module: Module, function: Function) -> None:
    """Refuse a leaf: these algorithms decide the whole launch.

    The entry at a chosen size is a different object and the same program, so
    the comparison follows what a specialised function records about where it
    came from. Comparing objects alone would refuse it; comparing names would
    accept anything called the same.
    """
    entry = module.entry_function()
    permitted = (entry, *entry.variants)
    candidate: object | None = function
    while candidate is not None:
        if any(candidate is allowed for allowed in permitted):
            return
        candidate = origin_of(candidate)
    raise ScheduleError(
        f"{topology} schedule requires the module entry function, got "
        f"{function.name!r}"
    )


def schedule_thread(
    module: Module,
    function: Function,
    target: CudaTarget,
    topology: object,
    options: object | None = None,
) -> object:
    """Build, close, solve, and export the intra-CTA pipeline schedule."""
    from tilefoundry.schedule.pipeline import (  # noqa: PLC0415
        PipelineFacts,
        build_pipeline_problem,
        build_pipeline_program,
        export_pipeline_plan,
        solve_pipeline_problem,
    )

    _entry_function(PIPELINE_TOPOLOGY, module, function)
    _options(PIPELINE_TOPOLOGY, options)
    program = build_pipeline_program(module, function)
    facts = target.get_facts(PipelineFacts, program.facts_query(PIPELINE_TOPOLOGY))
    problem = build_pipeline_problem(program, facts, topology)
    solution = solve_pipeline_problem(problem)
    return export_pipeline_plan(program, solution, target)


def schedule_cta(
    module: Module,
    function: Function,
    target: CudaTarget,
    topology: object,
    options: object | None = None,
) -> object:
    """Build, close, solve, and export the device-wide partition schedule."""
    from tilefoundry.schedule.partition import (  # noqa: PLC0415
        PartitionFacts,
        build_partition_problem,
        build_partition_program,
        export_partition_plan,
        solve_partition_problem,
    )

    _entry_function(PARTITION_TOPOLOGY, module, function)
    resolved = _options(PARTITION_TOPOLOGY, options)
    program = build_partition_program(module, function)
    facts = target.get_facts(PartitionFacts, program.facts_query(PARTITION_TOPOLOGY))
    problem = build_partition_problem(program, facts, topology)
    solution = solve_partition_problem(problem, resolved)
    return export_partition_plan(problem, solution)


__all__ = [
    "PARTITION_TOPOLOGY",
    "PIPELINE_TOPOLOGY",
    "schedule_cta",
    "cuda_scheduler",
    "schedule_thread",
]


def cuda_scheduler(topology: str) -> Scheduler | None:
    """Construct the CUDA scheduler requested for one topology level."""
    if topology == PIPELINE_TOPOLOGY:
        return Scheduler(PIPELINE_TOPOLOGY, schedule_thread)
    if topology == PARTITION_TOPOLOGY:
        return Scheduler(PARTITION_TOPOLOGY, schedule_cta)
    return None
