"""The `schedule` command: schedule an authored Module at one topology level."""

from __future__ import annotations

import sys
import textwrap
from dataclasses import replace
from typing import Mapping

from tilefoundry.cli.source import entry_function, load_authored_ir
from tilefoundry.schedule import ScheduleOptions, schedule


def guidance() -> str:
    """What the plan is, and what it is not."""
    return textwrap.dedent(
        """\
        advice, not a rewrite. By the time you ask, you have already chosen the
        program, the logical tile, the fusion boundary and the level -- this answers
        what a plan at that level costs, and nothing about whether you chose well.

        It does not rewrite the source, is not a stage of compilation, and decides
        nothing: accept the plan, reject it, or change the candidate and ask again.
        The Module it answers about is the one you handed it, unchanged.

        What a plan says is the level's own algorithm's to say, so two levels answer
        in different shapes. A CUDA partition plan names where each step sits across
        the level's positions, the layout changes it needs -- including the ones it
        synthesized for you -- and the interval each step lands in against the bound.

        A search that ends with no answer is reported as the search ending, not as
        the selection having no schedule. --first-plan asks for a plan rather than
        the best one, which is what you want while you are still deciding whether
        the candidate is worth the search at all.

        --topology must name a level the Module declares. The target is not a flag:
        it is the selected Module's own, because a kernel is authored against one
        target.
        """
    )


def run_schedule(
    source: str,
    topology: str,
    *,
    as_json: bool = False,
    dims: Mapping[str, int] | None = None,
    solver_timeout: float | None = None,
    solver_workers: int | None = None,
    first_plan: bool = False,
) -> int:
    """Schedule one authored Module through the public Schedule operation.

    The solver's budget is stated here rather than left to the library default
    because two things about it are the caller's to decide and neither is visible
    from inside. The worker count: the default lets the solver size itself to the
    machine, and several solvers each doing that on one machine oversubscribe it
    until none returns an answer, which looks like the model being unschedulable and
    is not. And whether the best plan is wanted at all: the search keeps improving
    until its limit, so a caller who needs a plan rather than the best one otherwise
    waits out the whole budget for an answer it had early.
    """
    ir = load_authored_ir(source)
    function = entry_function(ir)
    options = None
    if solver_timeout is not None or solver_workers is not None or first_plan:
        options = ScheduleOptions(stop_at_first_solution=first_plan)
        if solver_timeout is not None:
            options = replace(options, timeout_seconds=solver_timeout)
        if solver_workers is not None:
            options = replace(options, workers=solver_workers)
    result = schedule(ir, function, topology=topology, dims=dims, options=options)
    sys.stdout.write((result.plan.to_json() if as_json else result.plan.render()) + "\n")
    return 0


__all__ = ["guidance", "run_schedule"]
