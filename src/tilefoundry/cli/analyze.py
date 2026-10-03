"""The `analyze` command: type-check and analyse an authored HIR selection."""

from __future__ import annotations

import os
import sys
import textwrap
import threading
from pathlib import Path
from typing import Mapping

from tilefoundry.analysis import analyze, check_program
from tilefoundry.analysis.check import resolve_program_geometry
from tilefoundry.cli.source import load_authored_ir, require_bound_dims
from tilefoundry.inspection import PythonPrintOptions, as_script
from tilefoundry.inspection.analysis_report import (
    render_analysis,
    render_json,
    render_text,
)
from tilefoundry.ir.hir.specialize import SpecializationError
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext

EVIDENCE: dict[str, str] = {
    "compute-cost": "the logical work and traffic of every value: flops by dtype, bytes moved",
    "memory": "where traffic lands and what storage remains live against capacity",
    "roofline": "which of compute or memory limits each value, and the limit in time",
    "performance": "when each value runs, where its buffers fit, and the time that takes",
}


ANALYSES: tuple[str, ...] = tuple(EVIDENCE)
_ANALYSIS_TIMEOUT_SECONDS = 300.0


def _watch(limit: float) -> threading.Event:
    """Bound the whole analyze command until its caller disarms the timer."""
    disarmed = threading.Event()

    def stop() -> None:
        if disarmed.wait(limit):
            return
        sys.stderr.write(
            f"tilefoundry: error: analysis too complex, timed out after {limit:.0f}s\n"
        )
        sys.stderr.flush()
        os._exit(1)

    threading.Thread(target=stop, daemon=True).start()
    return disarmed


def guidance() -> str:
    """What the command reports, and what it leaves to whoever read it.

    What each analysis reports is not restated here: the flags above already say
    it, off the same table, and a second copy is one more thing that can drift.
    """
    return textwrap.dedent(
        """\
        evidence, not a decision. This reports what a program would cost. It
        searches nothing, rewrites nothing, and picks no optimization: what to do
        about a number is yours.

        Complete inferred types are printed whatever the flags are, and with no flag
        that is the whole answer: analyze type-checks the selection and prints it.
        Each flag above adds one root analysis, and every analysis is asked for by
        name.

        It reads the authored program, so it answers before any implementation of it
        exists -- and its numbers are the floor an implementation is read against,
        not a measurement of one. That is the whole use: without an independent
        bound, the fastest thing you have measured becomes the ceiling you believe
        in, and "already at roofline" is a claim about your own best attempt.
        Reading authored HIR is what it takes as input; making something fast is
        what it is for.

        The selection must be a Module that reaches a declared target, so name it
        from the root down. That Module's resolved Target is the hardware every
        number is measured against, which is why there is no --target.

        family         what --topology changes                 pass it when
        ------------   --------------------------------------  ---------------------
        compute-cost   nothing. Every kind states its total     never
                       and every level's per-unit share
        memory         nothing for traffic, which states every   the program shards
                       level; placement remains Function-wide
        roofline       nothing. The bound is the machine's     never
                       and is unchanged by program splits
        performance    which level's parallel capacity the     the program shards
                       plan is issued against

        Two assumptions the reported numbers rest on:
          logical traffic omits loop replication that does not change an access;
            total traffic counts every executed occurrence. Per-unit traffic is
            the selected topology unit's share of that executed total.
          a reported placement peak holds under the order this walk took. Which
            order the program really takes is settled by scheduling, so the peak
            is an observation, not a bound.

        Each family's record, how every field is computed, and what it prints:
          tilefoundry spec analysis 1.2.1    compute-cost
          tilefoundry spec analysis 1.2.2    memory
          tilefoundry spec analysis 1.2.3    roofline
          tilefoundry spec analysis 1.2.4    performance
        """
    )


def run_authored_analysis(
    source: str,
    analyses: tuple[str, ...],
    out_path: str,
    *,
    topology: str | None = None,
    as_json: bool = False,
    operands: bool = False,
    dims: Mapping[str, int] | None = None,
) -> int:
    """Analyse one authored HIR selection and print what was found.

    One public call resolves the requested roots' union closure. Each member
    runs once on one view, and Metadata ownership keeps one family from changing
    another's records.
    """
    disarmed = _watch(_ANALYSIS_TIMEOUT_SECONDS)
    try:
        module = load_authored_ir(source)
        function = module.entry_function()
        require_bound_dims(module, function, dims, command="analyze")
        if not analyses:
            try:
                checked_module, checked = resolve_program_geometry(
                    module,
                    function,
                    dims,
                    TypeInferContext(scope=FunctionScope(module, function)),
                )
            except SpecializationError as error:
                raise ValueError(f"analyze: {error}") from None
            expanded = check_program(checked_module, checked)
            annotated = as_script(expanded, options=PythonPrintOptions(show_types=True))
            Path(out_path).write_text(annotated, encoding="utf-8")
            return 0

        result = analyze(module, function, analysis=analyses, topology_level=topology, dims=dims)
        rendered = render_analysis(result, operands=operands and not as_json)
        if as_json:
            Path(out_path).write_text(
                f"{render_json({**rendered.data, 'source': rendered.annotated})}\n",
                encoding="utf-8",
            )
            return 0

        Path(out_path).write_text(
            f"{render_text(rendered)}\n\n{rendered.annotated}", encoding="utf-8"
        )
        return 0
    finally:
        disarmed.set()


__all__ = ["ANALYSES", "EVIDENCE", "guidance", "run_authored_analysis"]
