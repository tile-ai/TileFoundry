"""The `analyze` command: type-check and analyse an authored HIR selection."""

from __future__ import annotations

import sys
import textwrap
from typing import Mapping

from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.preflight import infer_authored_types, validate_authored
from tilefoundry.analysis.walk import reachable_functions
from tilefoundry.cli.source import load_authored_ir, suggested_extents
from tilefoundry.inspection import PythonPrintOptions, as_script
from tilefoundry.inspection.analysis_report import (
    render_json,
    render_text,
    report,
    selected_types,
)
from tilefoundry.ir.hir.specialize import dim_vars_reached, specialize_concretely
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext

EVIDENCE: dict[str, str] = {
    "compute-cost": "the logical work and traffic of every value: flops by dtype, bytes moved",
    "memory": "where that traffic lands, and the footprint it holds live against the capacity",
    "roofline": "which of compute or memory limits each value, and the limit in time",
    "timeline": "when each value runs under those limits, and over how many units",
}


ANALYSES: tuple[str, ...] = tuple(EVIDENCE)


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
        compute-cost   flops_per_unit. flops and traffic are   the program shards
                       the device's and do not move
        memory         nothing. Footprint follows its owner    never
                       for each storage level
        roofline       nothing. The bound is the machine's     never
                       and is unchanged by program splits
        timeline       which level's parallel capacity the     the program shards
                       plan is issued against

        Two assumptions the reported numbers rest on:
          traffic is the device's and counted once, so units reading one operand
            in common are assumed to read it from memory once. Whether they can is
            a residency question, reported as an advisory.
          a reported peak footprint holds under the order this walk took. Which
            order the program really takes is settled by scheduling, so the peak
            is an observation, not a bound.
        """
    )


def run_authored_analysis(
    source: str,
    analyses: tuple[str, ...],
    *,
    topology: str | None = None,
    as_json: bool = False,
    dims: Mapping[str, int] | None = None,
) -> int:
    """Analyse one authored HIR selection and print what was found.

    One public call per requested root, because the operation takes one root at
    a time. The renderings are composed from those results afterwards, so
    requesting two analyses cannot change what either of them reports.
    """
    module = load_authored_ir(source)
    function = module.entry_function()
    stated = {} if dims is None else dims
    unbound = [
        (name, dim_var)
        for name, dim_var in dim_vars_reached(function).items()
        if name not in stated
    ]
    if unbound:
        guidance = "; ".join(
            f"{name} is declared as [{dim_var.lo}, {dim_var.hi}); bind it with "
            f"--dim {name}=EXTENT (try {', '.join(map(str, suggested_extents(dim_var.lo, dim_var.hi)))})"
            for name, dim_var in unbound
        )
        raise ValueError(f"analyze needs one EXTENT for every open dimension: {guidance}")
    if not analyses:
        checked = (
            specialize_concretely(
                function, dims, TypeInferContext(scope=FunctionScope(module, function))
            )
            if dims is not None
            else function
        )
        functions = reachable_functions(checked)
        infer_authored_types(functions, module)
        validate_authored(functions)
        annotated = as_script(module, options=PythonPrintOptions(show_types=True))
        sys.stdout.write(annotated)
        return 0

    results = [
        analyze(module, function, analysis=name, level=topology, dims=dims)
        for name in analyses
    ]
    data = report(results)
    if as_json:
        sys.stdout.write(f"{render_json(data)}\n")
        return 0



    annotated = as_script(
        module,
        options=PythonPrintOptions(
            show_types=True, comment_metadata_types=selected_types(results)
        ),
    )
    sys.stdout.write(f"{render_text(data)}\n\n{annotated}")
    return 0


__all__ = ["ANALYSES", "EVIDENCE", "guidance", "run_authored_analysis"]
