"""The `analyze` command: type-check and analyse an authored HIR selection."""

from __future__ import annotations

import sys
from typing import Mapping

from tilefoundry.analysis.api import analyze
from tilefoundry.cli.source import load_authored_ir, suggested_extents
from tilefoundry.inspection import PythonPrintOptions, as_script
from tilefoundry.inspection.analysis_report import (
    render_json,
    render_text,
    report,
    selected_types,
)
from tilefoundry.ir.hir.specialize import dim_vars_reached

# The root analyses `analyze` can be asked for, in the order they are reported.
ANALYSES = ("compute-cost", "memory", "roofline", "timeline")


def run_authored_analysis(
    source: str,
    analyses: tuple[str, ...],
    *,
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
    results = [
        analyze(module, function, analysis=name, dims=dims) for name in analyses
    ]
    data = report(results)
    if as_json:
        sys.stdout.write(f"{render_json(data)}\n")
        return 0
    # The annotated IR and the report are two views of one run, so they choose
    # what to show the same way: a dependency that ran unrequested is not
    # commented into the source either.
    annotated = as_script(
        module,
        options=PythonPrintOptions(
            show_types=True, comment_metadata_types=selected_types(results)
        ),
    )
    sys.stdout.write(f"{render_text(data)}\n\n{annotated}")
    return 0


__all__ = ["ANALYSES", "run_authored_analysis"]
