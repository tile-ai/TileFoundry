"""Command-line assembly for schedule workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from tilefoundry import analysis, inspection, passes, schedule
from tilefoundry.passes import transforms

from .source import checked_program
from .target import target_by_identity


def write(out: str, text: str) -> None:
    destination = Path(out)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(f"{text.rstrip()}\n", encoding="utf-8")
    temporary.replace(destination)


def run_finalize(
    source: str, out: str, *, as_json: bool = False, dims: Sequence[str] | None = None
) -> int:
    """Finalize ``source`` and write the sole requested representation."""
    program = checked_program(source, dims, command="schedule finalize")
    lowering = program.module.resolve_target().resolve(transforms.ConvertHIRToTIR, passes.Options())
    analyzed = analysis.analyze(program.module, program.entry, analysis=lowering.analyses)
    module = schedule.finalize(analyzed)
    rendered = inspection.as_script(module)
    text = json.dumps({"source": rendered}, indent=2) if as_json else rendered
    write(out, text)
    return 0


def run_facts(
    op: str | None,
    target: str,
    out: str,
    *,
    as_json: bool = False,
) -> int:
    """Print target-filtered Op declarations to ``out``."""
    selected = target_by_identity(target)
    format = "json" if as_json else "text"
    text = (
        schedule.render_all(schedule.listing(selected), selected, format=format)
        if op is None else schedule.render(schedule.one(selected, op), selected, format=format)
    )
    write(out, text)
    return 0


def run_candidates(
    source: str,
    out: str,
    *,
    as_json: bool = False,
    dims: Sequence[str] | None = None,
) -> int:
    """Write instruction candidates for the unscheduled sites in ``source``."""
    program = checked_program(source, dims, command="schedule candidates")
    report = schedule.candidates(program.module, program.entry, source=source)
    text = json.dumps(report, indent=2) if as_json else schedule.render_candidates(report)
    write(out, text)
    return 0


__all__ = ["run_candidates", "run_facts", "run_finalize"]
