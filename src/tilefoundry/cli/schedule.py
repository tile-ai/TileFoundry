"""Command-line assembly for schedule workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.cli.target import target_by_identity
from tilefoundry.inspection import as_script
from tilefoundry.schedule import (
    candidates,
    finalize,
    listing,
    one,
    render,
    render_candidates,
)


def _write(out: str, text: str) -> None:
    destination = Path(out)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(f"{text.rstrip()}\n", encoding="utf-8")
    temporary.replace(destination)


def run_finalize(
    source: str, out: str, *, as_json: bool = False, dims: Sequence[str] | None = None
) -> int:
    """Finalize ``source`` and write the sole requested representation."""
    program = checked_program(source, dims, command="schedule finalize")
    analyzed = analyze(program.module, program.entry, analysis=("memory",))
    module = finalize(analyzed, authored=program.authored)
    rendered = as_script(module)
    text = json.dumps({"source": rendered}, indent=2) if as_json else rendered
    _write(out, text)
    return 0


def run_facts(
    instruction: str | None,
    target: str,
    out: str,
    *,
    as_json: bool = False,
) -> int:
    """Write target-filtered instruction facts to ``out``."""
    selected = target_by_identity(target)
    report = listing(selected) if instruction is None else one(selected, instruction)
    text = json.dumps(report, indent=2) if as_json else render(report)
    _write(out, text)
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
    report = candidates(program.module, program.entry, source=source)
    text = json.dumps(report, indent=2) if as_json else render_candidates(report)
    _write(out, text)
    return 0


__all__ = ["run_candidates", "run_facts", "run_finalize"]
