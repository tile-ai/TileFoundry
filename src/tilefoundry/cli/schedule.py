"""Command-line assembly for schedule workflows."""

from __future__ import annotations

import json
from pathlib import Path

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.inspection import as_script
from tilefoundry.schedule import finalize


def run_finalize(source: str, out: str, *, as_json: bool = False) -> int:
    """Finalize ``source`` and write the sole requested representation."""
    function = finalize(load_authored_ir(source))
    rendered = as_script(function)
    text = json.dumps({"source": rendered}, indent=2) if as_json else rendered
    destination = Path(out)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(f"{text.rstrip()}\n", encoding="utf-8")
    temporary.replace(destination)
    return 0


__all__ = ["run_finalize"]
