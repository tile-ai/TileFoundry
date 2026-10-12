"""Build an editable project through the authored Module Target generator."""

from pathlib import Path

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.compile import scaffold


def run_scaffold(source: str, out_dir: str) -> int:
    """Require empty output, then delegate project generation to the compiler."""
    destination = Path(out_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"scaffold: output directory {destination} must be empty")
    module = load_authored_ir(source)
    scaffold(module, destination)
    return 0
