"""Build an editable project through the authored Module Target generator."""

from pathlib import Path

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.codegen.link_module import link_program
from tilefoundry.codegen.options import CodegenOptions
from tilefoundry.compile import default_pipeline


def run_scaffold(source: str, out_dir: str) -> int:
    """Require empty output, then run host-entry normalization and skeleton linking."""
    destination = Path(out_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"scaffold: output directory {destination} must be empty")
    module = load_authored_ir(source)
    options = CodegenOptions(emit="skeleton")
    link_program(
        default_pipeline(module, options).run(module), workdir=destination, options=options
    )
    return 0
