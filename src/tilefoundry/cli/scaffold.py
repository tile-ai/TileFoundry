"""Build an editable project through the authored Module Target generator."""

from pathlib import Path

from tilefoundry import codegen, passes

from .source import load_authored_ir


def run_scaffold(source: str, out_dir: str) -> int:
    """Require empty output, then run host-entry normalization and skeleton linking."""
    destination = Path(out_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"scaffold: output directory {destination} must be empty")
    module = load_authored_ir(source)
    codegen.link_program(
        passes.default_pipeline(module, passes.Options(lower=False)).run(module),
        workdir=destination,
        options=codegen.Options(emit="skeleton"),
    )
    return 0


__all__ = ["run_scaffold"]
