"""Build an editable project through the authored Module Target generator."""

from pathlib import Path

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.codegen.link_module import link_program
from tilefoundry.codegen.options import CodegenOptions
from tilefoundry.ir.core.module import module_functions
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.passes.pass_manager import PassManager
from tilefoundry.passes.transforms import InsertHostEntryPass
from tilefoundry.target import CpuTarget


def run_scaffold(source: str, out_dir: str) -> int:
    """Require empty output, then run host-entry normalization and skeleton linking."""
    destination = Path(out_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"scaffold: output directory {destination} must be empty")
    module = load_authored_ir(source)
    prepared = PassManager().add(InsertHostEntryPass()).run(module)
    entries = tuple(
        fn
        for fn in module_functions(prepared)
        if isinstance(fn, PrimFunction) and isinstance(fn.target, CpuTarget)
    )
    link_program(
        prepared, workdir=destination, options=CodegenOptions(emit="skeleton"), entries=entries
    )
    return 0
