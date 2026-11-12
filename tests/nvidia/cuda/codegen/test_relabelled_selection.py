"""A relabelled selection that starts at thread 0 runs behind the runtime's ``contains`` guard."""

from __future__ import annotations

from pathlib import Path

from tilefoundry import analysis, cli
from tilefoundry.schedule import finalize
from tilefoundry_nvidia import cuda
from tilefoundry_nvidia.cuda import codegen

CODEGEN = Path(__file__).parents[1] / "fixtures" / "codegen"


def test_a_relabelled_head_is_guarded() -> None:
    module = cli.load_authored_ir(str(CODEGEN / "relabelled_head.py"))
    lowered = finalize(
        analysis.analyze(*analysis.check_program(module, module.entry_function()), analysis=("memory", "sync"))
    )
    (kernel,) = [function for function in lowered.functions if isinstance(function.target, cuda.Target)]
    context = codegen.Context(target=kernel.target)
    context.emit_node(kernel.body)
    assert context.source().count("if (tilefoundry::contains(") == 1
