"""A program with WGMMA group steps and the async proxy fence builds into one CUDA library."""

import shutil
from pathlib import Path

import pytest

from tilefoundry import cli, codegen, passes

CODEGEN = Path(__file__).parents[1] / "fixtures" / "codegen"


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="requires the CUDA compiler")
def test_wgmma_units_build(tmp_path):
    module = cli.load_authored_ir(f"{CODEGEN / 'wgmma_units.py'}:MixedUnits")
    module = passes.default_pipeline(module).run(module)
    linked = codegen.link_program(module, workdir=tmp_path)
    assert linked.library_path.is_file()
