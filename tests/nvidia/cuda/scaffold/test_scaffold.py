"""Standalone CUDA projects exercised through their authored runtime modules."""

import subprocess
from importlib import import_module
from pathlib import Path

import pytest
import torch

import tilefoundry
from tilefoundry.cli import main as cli_main
from tilefoundry.cli.source import load_authored_ir

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scaffold"

CASES = ("hir_moe", "hir_layers", "tir_launches", "tir_dispatch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fixture", CASES)
def test_skeleton_builds_and_every_export_is_callable(fixture, tmp_path):
    source = f"{FIXTURES / (fixture + '.py')}:Root"
    project = tmp_path / "project"
    assert cli_main(["scaffold", source, str(project)]) == 0
    assert (project / "CMakeLists.txt").is_file()
    assert (project / "host.cpp").is_file()
    assert list(project.glob("device_*.cu"))
    assert (project / "build" / "libRoot.so").is_file()
    fresh = tmp_path / "standalone"
    subprocess.run(
        ["cmake", "-G", "Ninja", "-S", str(project), "-B", str(fresh)],
        check=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        ["cmake", "--build", str(fresh), "--parallel", "2"],
        check=True,
        text=True,
        timeout=300,
    )
    fixture_module = import_module(f"tests.nvidia.cuda.fixtures.scaffold.{fixture}")
    runtime = fixture_module.Runtime(fresh / "libRoot.so")
    for extent in (2, 8, 14):
        runtime(torch.randn(extent, 8, dtype=torch.float32, device="cuda"))
        torch.cuda.synchronize()


def test_nonempty_output_is_preserved(tmp_path, capsys):
    sentinel = tmp_path / "authored.cu"
    sentinel.write_text("handwritten kernel")
    assert cli_main(["scaffold", f"{FIXTURES / 'hir_moe.py'}:Root", str(tmp_path)]) == 1
    assert "must be empty" in capsys.readouterr().err
    assert sentinel.read_text() == "handwritten kernel"
    assert tuple(tmp_path.iterdir()) == (sentinel,)


@pytest.mark.parametrize("target", ("CpuTarget()", 'CudaTarget("nvidia.h200_sxm", device_count=2)'))
def test_codegen_reports_unsupported_scaffold_targets(target, tmp_path, capsys):
    variant = "scaffold_cpu.py" if target == "CpuTarget()" else "scaffold_multi_cuda.py"
    source = FIXTURES.parent / "diagnostics" / variant
    project = tmp_path / "project"
    assert cli_main(["scaffold", f"{source}:Root", str(project)]) == 1
    message = "expected one device Target, found 0" if target == "CpuTarget()" else "only single-device CUDA"
    assert message in capsys.readouterr().err
    assert not project.exists()


@pytest.mark.parametrize("fixture", ("hir_moe", "hir_layers"))
def test_full_refuses_hir(fixture):
    module = load_authored_ir(f"{FIXTURES / (fixture + '.py')}:Root")
    with pytest.raises(ValueError, match='emit="skeleton" or lowering to TIR'):
        tilefoundry.compile(module)
