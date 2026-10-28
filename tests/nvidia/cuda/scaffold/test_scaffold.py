"""Standalone CUDA projects exercised through their authored runtime modules."""

import subprocess
from pathlib import Path

import pytest
import torch

import tilefoundry
from tilefoundry.cli import main as cli_main
from tilefoundry.cli.source import load_authored_ir, load_namespace

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scaffold"

CASES = ("hir_moe", "hir_layers", "tir_launches", "tir_dispatch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fixture", CASES)
def test_scaffold_skeleton_builds_and_runs(fixture, tmp_path):
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
    Runtime = load_namespace(str(FIXTURES / f"{fixture}.py"))[0]["Runtime"]
    runtime = Runtime(fresh / "libRoot.so")
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


def test_scaffold_rejects_omitted_child_weights(tmp_path, capsys):
    source = FIXTURES.parent / "schedule" / "hir" / "module_instances.py"
    assert cli_main(["scaffold", str(source), str(tmp_path / "project")]) == 1
    assert "omitted weights" in capsys.readouterr().err


@pytest.mark.parametrize("fixture", ("hir_moe", "hir_layers"))
def test_scaffold_full_refuses_hir(fixture):
    module = load_authored_ir(f"{FIXTURES / (fixture + '.py')}:Root")
    with pytest.raises(ValueError, match="finalize needs the result of analyze"):
        tilefoundry.compile(module)
