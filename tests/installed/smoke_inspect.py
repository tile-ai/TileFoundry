"""`inspect`: what this installation can do, and what one file says."""

from __future__ import annotations


def test_inspect_describes_its_commands(tf) -> None:
    done = tf("inspect")
    assert done.returncode == 0, done.stderr

    assert done.stdout.startswith("tilefoundry inspect — inspect installed target facts\n")
    assert "Usage:\n  tilefoundry inspect <command> [options]\n" in done.stdout
    assert "capabilities  the facts a selection's target was composed from" in done.stdout
    assert done.stderr == ""

    sub = tf("inspect", "capabilities", "--help")
    assert sub.returncode == 0, sub.stderr
    assert "capabilities" in sub.stdout


def test_inspect_capabilities_lists_installed_documents(tf) -> None:
    done = tf("inspect", "capabilities")
    assert done.returncode == 0, done.stderr

    assert done.stdout.startswith("Installed hardware documents:\n")
    for document in ("apple.amx", "apple.m2_pro", "nvidia.sm90", "nvidia.h200_sxm"):
        assert document in done.stdout
    assert "architectures: apple.amx" in done.stdout
    assert "architectures: nvidia.sm90" in done.stdout
    assert "Registered Target classes: amx, cpu, cuda" in done.stdout
    assert "tilefoundry inspect capabilities model.py:Model" in done.stdout
    assert done.stderr == ""


def test_inspect_capabilities_is_compact(tf, cwide) -> None:
    done = tf("inspect", "capabilities", f"{cwide}:Model.main")
    assert done.returncode == 0, done.stderr
    assert "architecture: nvidia.sm90" in done.stdout
    assert "device: nvidia.h200_sxm" in done.stdout
    assert "grid_cta_count: 168" in done.stdout
    assert "memory.hbm.bandwidth: 4800000000000 byte/s [vendor]" in done.stdout
    assert "memory.l2.bandwidth: unavailable" in done.stdout


def test_inspect_capabilities_rejects_an_uninstalled_cuda_target(
    tf, cwide, tmp_path
) -> None:
    custom = tmp_path / "custom.py"
    custom.write_text(
        cwide.read_text(encoding="utf-8")
        .replace(
            "from tilefoundry.target import CudaTarget",
            "from dataclasses import replace\n"
            "from tilefoundry.target import CudaTarget\n"
            "from tilefoundry.target.cuda.spec import installed_architecture",
        )
        .replace(
            'target=CudaTarget("nvidia.h200_sxm")',
            'target=CudaTarget("nvidia.h200_sxm", '
            'architecture=replace(installed_architecture(), name="sm_90_custom"))',
        ),
        encoding="utf-8",
    )

    done = tf("inspect", "capabilities", f"{custom}:Model.main")
    assert done.returncode == 1

    assert done.stdout == ""
    assert "no installed hardware documents" in done.stderr
    assert "sm_90_custom" in done.stderr
