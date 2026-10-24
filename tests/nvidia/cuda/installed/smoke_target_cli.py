"""`target`: constructible target values and their retained documents."""

from __future__ import annotations


def test_target_list_prints_reconstructing_values(tf) -> None:
    done = tf("target", "list")
    assert done.returncode == 0, done.stderr
    for expression, identity in (
        ("CpuTarget()", "cpu"),
        ('CudaTarget("nvidia.b200_sxm")', "nvidia.b200_sxm"),
        ('CudaTarget("nvidia.h200_sxm")', "nvidia.h200_sxm"),
    ):
        assert expression in done.stdout
        assert f"identity: {identity}" in done.stdout
    assert "from tilefoundry.target import CpuTarget" in done.stdout
    assert "from tilefoundry_nvidia import CudaTarget" in done.stdout
    assert done.stderr == ""

def test_target_show_prints_retained_documents(tf) -> None:
    done = tf("target", "show", "nvidia.h200_sxm")
    assert done.returncode == 0, done.stderr
    assert "architecture: nvidia.sm90" in done.stdout
    assert "device: nvidia.h200_sxm" in done.stdout
    assert done.stdout.count("  digest: ") == 2
    assert "memory.hbm.bandwidth: 4800000000000 byte/s [vendor]" in done.stdout
    assert "grid_cta_count" not in done.stdout
    assert done.stderr == ""


def test_target_show_rejects_an_unknown_identity(tf) -> None:
    done = tf("target", "show", "vendor.missing")
    assert done.returncode == 1
    assert done.stdout == ""
    for identity in ("cpu", "nvidia.b200_sxm", "nvidia.h200_sxm"):
        assert identity in done.stderr
