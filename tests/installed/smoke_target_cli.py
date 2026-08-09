"""`target`: constructible target values and their retained documents."""

from __future__ import annotations


def test_target_describes_its_commands(tf) -> None:
    done = tf("target")
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith(
        "tilefoundry target — list, show, add, or remove compilation targets\n"
    )
    assert (
        "Usage:\n  tilefoundry [--registry PATH] target <command> [options]\n"
        in done.stdout
    )
    assert "list    list every available target" in done.stdout
    assert "show    show the documents retained" in done.stdout
    assert "add     add one Target provider or hardware document" in done.stdout
    assert "remove  remove one entry shown by target list" in done.stdout
    assert done.stderr == ""


def test_target_list_prints_reconstructing_values(tf) -> None:
    done = tf("target", "list")
    assert done.returncode == 0, done.stderr
    for expression, identity in (
        ('AmxTarget("apple.m2_pro")', "apple.m2_pro"),
        ("CpuTarget()", "cpu"),
        ('CudaTarget("nvidia.b200_sxm")', "nvidia.b200_sxm"),
        ('CudaTarget("nvidia.h200_sxm")', "nvidia.h200_sxm"),
    ):
        assert expression in done.stdout
        assert f"identity: {identity}" in done.stdout
    assert "from tilefoundry.target import AmxTarget, CpuTarget, CudaTarget" in done.stdout
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


def test_target_show_without_documents_prints_only_identity(tf) -> None:
    done = tf("target", "show", "cpu")
    assert done.returncode == 0, done.stderr
    assert done.stdout == "identity: cpu\nCpuTarget()\nfacts: unavailable\n"
    assert done.stderr == ""


def test_target_show_rejects_an_unknown_identity(tf) -> None:
    done = tf("target", "show", "vendor.missing")
    assert done.returncode == 1
    assert done.stdout == ""
    for identity in ("apple.m2_pro", "cpu", "nvidia.b200_sxm", "nvidia.h200_sxm"):
        assert identity in done.stderr


def test_inspect_is_not_a_command(tf) -> None:
    done = tf("inspect")
    assert done.returncode == 2
    assert done.stdout == ""
    assert "invalid choice: 'inspect'" in done.stderr
