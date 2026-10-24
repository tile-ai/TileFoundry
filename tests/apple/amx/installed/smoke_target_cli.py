"""`target`: constructible target values and their retained documents."""

from __future__ import annotations


def test_target_list_prints_reconstructing_values(tf) -> None:
    done = tf("target", "list")
    assert done.returncode == 0, done.stderr
    for expression, identity in (
        ('AmxTarget("apple.m2_pro")', "apple.m2_pro"),
        ("CpuTarget()", "cpu"),
    ):
        assert expression in done.stdout
        assert f"identity: {identity}" in done.stdout
    assert "from tilefoundry.target import CpuTarget" in done.stdout
    assert "from tilefoundry_apple import AmxTarget" in done.stdout
    assert done.stderr == ""


def test_target_show_rejects_an_unknown_identity(tf) -> None:
    done = tf("target", "show", "vendor.missing")
    assert done.returncode == 1
    assert done.stdout == ""
    for identity in ("cpu", "apple.m2_pro"):
        assert identity in done.stderr
