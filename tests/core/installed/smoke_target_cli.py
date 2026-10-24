"""Tests and helpers for the owning target."""

from __future__ import annotations


def test_target_show_without_documents_prints_only_identity(tf) -> None:
    done = tf("target", "show", "cpu")
    assert done.returncode == 0, done.stderr
    assert done.stdout == "identity: cpu\nCpuTarget()\nfacts: unavailable\n"
    assert done.stderr == ""


def test_inspect_is_not_a_command(tf) -> None:
    done = tf("inspect")
    assert done.returncode == 2
    assert done.stdout == ""
    assert "invalid choice: 'inspect'" in done.stderr


def test_target_list_prints_reconstructing_values(tf) -> None:
    done = tf("target", "list")
    assert done.returncode == 0, done.stderr
    for expression, identity in (
        ("CpuTarget()", "cpu"),
    ):
        assert expression in done.stdout
        assert f"identity: {identity}" in done.stdout
    assert "from tilefoundry.target import CpuTarget" in done.stdout
    assert done.stderr == ""


def test_target_show_rejects_an_unknown_identity(tf) -> None:
    done = tf("target", "show", "vendor.missing")
    assert done.returncode == 1
    assert done.stdout == ""
    for identity in ("cpu",):
        assert identity in done.stderr
