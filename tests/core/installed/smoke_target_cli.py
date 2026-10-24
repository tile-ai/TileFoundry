"""Installed CPU identities and target command diagnostics."""

from __future__ import annotations

from tests.core.installed._targets import check_target_list, check_unknown_identity


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
    check_target_list(tf, (("CpuTarget()", "cpu"),), ("from tilefoundry.target import CpuTarget",))


def test_target_show_rejects_an_unknown_identity(tf) -> None:
    check_unknown_identity(tf, ("cpu",))
