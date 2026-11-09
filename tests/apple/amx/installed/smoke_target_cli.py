"""Installed AMX target expressions and identity diagnostics."""

from __future__ import annotations

from tests.core.installed._targets import check_target_list, check_unknown_identity


def test_target_list_prints_reconstructing_values(tf) -> None:
    check_target_list(
        tf,
        (
            ('amx.Target("apple.m2_pro")', "apple.m2_pro"),
            ("cpu.Target()", "cpu"),
        ),
        ("from tilefoundry.target import cpu", "from tilefoundry_apple import amx"),
    )


def test_target_show_rejects_an_unknown_identity(tf) -> None:
    check_unknown_identity(tf, ("cpu", "apple.m2_pro"))
