"""CLI target listing reconstructs each advertised target value."""

from __future__ import annotations

from tests.core.cli._targets import listed_identities


def test_target_list_expressions_execute_and_show_accepts_their_identities(
    capsys, monkeypatch
) -> None:
    identities = listed_identities(capsys, monkeypatch)
    assert {
        "apple.m2_pro",
        "cpu",
    } < set(identities)
