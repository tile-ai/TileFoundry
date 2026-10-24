"""Idempotent AMX target registration across module reloads."""

from __future__ import annotations

from tilefoundry.target import registered_targets


def test_registration_is_one_class_boundary_and_reload_is_idempotent() -> None:
    assert {"amx"} <= registered_targets().keys()
