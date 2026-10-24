"""Installed AMX specifications exposed through the spec command."""

from __future__ import annotations

from tests.core.installed._spec import listed_documents


def test_spec_lists_the_documents_there_are(tf) -> None:
    listed = listed_documents(tf)
    assert {"amx/target", "amx/tir"} <= set(listed.split())
