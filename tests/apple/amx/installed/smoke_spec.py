"""Tests and helpers for the owning target."""

from __future__ import annotations


def test_spec_lists_the_documents_there_are(tf) -> None:
    done = tf("spec")
    assert done.returncode == 0, done.stderr
    listed = done.stdout
    assert {'amx/target', 'amx/tir'} <= set(listed.split())
