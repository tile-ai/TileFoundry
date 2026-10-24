"""The specs the wheel ships, read back through the command."""

from __future__ import annotations

import pytest


def test_spec_lists_the_documents_there_are(tf) -> None:
    done = tf("spec")
    assert done.returncode == 0, done.stderr
    listed = done.stdout
    assert {
        "cuda/target",
        "cuda/cli",
        "cuda/tir",
        "cuda/runtime",
        "cuda/hir",
        "cuda/passes",
        "cuda/codegen",
    } <= set(listed.split())


@pytest.mark.parametrize(
    ("topic", "section", "expected"),
    (
        ("cuda/target", "topology-levels", "`Topology.size` MUST be an explicit `ShapeDim`"),
    ),
)
def test_spec_answers_askable_rules(tf, topic, section, expected) -> None:
    done = tf("spec", topic, section)
    assert done.returncode == 0, done.stderr
    assert expected in done.stdout
