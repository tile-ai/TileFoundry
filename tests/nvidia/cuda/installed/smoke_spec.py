"""Installed CUDA specifications and their queryable rules."""

from __future__ import annotations

import pytest

from tests.core.installed._spec import check_rule, listed_documents


def test_spec_lists_the_documents_there_are(tf) -> None:
    listed = listed_documents(tf)
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
    (("cuda/target", "topology-levels", "`Topology.size` MUST be an explicit `ShapeDim`"),),
)
def test_spec_answers_askable_rules(tf, topic, section, expected) -> None:
    check_rule(tf, topic, section, expected)
