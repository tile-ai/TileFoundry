"""The specs the wheel ships, read back through the command."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_spec_lists_the_documents_there_are(tf) -> None:
    done = tf("spec")
    assert done.returncode == 0, done.stderr
    listed = done.stdout

    assert "hir" in listed and "dsl" in listed
    assert "runtime" in listed


def test_spec_outlines_a_document_rather_than_printing_it(tf, shipped) -> None:
    done = tf("spec", "dsl")
    assert done.returncode == 0, done.stderr
    outline = done.stdout
    # `dsl` is an alias for hir.md, so this is the document being outlined.
    whole = (Path(shipped["spec"]) / "hir.md").read_text(encoding="utf-8")

    assert "Silu" in outline and "silu" in outline
    assert len(outline) < len(whole) / 4
    assert "class Silu(Op):" not in outline


def test_spec_prints_one_section_and_the_keys_beside_it(tf) -> None:
    done = tf("spec", "dsl", "silu")
    assert done.returncode == 0, done.stderr
    section = done.stdout

    assert "class Silu(Op):" in section
    assert "next:     rmsnorm" in section
    assert "class RMSNorm(Op):" not in section


def test_spec_lists_and_prints_cache_update(tf) -> None:
    outline = tf("spec", "hir")
    assert outline.returncode == 0, outline.stderr
    assert "CacheUpdate" in outline.stdout

    done = tf("spec", "hir", "cacheupdate")
    assert done.returncode == 0, done.stderr
    assert "class CacheUpdate(Op):" in done.stdout
    assert "eval/runtime, not typeinfer" in done.stdout


def test_spec_separates_two_sections_that_would_share_a_key(tf) -> None:
    """`tir.md` names a field `name` twice, under `SymbolRef` and under
    `MmaOpSpec`. Each is reachable by its enclosing section; the bare key is
    not, because it would have to pick one."""
    symbol = tf("spec", "tir", "symbolref/name")
    assert symbol.returncode == 0, symbol.stderr
    assert "canonical name of a `PrimFunction`" in symbol.stdout
    assert "uniquely identify the instruction" not in symbol.stdout

    atom = tf("spec", "tir", "mmaopspec/name")
    assert atom.returncode == 0, atom.stderr
    assert "uniquely identify the instruction" in atom.stdout

    bare = tf("spec", "tir", "name")
    assert bare.returncode == 1
    assert "no section 'name'" in bare.stderr


def test_spec_rejects_a_section_that_does_not_exist(tf) -> None:
    done = tf("spec", "dsl", "9.9")
    assert done.returncode == 1
    assert "no section '9.9'" in done.stderr
    assert "silu" in done.stderr


@pytest.mark.parametrize(
    ("topic", "section", "expected"),
    (
        ("target", "topology-levels", "Only `cta` MAY have a launch-provided"),
        ("core-ir", "target-inheritance", 'with `target="cuda"`'),
        ("core-ir", "default-step", "MUST have no default step"),
        ("cli", "check", "A FAIL with `--inputs random`"),
    ),
)
def test_spec_answers_askable_rules(
    tf, topic, section, expected
) -> None:
    done = tf("spec", topic, section)
    assert done.returncode == 0, done.stderr
    assert expected in done.stdout
