"""The shipped catalog still describes the corpus it was generated from.

A copy nobody checks goes stale. What the catalog is for is in
[cli.md](../../docs/spec/cli.md).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.generate_model_catalog import catalog
from tests.models.registry import MODELS

_SHIPPED = Path(__file__).parent / "catalog.json"


def test_the_shipped_catalog_matches_the_live_corpus() -> None:
    """Regenerating it changes nothing. Compared whole rather than field by field,
    so a part nobody thought to check cannot drift either."""
    assert json.loads(_SHIPPED.read_text(encoding="utf-8")) == catalog(), (
        "tests/models/catalog.json no longer describes the corpus; regenerate it "
        "with python -m scripts.generate_model_catalog"
    )


def test_the_catalog_describes_every_model_and_ranks_none() -> None:
    """Every model in the corpus is described, and nothing grades it.

    The second half is the point: a verification ranking carried in shipped data is
    a claim about tests that the shipped artifact cannot check, and one outlived the
    test it named. So no record may carry a level, an evidence string or an oracle
    identity, and this fails if one comes back.
    """
    shipped = json.loads(_SHIPPED.read_text(encoding="utf-8"))

    assert {model["name"] for model in shipped["models"]} == set(MODELS)
    assert set(shipped) == {"models"}
    for model in shipped["models"]:
        assert set(model) == {"name", "counts", "modules"}, model


def _tally(node: dict) -> tuple[int, int]:
    """A node's leaf Modules and functions, counting a run of N as N of them."""
    leaves = functions = 0
    for child in node["modules"]:
        child_leaves, child_functions = _tally(child)
        leaves += child_leaves
        functions += child_functions
    if not node["modules"]:
        leaves += 1
    functions += len(node["functions"])
    return len(node["names"]) * leaves, len(node["names"]) * functions


def test_the_counts_are_the_tree() -> None:
    """Recounted from the shipped forest, so counts and forest cannot disagree.

    Against the shipped tree rather than the corpus: one wrong traversal would
    regenerate both halves alike and pass a comparison with it.
    """
    shipped = json.loads(_SHIPPED.read_text(encoding="utf-8"))

    for model in shipped["models"]:
        leaves = functions = 0
        for root in model["modules"]:
            root_leaves, root_functions = _tally(root)
            leaves += root_leaves
            functions += root_functions
        assert (leaves, functions) == (
            model["counts"]["leaf_modules"], model["counts"]["functions"]
        ), model["name"]


def test_a_collapsed_run_names_every_module_it_stands_for() -> None:
    """`qwen3_1_7b`'s 28 layers are one entry carrying all 28 names, in order."""
    shipped = json.loads(_SHIPPED.read_text(encoding="utf-8"))
    model = next(m for m in shipped["models"] if m["name"] == "qwen3_1_7b")

    runs = []
    pending = list(model["modules"])
    while pending:
        node = pending.pop()
        pending.extend(node["modules"])
        if len(node["names"]) > 1:
            runs.append(node["names"])

    assert len(runs) == 1
    assert runs[0] == [f"layer{index}" for index in range(28)]


def test_the_documented_command_reports_the_catalog_as_current() -> None:
    """`--check` passes when run as documented; an import would not prove that."""
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.generate_model_catalog", "--check"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
