"""The tutorial pages, and the shipped source they splice from."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_the_overview_names_the_pages_and_the_commands_it_delegates_to(tf) -> None:
    done = tf("tutorial")
    assert done.returncode == 0, done.stderr
    assert "source to source" in done.stdout
    for page in ("migrate", "optimize"):
        assert page in done.stdout, page
    assert "tilefoundry spec" in done.stdout
    assert "tilefoundry check --help" in done.stdout


@pytest.mark.parametrize("page", ("migrate", "optimize"))
def test_each_page_renders_from_the_installation(tf, page) -> None:
    """A rendered page has no unresolved source directive."""
    done = tf("tutorial", page)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("# ")
    assert "{{fixture:" not in done.stdout


def test_optimize_tells_the_reader_to_copy_the_shipped_model_directory(tf) -> None:
    done = tf("tutorial", "optimize")
    assert done.returncode == 0, done.stderr
    assert 'source=$(tilefoundry models qwen3_5_35b_a3b --source | sed -n \'1p\')' in done.stdout
    assert 'cp -r "$source" mine' in done.stdout
    assert "tilefoundry check mine/model.py:MyFused.fused" in done.stdout


def test_migrate_splices_the_shipped_model_source_verbatim(tf, shipped) -> None:
    """A parameter only the packaged source declares proves which file was read."""
    done = tf("tutorial", "migrate")
    assert done.returncode == 0, done.stderr
    assert "w_router: ConstTensor" in done.stdout

    authored = (Path(shipped["models"]) / "qwen3_5_35b_a3b" / "model.py").read_text(
        encoding="utf-8"
    )
    assert "w_router: ConstTensor" in authored


def test_orchestrator_lists_and_describes_its_shipped_family(tf) -> None:
    listing = tf("tutorial", "orchestrator")
    assert listing.returncode == 0, listing.stderr
    assert (
        "causal_lm  Autoregressive decode: one token per step; the caller owns the state."
        in listing.stdout
    )

    detail = tf("tutorial", "orchestrator", "causal_lm")
    assert detail.returncode == 0, detail.stderr
    lines = detail.stdout.splitlines()
    assert Path(lines[0]).is_absolute()
    assert lines[0].endswith("/orchestrator/causal_lm")
    assert lines[1:] == [
        "generation.py  Autoregressive decode: one token per step; the caller owns the state.",
        "run.py         Run a shipped causal-LM source directory against its published checkpoint.",
    ]


def test_unknown_orchestrator_family_names_the_available_families(tf) -> None:
    done = tf("tutorial", "orchestrator", "missing")
    assert done.returncode == 1
    assert "no orchestrator family 'missing'" in done.stderr
    assert "causal_lm" in done.stderr
