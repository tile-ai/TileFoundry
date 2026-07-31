"""The tutorial pages, and the shipped source they splice from."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_the_overview_names_the_pages_and_the_commands_it_delegates_to(tf) -> None:
    done = tf("tutorial")
    assert done.returncode == 0, done.stderr
    assert "source to source" in done.stdout
    for page in ("migrate", "run", "optimize"):
        assert page in done.stdout, page
    assert "tilefoundry spec" in done.stdout
    assert "tilefoundry check --help" in done.stdout


@pytest.mark.parametrize("page", ("migrate", "run", "optimize"))
def test_each_page_renders_from_the_installation(tf, page) -> None:
    """A rendered page has no unresolved source directive."""
    done = tf("tutorial", page)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("# ")
    assert "{{fixture:" not in done.stdout


def test_migrate_splices_the_shipped_model_source_verbatim(tf, shipped) -> None:
    """A parameter only the packaged source declares proves which file was read."""
    done = tf("tutorial", "migrate")
    assert done.returncode == 0, done.stderr
    assert "w_router: ConstTensor" in done.stdout

    authored = (Path(shipped["models"]) / "qwen3_5_35b_a3b" / "model.py").read_text(
        encoding="utf-8"
    )
    assert "w_router: ConstTensor" in authored
