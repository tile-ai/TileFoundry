"""The tutorial pages, reproduced the way a reader reproduces them."""

from __future__ import annotations

import difflib
import os
import re
import subprocess
from pathlib import Path

import pytest

_FENCE = re.compile(
    r"^```(?P<language>[A-Za-z0-9_-]*)\n(?P<body>.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_OWNED = re.compile(r"<!-- tilefoundry-source: [^\n]+ -->\n\n\Z")
_SHOWN = "```text\n"


def _asked_of_the_reader(text: str) -> list[re.Match[str]]:
    """The page's fenced blocks a reader is told to run, in page order.

    A block under a source tag is not one of them: it is a program the page's
    own extraction command writes to a file, and the commands that follow run
    against that file rather than against the block.
    """
    return [
        match
        for match in _FENCE.finditer(text)
        if match.group("language") in {"bash", "python"}
        and not _OWNED.search(text[: match.start()])
    ]


def _difference(shown: str, ahead: str) -> str:
    """A readable account of where the page and this run parted."""
    lines = ahead.splitlines()[: len(shown.splitlines()) + 4]
    return "\n".join(
        difflib.unified_diff(
            shown.splitlines(), lines, fromfile="produced", tofile="page", lineterm=""
        )
    )


def _reproduce(match: re.Match[str], region: str, *, python: Path, cwd: Path) -> None:
    """Run one block the reader's way and hold the page to what it produced.

    What the page shows must be exactly what came out, and it must sit directly
    after the block: a stale number, a dropped line, or a command whose flags
    moved all part the two. Prose after the output is not checked -- a page is
    free to write a ``text`` block of its own, and telling that apart from an
    output needs the notebook, which a reader does not have.
    """
    language = match.group("language")
    block = match.group("body").rstrip("\n")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PATH"] = f"{python.parent}{os.pathsep}{environment['PATH']}"
    argv = ["bash", "-c", block] if language == "bash" else [str(python), "-c", block]
    done = subprocess.run(
        argv, cwd=str(cwd), env=environment, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, f"{block}\n{done.stdout}{done.stderr}"
    assert not done.stderr, f"{block}\n{done.stderr}"

    ahead = region.lstrip("\n")
    produced = done.stdout.rstrip()
    if not produced:
        assert not ahead.startswith(_SHOWN), f"{block}\nthe page shows output:\n{ahead[:400]}"
        return
    shown = f"{_SHOWN}{produced}\n```" if ahead.startswith(_SHOWN) else produced
    assert ahead.startswith(shown), f"{block}\n{_difference(shown, ahead)}"


NOTEBOOK_PAGES = ("migrate", "optimize", "authoring")


def test_the_index_names_the_pages_there_are(tf) -> None:
    """The index says what the project is and which page answers what."""
    done = tf("tutorial")
    assert done.returncode == 0, done.stderr
    assert "source to source" in done.stdout
    for page in NOTEBOOK_PAGES:
        assert page in done.stdout, page


@pytest.mark.parametrize("page", ("migrate", "optimize"))
def test_a_workflow_page_hands_the_reference_questions_on(tf, page) -> None:
    """The tutorial MUST point at `spec` and `check --help`; the pages are where.

    The index lists the pages it has and nothing else, so this obligation lands on
    each page that teaches a step rather than on the front door.
    """
    done = tf("tutorial", page)
    assert done.returncode == 0, done.stderr
    assert "tilefoundry spec" in done.stdout
    assert "tilefoundry check --help" in done.stdout


@pytest.mark.parametrize("page", NOTEBOOK_PAGES)
def test_each_page_renders_from_the_installation(tf, page) -> None:
    done = tf("tutorial", page)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("# ")


@pytest.mark.parametrize("page", NOTEBOOK_PAGES)
def test_a_reader_reproduces_the_page_from_the_page(installation, tf, page, tmp_path) -> None:
    """Do what the page tells a reader to do; the outputs must be the page's own.

    Stronger than asserting the page contains some string, and it runs where the
    reader is: an installation, with no checkout to import and no ``scripts/`` to
    fall back on. The first block extracts the programs from the page itself, so
    every command after it is measuring the source the page displays.
    """
    text = tf("tutorial", page).stdout
    (tmp_path / f"{page}.md").write_text(text, encoding="utf-8")
    blocks = _asked_of_the_reader(text)
    assert blocks, f"{page}: the page asks the reader to run nothing"
    for index, match in enumerate(blocks):
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(text)
        _reproduce(
            match,
            text[match.end() : end],
            python=installation / "bin" / "python",
            cwd=tmp_path,
        )


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
