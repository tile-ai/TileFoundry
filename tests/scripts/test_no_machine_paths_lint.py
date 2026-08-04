"""The machine-path checker, held to catching things and to not crying wolf.

A checker that only ever runs over a clean tree proves nothing: it passes, and so
would a checker that looks for nothing. So both directions are asserted from real
shapes -- the ones that have actually been committed by mistake, and the ones that
look similar and are fine.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "no_machine_paths_lint.py"


def _lint():
    """The checker, loaded from the script the hook runs."""
    spec = importlib.util.spec_from_file_location("no_machine_paths_lint", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lint():
    return _lint()


# The sample paths are assembled from fragments rather than written out, so this
# file states no machine path itself. The checker reads source text, so a literal
# here would make the guard report its own test data -- and marking those lines
# exempt would instead stop the checker being exercised on them at all. Assembling
# keeps both: the source is clean, the values fed in are the real thing.
_HOME = "/" + "home"
_USERS = "/" + "Users"
_SCRATCH = "/" + "data3" + "/shared"
_CONDA = "/miniconda3/envs/top/bin/python"

#: Shapes that have really been committed by mistake, one per kind.
CAUGHT = [
    f'CKPT_DIR = "{_SCRATCH}/someone/Qwen3.5-35B-A3B"',
    f'_PREPARED_DIR_CACHE = "{_SCRATCH}/someone/tf-prepared"',
    f"cd {_HOME}/someone/zqh/TileFoundry-poc-sched",
    f"PY={_HOME}/someone{_CONDA}",
    f"  `{_HOME}/someone{_CONDA}`",
    f'BASE = "{_USERS}/someone/envs/dev"',
    # The fallback form: configurable-looking, hardcoded for everyone else.
    f'os.environ.get("TF_CKPT", "{_SCRATCH}/someone/prepared")',
    # The same fallback in shell, which reached a shipped example unreported: the
    # `-` of `:-` sat where a token character would, so the whole default was
    # invisible to the guard that keeps this off the middle of a URL.
    f"CKPT=${{CKPT:-{_SCRATCH}/someone/Qwen3-1.7B}}",
    f"python run.py --ckpt={_HOME}/someone/models",
]

#: Shapes that resemble the above and are not machine-specific.
ALLOWED = [
    'url = "https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json"',
    'path = Path(__file__).parent / "model" / "decoder_layer.py"',
    'CKPT_DIR = os.environ["TILEFOUNDRY_QWEN35_CKPT"]',
    'shard = "model-00001-of-00001.safetensors"',
    "from tilefoundry.ir.types.shard import Layout",
    'doc = "/usr/share/doc"',
    f'note = "put it under {_HOME}/<you>/checkouts"',
    f"# see {_HOME}/someone/notes.md  # no-machine-path: allow",
]


@pytest.mark.parametrize("line", CAUGHT, ids=range(len(CAUGHT)))
def test_a_machine_specific_path_is_reported(lint, tmp_path, line) -> None:
    target = tmp_path / "leaky.py"
    target.write_text(line, encoding="utf-8")

    found = lint.findings(target)

    assert found, f"not reported: {line}"
    assert found[0][0] == 1


@pytest.mark.parametrize("line", ALLOWED, ids=range(len(ALLOWED)))
def test_a_path_that_only_looks_like_one_is_left_alone(lint, tmp_path, line) -> None:
    target = tmp_path / "fine.py"
    target.write_text(line, encoding="utf-8")

    assert lint.findings(target) == [], f"wrongly reported: {line}"


def test_the_exit_status_and_the_report_name_the_line(lint, tmp_path, capsys) -> None:
    """The hook's own contract: non-zero, and enough on stdout to go fix it."""
    target = tmp_path / "leaky.py"
    target.write_text(f'a = 1\nCKPT = "{_SCRATCH}/someone/prepared"\n', encoding="utf-8")

    status = lint.main([str(target)])

    assert status == 1
    out = capsys.readouterr().out
    assert f"{target}:2:" in out
    assert f"{_SCRATCH}/someone" in out


def test_a_clean_file_passes(lint, tmp_path, capsys) -> None:
    target = tmp_path / "fine.py"
    target.write_text("\n".join(ALLOWED), encoding="utf-8")

    assert lint.main([str(target)]) == 0
    assert capsys.readouterr().out == ""


def test_this_repository_is_clean(lint) -> None:
    """Every tracked file, so the guard is a fact about the tree and not only
    about the commit that happens to be in flight."""
    import subprocess  # noqa: PLC0415

    root = _SCRIPT.parent.parent
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.split("\0")

    offenders = {
        name: lint.findings(root / name)
        for name in tracked
        if name and (root / name).is_file()
    }
    assert not {k: v for k, v in offenders.items() if v}
