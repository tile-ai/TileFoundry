"""The model catalogue, and the shipped source answering as it ships."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from tilefoundry.cli import data, models

REPO = Path(__file__).resolve().parents[3]


def _listed_names(output: str) -> set[str]:
    return {line.split(maxsplit=1)[0] for line in output.splitlines()[1:]}


def test_models_separates_oracles_from_everything_else(tf) -> None:
    done = tf("models")
    assert done.returncode == 0, done.stderr
    listed = done.stdout

    assert "usable as an oracle" in listed and "not usable as an oracle" in listed
    for name in ("qwen3_1_7b", "kimi_linear_48b_a3b"):
        assert name in listed
    assert "L1" in listed and "L2" in listed and "L3" in listed


def test_models_renders_the_whole_forest_with_leaf_modules_marked(tf) -> None:
    done = tf("models", "qwen3_1_7b")
    assert done.returncode == 0, done.stderr
    forest = done.stdout

    assert "28 leaf modules, 115 functions" in forest
    assert "  Qwen3_1_7B_Decoder\n" in forest
    assert "* Qwen3_1_7B\n" not in forest
    assert "*   layer0..layer27  (28 identical, each as shown)" in forest
    assert "layer1\n" not in forest
    assert "input_rms_norm(hidden: Tensor[(1, 1, 2048), \"bf16\"]" in forest


def test_models_source_names_the_shipped_directory_and_its_files(tf, shipped, tmp_path) -> None:
    done = tf("models", "qwen3_1_7b", "--source")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    source = Path(lines[0])
    assert source.is_absolute()
    assert source == Path(shipped["models"]) / "qwen3_1_7b"
    assert _listed_names(done.stdout) == {
        entry.name for entry in source.iterdir() if entry.is_file()
    }
    assert any(
        line.startswith("model.py")
        and "Qwen3-1.7B's dense decoder layer and the stack that closes it" in line
        for line in lines[1:]
    )
    assert any(line.startswith("config.json") and line.endswith("-") for line in lines[1:])

    copied = tmp_path / "mine"
    shutil.copytree(source, copied)
    static = f"{copied / 'model.py'}:Qwen3_1_7B_Decoder.layer0.mlp"
    analysed = tf("analyze", static, "--compute-cost")
    assert analysed.returncode == 0, analysed.stderr
    assert "target=cuda" in analysed.stdout
    assert "flops" in analysed.stdout and "traffic gmem=" in analysed.stdout

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    checkout = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tilefoundry.cli import main; raise SystemExit(main())",
            "models",
            "qwen3_1_7b",
            "--source",
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checkout.returncode == 0, checkout.stderr
    assert _listed_names(checkout.stdout) == _listed_names(done.stdout)


def test_models_source_follows_the_manifest_without_importing_files(
    tmp_path, monkeypatch, capsys
) -> None:
    root = tmp_path / "checkout"
    model = root / "tests" / "models" / "added"
    model.mkdir(parents=True)
    (model / "model.py").write_text(
        '"""A model that must not run while it is described."""\nraise RuntimeError\n',
        encoding="utf-8",
    )
    (model / "run.py").write_text('"""An added companion."""\n', encoding="utf-8")
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model.parent / "catalog.json").write_text(
        '{"models": [{"name": "added"}]}\n', encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        """[tool.setuptools.data-files]
"share/tilefoundry/models/added" = [
    "tests/models/added/model.py",
    "tests/models/added/run.py",
    "tests/models/added/config.json",
]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(data, "_REPOSITORY_ROOT", root)

    assert models.run_models("added", source=True) == 0
    output = capsys.readouterr().out
    assert _listed_names(output) == {"model.py", "run.py", "config.json"}
    assert "A model that must not run while it is described." in output
    assert "An added companion." in output
    assert "config.json  -" in output


def test_models_rejects_a_name_the_catalog_does_not_have(tf) -> None:
    done = tf("models", "nope")
    assert done.returncode == 1
    assert "no model named 'nope'" in done.stderr
    assert "qwen3_1_7b" in done.stderr


def test_the_shipped_source_answers_the_public_commands_as_it_ships(
    tf, shipped
) -> None:
    """No editing step: the root declares its machine, so the commands answer."""
    source = Path(shipped["models"]) / "qwen3_1_7b" / "model.py"
    static = f"{source}:Qwen3_1_7B_Decoder.layer0.mlp"

    scheduled = tf("schedule", static, "--topology", "cta")
    assert scheduled.returncode == 0, scheduled.stderr
    assert "partition cta x132 on nvidia.h200_sxm" in scheduled.stdout

    threaded = tf("schedule", static, "--topology", "thread")
    assert threaded.returncode == 0, threaded.stderr
    assert "pipeline schedule" in threaded.stdout

    # A selector whose extent is stated at launch takes it on the command line.
    dynamic = f"{source}:Qwen3_1_7B_Decoder.layer0.self_attention"
    sized = tf("analyze", dynamic, "--compute-cost", "--dim", "ctx_len=1024")
    assert sized.returncode == 0, sized.stderr
    assert "flops" in sized.stdout
