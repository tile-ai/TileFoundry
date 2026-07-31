"""The model catalogue, and the shipped source answering as it ships."""

from __future__ import annotations

from pathlib import Path


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


def test_models_source_is_the_authored_file_byte_for_byte(tf, shipped) -> None:
    done = tf("models", "qwen3_1_7b", "--source")
    assert done.returncode == 0, done.stderr
    authored = (Path(shipped["models"]) / "qwen3_1_7b" / "model.py").read_text(
        encoding="utf-8"
    )
    assert done.stdout == authored


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

    analysed = tf("analyze", static, "--compute-cost")
    assert analysed.returncode == 0, analysed.stderr
    assert "target=cuda" in analysed.stdout
    assert "flops" in analysed.stdout and "traffic gmem=" in analysed.stdout

    scheduled = tf("schedule", static, "--topology", "cta")
    assert scheduled.returncode == 0, scheduled.stderr
    assert "partition cta x132 on nvidia.h200_sxm" in scheduled.stdout

    # A selector whose extent is stated at launch takes it on the command line.
    dynamic = f"{source}:Qwen3_1_7B_Decoder.layer0.self_attention"
    sized = tf("analyze", dynamic, "--compute-cost", "--dim", "ctx_len=1024")
    assert sized.returncode == 0, sized.stderr
    assert "flops" in sized.stdout
