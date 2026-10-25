"""`check` on models the installation never shipped."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from tests.core.cli._check import MESH_CHECK_ARGS

_ARGS = (
    "--inputs",
    "random",
    "--weights",
    "random",
    "--out",
    "output",
    "--fn",
    "allclose",
    "--atol",
    "1e-6",
    "--rtol",
    "1e-6",
)


def test_check_reports_grid_loop_errors(tf) -> None:
    for name, message in (
        ("single", "tf.tile(extent) is not supported; use range(extent)"),
        ("keyword", "positional-only at the IR level"),
    ):
        source = Path(__file__).parents[1] / f"fixtures/diagnostics/parser_tile_{name}.py"

        done = tf("check", f"{source}:Bad", *_ARGS)

        assert done.returncode == 1
        assert message in done.stderr


def test_check_help_explains_positions(tf) -> None:
    done = tf("check", "--help")
    assert done.returncode == 0, done.stderr

    assert "--inputs random|files:A.pt,B.pt" in done.stdout
    assert "declared order" in done.stdout
    assert "--out OUTPUT" in done.stdout
    assert "`output[0]`" in done.stdout
    assert "return order" in done.stdout
    assert "positions, not the names your code gives them" in done.stdout
    assert done.stderr == ""


def test_twin_compared_against_stated_module(
    tf, square_twin, fused_twin
) -> None:
    faithful = tf(
        "check",
        f"{square_twin}:Twin.main",
        *_ARGS,
        "--fn",
        "cosine",
        "--min",
        "0.9999",
    )
    assert faithful.returncode == 0, faithful.stderr
    assert "reference: evaluator on Model.main" in faithful.stdout
    assert "max_violation 0" in faithful.stdout

    drifted = tf("check", f"{square_twin}:Drifted.main", *_ARGS)
    assert drifted.returncode == 1
    assert "FAIL" in drifted.stdout

    fused = tf("check", f"{fused_twin}:FusedTwin.fused", *_ARGS)
    assert fused.returncode == 0, fused.stderr
    assert "reference: evaluator on Fused.fused" in fused.stdout


def test_module_checked_against_expected_file(
    tf, square_twin, tmp_path
) -> None:
    activation = torch.arange(168, dtype=torch.float32)
    torch.save(activation, tmp_path / "x.pt")
    torch.save(activation * activation, tmp_path / "expected.pt")

    done = tf(
        "check",
        f"{square_twin}:Model",
        "--inputs",
        f"files:{tmp_path / 'x.pt'}",
        "--weights",
        "random",
        "--expected",
        str(tmp_path / "expected.pt"),
        "--out",
        "output",
        "--fn",
        "equal",
    )
    assert done.returncode == 0, done.stderr

    assert "expected.pt" in done.stdout
    assert "elements 168" in done.stdout


def test_check_refuses_orchestration_method(tf, orchestrated_twin) -> None:
    for target in ("Orchestrated.forward", "OrchestratedTwin.forward"):
        done = tf(
            "check",
            f"{orchestrated_twin}:{target}",
            *MESH_CHECK_ARGS,
        )
        assert done.returncode == 1
        assert "not orchestration method Orchestrated.forward" in done.stderr
        assert "select one of its HIR functions instead" in done.stderr


def test_non_tensor_leaf_names_position(
    tf, orchestrated_twin, tmp_path
) -> None:
    torch.save(torch.arange(168, dtype=torch.float32), tmp_path / "hidden.pt")
    torch.save(
        (torch.ones(168), ("not a tensor", torch.ones(168)), torch.ones(168), torch.ones(168)),
        tmp_path / "mixer_args.pt",
    )

    done = tf(
        "check",
        f"{orchestrated_twin}:OrchestratedTwin.add_pair",
        "--inputs",
        f"files:{tmp_path / 'hidden.pt'},{tmp_path / 'mixer_args.pt'}",
        "--weights",
        "random",
        "--out",
        "output[0]",
        "--fn",
        "equal",
        "--out",
        "output[1]",
        "--fn",
        "equal",
    )
    assert done.returncode == 1
    assert "mixer_args.pt[1][0]" in done.stderr


def test_orchestration_method_names_input_files(tf, orchestrated_twin) -> None:
    done = tf(
        "check",
        f"{orchestrated_twin}:OrchestratedTwin",
        *MESH_CHECK_ARGS,
    )
    assert done.returncode == 1
    assert "orchestration method" in done.stderr
    assert "Orchestrated.forward" in done.stderr
    assert "add_pair, affine_pair" in done.stderr


def test_check_requires_inputs(tf, square_twin) -> None:
    done = tf(
        "check",
        f"{square_twin}:Twin.main",
        "--weights",
        "random",
        "--out",
        "output",
        "--fn",
        "nan_inf",
    )
    assert done.returncode == 1
    assert "no inputs stated" in done.stderr


def test_zero_sides_match(tf, square_twin) -> None:
    done = tf(
        "check",
        f"{square_twin}:Twin.zeroed",
        "--inputs",
        "random",
        "--weights",
        "random",
        "--out",
        "output",
        "--fn",
        "cosine",
        "--min",
        "0.999",
        "--fn",
        "rel_l2",
        "--max",
        "1e-6",
    )
    assert done.returncode == 0, done.stderr

    assert "cosine 1" in done.stdout
    assert "both sides are entirely zero" in done.stdout
    assert "ref_norm 0" in done.stdout
    assert "PASS" in done.stdout


def test_check_reads_checkpoint_weights(
    tf, weighted_twin, tmp_path
) -> None:
    save_file({"w": torch.linspace(0.5, 2.0, 168)}, str(tmp_path / "model.safetensors"))

    done = tf(
        "check",
        f"{weighted_twin}:WeightedRootTwin.scaled",
        "--inputs",
        "random",
        "--weights",
        f"ckpt:{tmp_path}",
        "--out",
        "output",
        "--fn",
        "allclose",
        "--atol",
        "1e-6",
        "--rtol",
        "1e-6",
    )
    assert done.returncode == 0, done.stderr

    assert "random (seed " in done.stdout
    assert "activations actual f32 (declared f32)" in done.stdout
    assert "max_violation 0" in done.stdout


def test_unlinked_runtime_module_refused(
    tf, handwritten_twin, mislabelled_twin
) -> None:
    for source, target, refused in (
        (handwritten_twin, "Handwritten", "names no authored Module"),
        (mislabelled_twin, "Mislabelled", "module must be Module or None, got str"),
    ):
        done = tf(
            "check",
            f"{source}:{target}",
            *MESH_CHECK_ARGS,
        )
        assert done.returncode == 1
        assert refused in done.stderr


def test_check_reports_json_verdict(tf, mine, tmp_path) -> None:
    done = tf("check", f"{mine}:MineTwin.main", *_ARGS, "--json", str(tmp_path / "report.json"))
    assert done.returncode == 0, done.stderr
    assert done.stdout == ""
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["target"].endswith("runtime_model.py:MineTwin.main")
    assert payload["runs"][0]["inputs"] == {
        "activations": {
            "source": "random (seed 0)",
            "actual_dtypes": ["f32"],
            "declared_dtypes": ["f32"],
            "files": [],
        },
    }


def test_check_refuses_unknown_criterion(tf, mine) -> None:
    done = tf(
        "check", f"{mine}:MineTwin.main", "--inputs", "random", "--weights", "random",
        "--out", "output", "--fn", "nope"
    )
    assert done.returncode != 0
    assert "nope" in done.stderr


def test_check_names_missing_sibling(tf, mine) -> None:
    (mine.parent / "model.py").unlink()
    done = tf("check", f"{mine}:MineTwin.main", *_ARGS)
    assert done.returncode == 1
    assert "model" in done.stderr
