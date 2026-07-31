"""`check` on models the installation never shipped."""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

_ARGS = ("--inputs", "random", "--out", "output", "--fn", "allclose",
         "--atol", "1e-6", "--rtol", "1e-6")


def test_check_help_explains_input_and_output_positions(tf) -> None:
    done = tf("check", "--help")
    assert done.returncode == 0, done.stderr

    assert "--input PATH" in done.stdout
    assert "parameter's declared order" in done.stdout
    assert "--out OUTPUT" in done.stdout
    assert "`output[0]`" in done.stdout
    assert "return order" in done.stdout
    assert "positions, not the names your code gives them" in done.stdout
    assert done.stderr == ""


def test_a_twin_is_compared_against_the_module_it_states(tf, twin) -> None:
    faithful = tf(
        "check", f"{twin}:Twin.main", *_ARGS, "--fn", "cosine", "--min", "0.9999",
    )
    assert faithful.returncode == 0, faithful.stderr
    assert "reference: evaluator on Model.main" in faithful.stdout
    assert "max_violation 0" in faithful.stdout

    drifted = tf("check", f"{twin}:Drifted.main", *_ARGS)
    assert drifted.returncode == 1
    assert "FAIL" in drifted.stdout

    fused = tf("check", f"{twin}:FusedTwin.fused", *_ARGS)
    assert fused.returncode == 0, fused.stderr
    assert "reference: evaluator on Fused.fused" in fused.stdout


def test_a_whole_module_is_checked_against_an_expected_output_file(
    tf, twin, tmp_path
) -> None:
    activation = torch.arange(168, dtype=torch.float32)
    torch.save(activation, tmp_path / "x.pt")
    torch.save(activation * activation, tmp_path / "expected.pt")

    done = tf(
        "check", f"{twin}:Model",
        "--input", str(tmp_path / "x.pt"),
        "--expected", str(tmp_path / "expected.pt"),
        "--out", "output", "--fn", "equal",
    )
    assert done.returncode == 0, done.stderr

    assert "expected.pt" in done.stdout
    assert "elements 168" in done.stdout


def test_the_inputs_are_exactly_one_form(tf, twin, tmp_path) -> None:
    torch.save(torch.arange(168, dtype=torch.float32), tmp_path / "x.pt")

    for form in ("random", "real"):
        done = tf(
            "check", f"{twin}:Twin.main", "--input", str(tmp_path / "x.pt"),
            "--inputs", form, "--out", "output", "--fn", "nan_inf",
        )
        assert done.returncode == 1
        assert "give exactly one form" in done.stderr


def test_two_entirely_zero_sides_are_a_match_not_a_total_mismatch(tf, twin) -> None:
    done = tf(
        "check", f"{twin}:Twin.zeroed", "--inputs", "random", "--out", "output",
        "--fn", "cosine", "--min", "0.999", "--fn", "rel_l2", "--max", "1e-6",
    )
    assert done.returncode == 0, done.stderr

    assert "cosine 1" in done.stdout
    assert "both sides are entirely zero" in done.stdout
    assert "ref_norm 0" in done.stdout
    assert "PASS" in done.stdout


def test_real_weights_come_from_the_checkpoint_and_activations_are_drawn(
    tf, twin, tmp_path
) -> None:
    save_file(
        {"w": torch.linspace(0.5, 2.0, 168)}, str(tmp_path / "model.safetensors")
    )

    done = tf(
        "check", f"{twin}:WeightedRootTwin.scaled", "--inputs", "real",
        "--ckpt", str(tmp_path), "--out", "output", "--fn", "allclose",
        "--atol", "1e-6", "--rtol", "1e-6",
    )
    assert done.returncode == 0, done.stderr

    assert "weights the checkpoint" in done.stdout
    assert "random, seed " in done.stdout
    assert "max_violation 0" in done.stdout

    refused = tf("check", f"{twin}:WeightedRootTwin.scaled", "--inputs", "real", *_ARGS[2:])
    assert refused.returncode == 1
    assert "needs --ckpt DIR" in refused.stderr


def test_a_nested_twin_is_reached_through_the_child_it_is_declared_under(
    tf, twin
) -> None:
    done = tf("check", f"{twin}:NestedTwin.child.scaled", *_ARGS)
    assert done.returncode == 0, done.stderr

    assert "reference: evaluator on child.scaled" in done.stdout
    assert "max_violation 0" in done.stdout


def test_a_runtime_module_that_names_no_authored_module_is_refused(tf, twin) -> None:
    for target, refused in (
        ("Handwritten", "names no authored Module"),
        ("Mislabelled", "module must be Module or None, got str"),
    ):
        done = tf(
            "check", f"{twin}:{target}", "--inputs", "random",
            "--out", "output", "--fn", "nan_inf",
        )
        assert done.returncode == 1
        assert refused in done.stderr


def test_check_reports_the_same_verdict_as_json(tf, mine) -> None:
    done = tf("check", f"{mine}:MineTwin.main", *_ARGS, "--json")
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    assert payload["passed"] is True
    assert payload["target"].endswith("runtime_model.py:MineTwin.main")


def test_check_refuses_a_criterion_it_does_not_have(tf, mine) -> None:
    done = tf("check", f"{mine}:MineTwin.main", "--inputs", "random",
              "--out", "output", "--fn", "nope")
    assert done.returncode != 0
    assert "nope" in done.stderr


def test_check_names_the_sibling_it_could_not_import(tf, mine) -> None:
    (mine.parent / "model.py").unlink()
    done = tf("check", f"{mine}:MineTwin.main", *_ARGS)
    assert done.returncode == 1
    assert "model" in done.stderr
