"""`check` on models the installation never shipped."""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

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

_MESH_CHECK_ARGS = (
    "--inputs", "random", "--weights", "random", "--out", "output", "--fn", "nan_inf"
)

_JUDGE = ("--inputs", "random", "--out", "output", "--fn", "nan_inf")


def test_check_on_a_leaf_does_not_materialise_its_siblings_weights(tf, leaf_weights):
    """A leaf check must not allocate its siblings' 96 GiB weight union."""
    done = tf(
        "check",
        f"{leaf_weights}:Mod.leaf",
        "--inputs",
        "random",
        "--weights",
        "random",
        "--out",
        "output",
        "--fn",
        "nan_inf",
    )
    assert done.returncode == 0, done.stderr


def test_check_specialises_through_a_dispatching_callee(tf, specialize_through_call):
    """Check dispatches from the actual inputs instead of rebuilding the call tree."""
    done = tf(
        "check",
        f"{specialize_through_call}:ToCallee",
        "--inputs",
        "random",
        "--weights",
        "random",
        "--dim",
        "n=64",
        "--out",
        "output",
        "--fn",
        "nan_inf",
    )
    assert done.returncode == 0, done.stderr


def test_analyze_specialises_through_a_dispatching_callee(
    tf, specialize_through_call, tmp_path
):
    """Analyze picks the callee's variant from --dim instead of refusing the rebuild."""
    costs = {}
    for extent in ("n=64", "n=512"):
        report = tmp_path / f"{extent}.md"
        done = tf(
            "analyze",
            f"{specialize_through_call}:ToCallee",
            str(report),
            "--dim",
            extent,
            "--compute-cost",
        )
        assert done.returncode == 0, done.stderr
        assert "out of memory" not in done.stderr
        costs[extent] = next(
            line
            for line in report.read_text().splitlines()
            if line.startswith("# compute-cost")
        )
    assert costs["n=64"] != costs["n=512"]


def test_weights_are_needed_only_where_one_is_reached(tf, square_cpu, hir_composition):
    """A run reaching no weight needs no source; a reached weight reads from one."""
    for extra in ((), ("--weights", "random")):
        done = tf("check", f"{square_cpu}:Mine", *_JUDGE, *extra)
        assert done.returncode == 0, done.stderr

    done = tf(
        "check",
        f"{hir_composition}:CrossModule",
        *_JUDGE,
        "--weights",
        "random",
    )
    assert done.returncode == 0, done.stderr


def test_a_reached_weight_with_no_source_is_named_where_it_is_first_asked_for(
    tf, hir_composition
):
    """The first use names a missing child weight and the Module declaring it."""
    done = tf("check", f"{hir_composition}:CrossModule", *_JUDGE)
    assert done.returncode != 0
    assert "'expert'" in done.stderr and "'w'" in done.stderr


def test_check_reports_grid_loop_parser_errors_from_the_installed_wheel(
    tf, tmp_path
) -> None:
    for name, loop, message in (
        ("single", "tile(8)", "tile(extent) is not supported; use range(extent)"),
        ("keyword", "tile(8, step=2)", "positional-only at the IR level"),
    ):
        source = tmp_path / f"{name}.py"
        source.write_text(
            "from tilefoundry import module\n"
            "from tilefoundry.dsl import Tensor, func, tf\n\n"
            '@module(entry="main")\n'
            "class Bad:\n"
            "    @func\n"
            '    def main(x: Tensor[(8,), "f32"]):\n'
            f"        for i in {loop}:\n"
            "            y = tf.relu(x)\n",
            encoding="utf-8",
        )

        done = tf("check", f"{source}:Bad", *_ARGS)

        assert done.returncode == 1
        assert message in done.stderr


def test_check_points_at_the_line_when_the_program_is_wrong(
    tf, mesh_slice_start
) -> None:
    done = tf("check", f"{mesh_slice_start}:OutOfWindow", *_MESH_CHECK_ARGS)

    assert done.returncode == 1
    assert "op=Slice" in done.stderr
    assert "mesh_slice_start.py:" in done.stderr
    assert "Slice window exceeds axis 1" in done.stderr
    assert "not modelled" not in done.stderr


def test_check_never_leaks_a_backend_error(tf, mesh_slice_start) -> None:
    done = tf("check", f"{mesh_slice_start}:Strided", *_MESH_CHECK_ARGS)

    assert "invalid for input of size" not in done.stderr
    assert done.stderr == "" or "mesh_slice_start.py:" in done.stderr


def test_check_runs_a_mesh_program_that_reads_no_coordinate(tf, mesh_slice_start) -> None:
    done = tf("check", f"{mesh_slice_start}:Fixed", *_MESH_CHECK_ARGS)
    assert done.returncode == 0, done.stderr


def test_check_help_explains_input_and_output_positions(tf) -> None:
    done = tf("check", "--help")
    assert done.returncode == 0, done.stderr

    assert "--inputs random|files:A.pt,B.pt" in done.stdout
    assert "declared order" in done.stdout
    assert "--out OUTPUT" in done.stdout
    assert "`output[0]`" in done.stdout
    assert "return order" in done.stdout
    assert "positions, not the names your code gives them" in done.stdout
    assert done.stderr == ""


def test_a_twin_is_compared_against_the_module_it_states(
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


def test_a_whole_module_is_checked_against_an_expected_output_file(
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


def test_check_refuses_a_named_orchestration_method_on_both_sides(
    tf, orchestrated_twin
) -> None:
    for target in ("Orchestrated.forward", "OrchestratedTwin.forward"):
        done = tf(
            "check",
            f"{orchestrated_twin}:{target}",
            "--inputs",
            "random",
            "--weights",
            "random",
            "--out",
            "output",
            "--fn",
            "nan_inf",
        )
        assert done.returncode == 1
        assert "not orchestration method Orchestrated.forward" in done.stderr
        assert "select one of its HIR functions instead" in done.stderr


def test_a_non_tensor_nested_activation_leaf_names_its_position(
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


def test_an_orchestration_method_names_the_files_its_inputs_need(tf, orchestrated_twin) -> None:
    done = tf(
        "check",
        f"{orchestrated_twin}:OrchestratedTwin",
        "--inputs",
        "random",
        "--weights",
        "random",
        "--out",
        "output",
        "--fn",
        "nan_inf",
    )
    assert done.returncode == 1
    assert "orchestration method" in done.stderr
    assert "Orchestrated.forward" in done.stderr
    assert "add_pair, affine_pair" in done.stderr


def test_inputs_are_required(tf, square_twin) -> None:
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


def test_two_entirely_zero_sides_are_a_match_not_a_total_mismatch(tf, square_twin) -> None:
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


def test_real_weights_come_from_the_checkpoint_and_activations_are_drawn(
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


def test_a_nested_twin_is_reached_through_the_child_it_is_declared_under(tf, nested_twin) -> None:
    done = tf("check", f"{nested_twin}:NestedTwin.child.scaled", *_ARGS)
    assert done.returncode == 0, done.stderr

    assert "reference: evaluator on child.scaled" in done.stdout
    assert "max_violation 0" in done.stdout


def test_a_runtime_module_that_names_no_authored_module_is_refused(
    tf, handwritten_twin, mislabelled_twin
) -> None:
    for source, target, refused in (
        (handwritten_twin, "Handwritten", "names no authored Module"),
        (mislabelled_twin, "Mislabelled", "module must be Module or None, got str"),
    ):
        done = tf(
            "check",
            f"{source}:{target}",
            "--inputs",
            "random",
            "--weights",
            "random",
            "--out",
            "output",
            "--fn",
            "nan_inf",
        )
        assert done.returncode == 1
        assert refused in done.stderr


def test_check_reports_the_same_verdict_as_json(tf, mine, tmp_path) -> None:
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


def test_check_refuses_a_criterion_it_does_not_have(tf, mine) -> None:
    done = tf(
        "check", f"{mine}:MineTwin.main", "--inputs", "random", "--weights", "random",
        "--out", "output", "--fn", "nope"
    )
    assert done.returncode != 0
    assert "nope" in done.stderr


def test_check_names_the_sibling_it_could_not_import(tf, mine) -> None:
    (mine.parent / "model.py").unlink()
    done = tf("check", f"{mine}:MineTwin.main", *_ARGS)
    assert done.returncode == 1
    assert "model" in done.stderr
