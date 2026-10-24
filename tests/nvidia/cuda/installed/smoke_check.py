"""`check` on models the installation never shipped."""

from __future__ import annotations

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
    assert all(name in done.stderr for name in ("'Expert'", "'w'", "'expert.w'"))


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
