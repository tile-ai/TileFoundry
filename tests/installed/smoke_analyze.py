"""`analyze` against a file of one's own."""

from __future__ import annotations

import json
from pathlib import Path

_BAD_MODULE = """
from tilefoundry import module
from tilefoundry.dsl import Tensor, func, tf

@module(entry="main")
class Bad:
    @func
    def main(x: Tensor[(8,), "f32"]):
        wrong = tf.add(x, tf.cast(x, "i32"))
        return wrong
"""

_OPEN_MODULE = '''
from tilefoundry import module
from tilefoundry.dsl import DimVar, Tensor, Topology, func, tf
from tilefoundry.target import CudaTarget

N = DimVar("N", 1, 9)

@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class Open:
    @func
    def main(x: Tensor[(N,), "f32"]):
        return tf.add(x, x)
'''


def test_logical_analyses_run_and_performance_requires_an_execution_domain(tf, cmine) -> None:
    done = tf(
        "analyze",
        f"{cmine}:CMine.root",
        "--compute-cost",
        "--memory",
        "--roofline",
    )
    assert done.returncode == 0, done.stderr
    for conclusion in ("# compute-cost flops=", "# peak-footprint=", "# roofline ideal-ns="):
        assert conclusion in done.stdout, conclusion

    rejected = tf("analyze", f"{cmine}:CMine.root", "--performance")
    assert rejected.returncode == 1
    assert "performance:" in rejected.stderr
    assert "has no cta execution domain" in rejected.stderr


def test_mega_kernel_reports_four_families_on_one_expanded_program(tf) -> None:
    source = Path(__file__).resolve().parents[1] / "fixtures" / "placed" / "moe_mega_kernel.py"
    done = tf(
        "analyze",
        f"{source}:MoEMegaKernel",
        "--compute-cost",
        "--memory",
        "--roofline",
        "--performance",
        "--json",
    )
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)

    as_text = tf(
        "analyze",
        f"{source}:MoEMegaKernel",
        "--compute-cost",
        "--memory",
        "--roofline",
        "--performance",
    )
    assert as_text.returncode == 0, as_text.stderr
    text = as_text.stdout
    assert as_text.stderr == ""

    families = ["compute-cost", "memory", "roofline", "performance"]
    assert payload["requested"] == payload["executed"] == families
    assert set(payload["function_records"]) == {*families, "traffic"}
    assert len(payload["calls"]) == 7
    assert all(
        set(row) - {"performance"} == {"value", "compute-cost", "roofline", "traffic"}
        for row in payload["calls"]
    )
    assert [
        index for index, row in enumerate(payload["calls"]) if "performance" in row
    ] == [1, 4, 6]
    assert text.startswith(
        "# analysis target=nvidia.h200_sxm module=MoEMegaKernel function=experts"
    )
    for conclusion in (
        "# selection requested=compute-cost,memory,roofline,performance",
        "# compute-cost flops=f32:",
        "# peak-footprint=",
        "# roofline ideal-ns=",
        "# performance root=MoEMegaKernel::experts predicted-ns=",
    ):
        assert conclusion in text


def test_usage_errors_include_the_command_help(tf) -> None:
    done = tf("analyze")
    assert done.returncode == 2

    assert done.stdout == ""
    assert done.stderr.startswith(
        "tilefoundry analyze: error: the following arguments are required: SOURCE\n\n"
    )
    assert "usage: tilefoundry analyze" in done.stderr
    assert "SOURCE" in done.stderr
    assert "model.py[:Module[.child_module...][.function]]" in done.stderr


def test_a_bare_analyze_typechecks_and_prints_only_typed_hir(tf, cmine) -> None:
    done = tf("analyze", f"{cmine}:CMine.root", "--topology", "not-a-level")
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    assert "# Tensor[" in done.stdout
    assert "# analysis " not in done.stdout
    assert "compute-cost" not in done.stdout
    assert "memory peak=" not in done.stdout
    assert "roofline" not in done.stdout
    assert "performance=" not in done.stdout


def test_analyze_json_needs_an_explicit_analysis(tf, cmine) -> None:
    done = tf("analyze", f"{cmine}:CMine.root", "--json")
    assert done.returncode == 2
    assert done.stdout == ""
    assert done.stderr.startswith(
        "tilefoundry analyze: error: --json requires at least one analysis flag:"
    )
    assert "usage: tilefoundry analyze" in done.stderr


def test_a_bare_analyze_binds_every_open_dimension(tf, tmp_path) -> None:
    source = tmp_path / "open.py"
    source.write_text(_OPEN_MODULE, encoding="utf-8")

    unbound = tf("analyze", f"{source}:Open")
    assert unbound.returncode == 1
    assert unbound.stdout == ""
    assert "N is declared as [1, 9)" in unbound.stderr

    bound = tf("analyze", f"{source}:Open", "--dim", "N=4")
    assert bound.returncode == 0, bound.stderr
    assert "# analysis " not in bound.stdout


def test_performance_resolves_derived_execution_geometry(tf, derived_prefill) -> None:
    source = f"{derived_prefill}:DerivedPrefill.prefill"
    unbound = tf("analyze", source, "--performance", "--json")
    assert unbound.returncode == 1
    assert unbound.stdout == ""
    assert "prefill_n is declared as [1, 65)" in unbound.stderr
    assert "topology_only is declared as [1, 1025)" in unbound.stderr

    bound = tf(
        "analyze",
        source,
        "--performance",
        "--dim",
        "prefill_n=17",
        "--dim",
        "topology_only=32",
        "--json",
    )
    assert bound.returncode == 0, bound.stderr
    payload = json.loads(bound.stdout)
    record = payload["function_records"]["performance"]
    assert record == {
        "timeline": {
            "start_ns": 0,
            "end_ns": record["timeline"]["end_ns"],
            "trips": 1,
            "stride_ns": 0,
        },
        "waves": 1,
    }


def test_analyze_reports_only_the_analyses_that_were_requested(tf, cwide) -> None:
    done = tf("analyze", f"{cwide}:Model", "--roofline")
    assert done.returncode == 0, done.stderr

    assert (
        "# selection requested=roofline executed=compute-cost,memory,roofline"
        in done.stdout
    )
    assert "# compute-cost flops=" in done.stdout
    assert "# roofline ideal-ns=" in done.stdout
    assert "# peak-footprint=" not in done.stdout
    assert "# performance " not in done.stdout
    assert "; roofline ideal-ns=" in done.stdout
    assert "; memory peak=" not in done.stdout
    assert "; compute-cost" not in done.stdout
    assert "; performance=" not in done.stdout


def test_analyze_failure_reports_line_variable_and_reason(tf, tmp_path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(_BAD_MODULE, encoding="utf-8")

    done = tf("analyze", f"{bad}:Bad")
    assert done.returncode == 1

    assert done.stdout == ""
    assert f"{bad}:9:" in done.stderr
    assert "variable 'wrong'" in done.stderr
    assert "dtype mismatch" in done.stderr


def test_analyze_loads_sibling_modules_without_leaking_paths_or_output(tf, siblings) -> None:
    first = siblings("first", "First")
    second = siblings("second", "Second")

    for source, name in ((first, "First"), (second, "Second")):
        done = tf("analyze", f"{source}:{name}", "--memory")
        assert done.returncode == 0, done.stderr

        assert "source output" not in done.stdout


def test_analyze_names_the_source_directory_when_a_sibling_is_missing(tf, tmp_path) -> None:
    source = tmp_path / "runtime_model.py"
    source.write_text("from model import Model\n", encoding="utf-8")

    done = tf("analyze", str(source), "--memory")
    assert done.returncode == 1

    assert str(tmp_path) in done.stderr
    assert "a sibling module must sit in that directory" in done.stderr


def test_the_cli_reports_a_duplicate_dimension_and_analyses_nothing(tf, tmp_path) -> None:
    source = tmp_path / "empty.py"
    source.write_text("", encoding="utf-8")

    done = tf("analyze", str(source), "--dim=ctx_len=8", "--dim=ctx_len=512")
    assert done.returncode == 1

    assert done.stdout == ""
    assert "ctx_len was given twice" in done.stderr


def test_a_dim_that_is_not_understood_is_refused(tf, cmine) -> None:
    for bad in ("ctx_len", "ctx_len=", "ctx_len=wide"):
        done = tf("analyze", f"{cmine}:CMine.root", "--compute-cost", "--dim", bad)
        assert done.returncode != 0, bad
        assert done.stderr.strip(), bad


def test_analyze_names_an_open_dimension_and_suggests_extents(tf, shipped) -> None:
    model = f"{Path(shipped['models']) / 'qwen3_5_35b_a3b' / 'model.py'}:Qwen3_5_35B_A3B.layer3.mixer.full_attention"
    done = tf("analyze", model, "--memory")
    assert done.returncode == 1
    assert done.stdout == ""
    assert "ctx_len" in done.stderr
    assert "[0, 262144)" in done.stderr
    assert "0, 1, 131072, 262143" in done.stderr


def test_analyze_rejects_several_extents_for_one_dimension(tf, shipped) -> None:
    model = f"{Path(shipped['models']) / 'qwen3_5_35b_a3b' / 'model.py'}:Qwen3_5_35B_A3B.layer3.mixer.full_attention"
    done = tf("analyze", model, "--dim", "ctx_len=0,1")
    assert done.returncode == 1
    assert done.stdout == ""
    assert "ctx_len takes one EXTENT at a time" in done.stderr


def test_a_selector_that_names_nothing_says_so(tf, cmine) -> None:
    done = tf("analyze", f"{cmine}:CMine.nope", "--compute-cost")
    assert done.returncode == 1
    assert "nope" in done.stderr
