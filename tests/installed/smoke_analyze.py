"""`analyze` against a file of one's own."""

from __future__ import annotations

import json
from pathlib import Path

_BAD_MODULE = '''
from tilefoundry import module
from tilefoundry.dsl import Tensor, func, tf

@module(entry="main")
class Bad:
    @func
    def main(x: Tensor[(8,), "f32"]):
        wrong = tf.add(x, tf.cast(x, "i32"))
        return wrong
'''


def test_every_analysis_the_command_offers_runs(tf, cmine) -> None:
    done = tf(
        "analyze", f"{cmine}:CMine.root",
        "--compute-cost", "--memory", "--roofline", "--timeline",
    )
    assert done.returncode == 0, done.stderr
    for conclusion in ("flops ", "traffic ", "peak-footprint ", "ideal-bound="):
        assert conclusion in done.stdout, conclusion


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


def test_analyze_reports_only_the_analyses_that_were_requested(tf, cwide) -> None:
    done = tf("analyze", f"{cwide}:Model", "--roofline")
    assert done.returncode == 0, done.stderr

    assert "# analyses=roofline executed=compute-cost,memory,roofline" in done.stdout
    assert "# ideal-bound=" in done.stdout
    assert "# peak-footprint" not in done.stdout
    assert "# theoretical-makespan" not in done.stdout
    assert "roofline bound=" in done.stdout
    assert "memory peak=" not in done.stdout
    assert "compute-cost flops=" not in done.stdout
    assert "timeline units=" not in done.stdout


def test_analyze_json_and_text_report_the_same_conclusions(tf, cwide) -> None:
    as_json = tf("analyze", f"{cwide}:Model", "--json")
    assert as_json.returncode == 0, as_json.stderr
    payload = json.loads(as_json.stdout)

    as_text = tf("analyze", f"{cwide}:Model")
    assert as_text.returncode == 0, as_text.stderr
    text = as_text.stdout
    assert as_text.stderr == ""

    assert payload["target"] == "cuda"
    assert payload["function"] == "main"
    assert payload["executed"] == ["compute-cost", "memory", "roofline", "timeline"]
    for level, value in payload["totals"]["traffic"].items():
        assert f"{level}=r{value['read']}/w{value['write']}" in text
    for item in payload["function_records"]["memory"]["footprint"]:
        assert f"{item['level']}={item['peak_bytes']}" in text
    assert f"by={payload['function_records']['roofline']['bound_by']}" in text

    assert text.startswith("# analysis target=cuda module=Model function=main")
    assert "# Tensor[" in text
    assert "compute-cost flops=f32:" in text
    assert "roofline bound=" in text
    assert "timeline units=168 waves=2" in text


def test_analyze_failure_reports_line_variable_and_reason(tf, tmp_path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(_BAD_MODULE, encoding="utf-8")

    done = tf("analyze", f"{bad}:Bad")
    assert done.returncode == 1

    assert done.stdout == ""
    assert f"{bad}:9:" in done.stderr
    assert "variable 'wrong'" in done.stderr
    assert "dtype mismatch" in done.stderr


def test_analyze_loads_sibling_modules_without_leaking_paths_or_output(
    tf, siblings
) -> None:
    first = siblings("first", "First")
    second = siblings("second", "Second")

    for source, name in ((first, "First"), (second, "Second")):
        done = tf("analyze", f"{source}:{name}", "--memory")
        assert done.returncode == 0, done.stderr
        # The sibling is imported, so its print runs -- but not onto this output.
        assert "source output" not in done.stdout


def test_analyze_names_the_source_directory_when_a_sibling_is_missing(
    tf, tmp_path
) -> None:
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
