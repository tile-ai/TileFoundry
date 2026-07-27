from __future__ import annotations

import json
import textwrap

from tilefoundry import cli

_VALID_MODULE = """
from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.target import CudaTarget

@module(entry="main", target=CudaTarget())
class Model:
    topologies = (Topology("cta", 168),)

    @func
    def main(x: Tensor[(168,), "f32"]):
        with Mesh(Topology("cta", 168), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            squared = tf.square(x_local)
            return tf.reshard(squared, (168 @ cta.block,), "gmem")
"""


def _write_module(tmp_path, source: str = _VALID_MODULE):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_help_dsl_is_the_hir_spec(capsys) -> None:
    assert cli.main(["help", "dsl"]) == 0
    assert cli.dsl_spec_path() == cli.spec_path("hir")
    assert capsys.readouterr().out == cli.spec_path("hir").read_text(encoding="utf-8")


def test_help_cli_is_the_source_spec(capsys) -> None:
    assert cli.main(["help", "cli"]) == 0
    assert capsys.readouterr().out == cli.spec_path("cli").read_text(encoding="utf-8")


def test_inspect_capabilities_is_compact(tmp_path, capsys) -> None:
    path = _write_module(tmp_path)
    assert cli.main(["inspect", "capabilities", f"{path}:Model.main"]) == 0
    output = capsys.readouterr().out
    assert "architecture: nvidia.sm90" in output
    assert "device: nvidia.h200_sxm" in output
    assert "grid_cta_count: 168" in output
    assert "memory.hbm.bandwidth: 4800000000000 byte/s [vendor]" in output
    assert "memory.l2.bandwidth: unavailable" in output


def test_inspect_capabilities_rejects_an_uninstalled_cuda_target(tmp_path, capsys) -> None:
    path = _write_module(
        tmp_path,
        _VALID_MODULE.replace(
            "from tilefoundry.target import CudaTarget",
            "from dataclasses import replace\n"
            "from tilefoundry.target import CudaTarget\n"
            "from tilefoundry.target.cuda.spec import installed_architecture",
        ).replace(
            "target=CudaTarget()",
            "target=CudaTarget("
            'architecture=replace(installed_architecture(), name="sm_90_custom"))',
        ),
    )

    assert cli.main(["inspect", "capabilities", f"{path}:Model.main"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no installed hardware documents" in captured.err
    assert "sm_90_custom" in captured.err


def test_analyze_selects_default_or_requested_analyses(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...], bool]] = []
    monkeypatch.setattr(
        cli,
        "run_authored_analysis",
        lambda source, analyses, as_json=False: calls.append(
            (source, analyses, as_json)
        ),
    )

    assert cli.main(["analyze", "model.py"]) is None
    assert cli.main(["analyze", "model.py", "--timeline"]) is None
    assert cli.main(["analyze", "model.py", "--memory", "--json"]) is None
    assert calls == [
        ("model.py", ("compute-cost", "memory", "roofline", "timeline"), False),
        ("model.py", ("timeline",), False),
        ("model.py", ("memory",), True),
    ]


def test_analyze_prints_summary_types_and_selected_metadata(tmp_path, capsys) -> None:
    path = _write_module(tmp_path)

    assert cli.main(["analyze", f"{path}:Model"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith(
        "# analysis target=cuda module=Model function=main"
    )
    assert "type=Tensor[" in captured.out
    # Every reported line comes off a record; the annotated body carries the
    # per-Call ones as comments.
    assert "# peak-footprint gmem=" in captured.out
    assert "# theoretical-bound=" in captured.out
    assert "# theoretical-makespan=" in captured.out
    assert "compute-cost flops=f32:" in captured.out
    assert "roofline bound=" in captured.out
    assert "timeline units=168 waves=2" in captured.out


def test_analyze_reports_only_the_analyses_that_were_requested(tmp_path, capsys) -> None:
    """A requested root pulls its dependencies in, so their records reach the IR
    without having been asked for. Every view of the run shows what was
    requested -- the report and the annotated source alike; the executed line
    still names the whole closure that ran."""
    path = _write_module(tmp_path)

    assert cli.main(["analyze", f"{path}:Model", "--roofline"]) == 0

    captured = capsys.readouterr()
    assert "# analyses=roofline executed=compute-cost,memory,roofline" in captured.out
    assert "# theoretical-bound=" in captured.out
    assert "# peak-footprint" not in captured.out
    assert "# theoretical-makespan" not in captured.out
    # The annotated source is the other view, and it withholds the same records.
    assert "roofline bound=" in captured.out
    assert "memory peak=" not in captured.out
    assert "compute-cost flops=" not in captured.out
    assert "timeline units=" not in captured.out


def test_analyze_json_and_text_report_the_same_conclusions(tmp_path, capsys) -> None:
    """Both formats render one report, so neither can state something the other
    does not."""
    path = _write_module(tmp_path)

    assert cli.main(["analyze", f"{path}:Model", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert cli.main(["analyze", f"{path}:Model"]) == 0
    text = capsys.readouterr().out

    assert payload["target"] == "cuda"
    assert payload["function"] == "main"
    assert payload["executed"] == ["compute-cost", "memory", "roofline", "timeline"]
    for level, value in payload["totals"]["traffic"].items():
        assert f"{level}=r{value['read_bytes']}/w{value['write_bytes']}" in text
    for item in payload["function_records"]["memory"]["footprint"]:
        assert f"{item['level']}={item['peak_bytes']}" in text
    assert (
        f"by={payload['function_records']['roofline']['bound_by']}" in text
    )


def test_analyze_failure_reports_line_variable_and_reason(tmp_path, capsys) -> None:
    path = _write_module(
        tmp_path,
        """
        from tilefoundry import module
        from tilefoundry.dsl import Tensor, func, tf

        @module(entry="main")
        class Bad:
            @func
            def main(x: Tensor[(8,), "f32"]):
                wrong = tf.add(x, tf.cast(x, "i32"))
                return wrong
        """,
    )

    assert cli.main(["analyze", f"{path}:Bad"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"{path}:9:" in captured.err
    assert "variable 'wrong'" in captured.err
    assert "dtype mismatch" in captured.err
