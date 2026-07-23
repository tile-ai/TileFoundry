from __future__ import annotations

import textwrap

from tilefoundry import cli

_VALID_MODULE = """
from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.target import CudaTarget

@module(entry="main")
class Model:
    @func(target=CudaTarget(), topologies=(Topology("cta", 168),))
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
    assert "target: h200_sxm_sm90" in output
    assert "grid_cta_count: 168" in output
    assert "hbm_bandwidth: 4.8 TB/s [direct]" in output
    assert "l2_bandwidth: unavailable [unavailable]" in output


def test_inspect_capabilities_rejects_an_uninstalled_cuda_target(tmp_path, capsys) -> None:
    path = _write_module(
        tmp_path,
        _VALID_MODULE.replace(
            "from tilefoundry.target import CudaTarget",
            "from tilefoundry.target import CudaTarget, SM90",
        ).replace(
            "target=CudaTarget()",
            'target=CudaTarget(architecture=SM90(name="sm_90_custom"))',
        ),
    )

    assert cli.main(["inspect", "capabilities", f"{path}:Model.main"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no installed authored-analysis hardware spec" in captured.err
    assert "sm_90_custom" in captured.err


def test_analyze_selects_default_or_requested_analyses(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(cli, "run_authored_analysis", lambda source, analyses: calls.append((source, analyses)))

    assert cli.main(["analyze", "model.py"]) is None
    assert cli.main(["analyze", "model.py", "--timeline"]) is None
    assert calls == [
        ("model.py", ("roofline", "footprint", "timeline")),
        ("model.py", ("timeline",)),
    ]


def test_analyze_prints_summary_types_and_selected_metadata(tmp_path, capsys) -> None:
    path = _write_module(tmp_path)

    assert cli.main(["analyze", f"{path}:Model"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("# analysis target=cuda analyses=roofline,footprint,timeline")
    assert "type=Tensor[" in captured.out
    assert "roofline flops=" in captured.out
    assert "footprint live=" in captured.out
    assert "timeline ctas=168 waves=2" in captured.out


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
