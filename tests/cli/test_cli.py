from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

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


def test_models_separates_oracles_from_everything_else(capsys) -> None:
    """The list has to say which models can be a reference and which cannot.

    A model hidden for being below the bar gets rebuilt by whoever needed it, so
    they stay listed and are marked instead.
    """
    assert cli.main(["models"]) == 0
    listed = capsys.readouterr().out

    assert "usable as an oracle" in listed and "not usable as an oracle" in listed
    for name in ("qwen3_1_7b", "kimi_linear_48b_a3b"):
        assert name in listed
    # Every level the entries use is explained in the same output.
    assert "L1" in listed and "L2" in listed and "L3" in listed


def test_models_renders_the_whole_forest_with_leaf_modules_marked(capsys) -> None:
    """Every top-level Module, `*` on the ones with no children, counts beside them.

    `*` marks a Module and not a function because a twin is written per Module and
    has to cover all of that Module's functions at once.
    """
    assert cli.main(["models", "qwen3_1_7b"]) == 0
    forest = capsys.readouterr().out

    assert "29 leaf modules, 148 functions" in forest
    # Both roots, and a layer marked as a leaf while the stack that owns it is not.
    assert "* Qwen3_1_7B\n" in forest
    assert "  Qwen3_1_7B_Decoder\n" in forest
    # The 28 layers are one entry naming the range and how many, written once.
    assert "*   layer0..layer27  (28 identical, each as shown)" in forest
    assert "layer1\n" not in forest
    assert "input_rms_norm(hidden: Tensor[(1, 1, 2048), \"f32\"]" in forest


def test_models_source_is_the_authored_file_byte_for_byte(capsys) -> None:
    """`--source` is the reference an agent copies, so it is the file, not a render."""
    assert cli.main(["models", "qwen3_1_7b", "--source"]) == 0

    printed = capsys.readouterr().out
    authored = Path("tests/models/qwen3_1_7b/model.py").read_text(encoding="utf-8")
    assert printed == authored


def test_models_rejects_a_name_the_catalog_does_not_have(capsys) -> None:
    """The refusal lists the models there are."""
    assert cli.main(["models", "nope"]) == 1

    error = capsys.readouterr().err
    assert "no model named 'nope'" in error
    assert "qwen3_1_7b" in error


def test_spec_lists_the_documents_there_are(capsys) -> None:
    """With no topic, the answer is what can be asked for -- including the alias."""
    assert cli.main(["spec"]) == 0
    listed = capsys.readouterr().out

    assert "hir" in listed and "dsl" in listed
    assert "runtime" in listed


def test_spec_outlines_a_document_rather_than_printing_it(capsys) -> None:
    """An outline is the disclosure; the whole document is the thing being avoided.

    Asserted as "shorter, and without the prose" rather than by matching wording:
    a dump would satisfy any test that only looked for section titles.
    """
    assert cli.main(["spec", "dsl"]) == 0
    outline = capsys.readouterr().out
    whole = cli.spec_path("hir").read_text(encoding="utf-8")

    assert "Silu" in outline and "silu" in outline
    assert len(outline) < len(whole) / 4
    assert "class Silu(Op):" not in outline


def test_spec_prints_one_section_and_the_keys_beside_it(capsys) -> None:
    """One section, its own body, and where to go next without the outline."""
    assert cli.main(["spec", "dsl", "silu"]) == 0
    section = capsys.readouterr().out

    assert "class Silu(Op):" in section
    # The neighbours are named, and the neighbour's own body is not dragged in.
    assert "next:     rmsnorm" in section
    assert "class RMSNorm(Op):" not in section


def test_spec_separates_two_sections_that_would_share_a_key(capsys) -> None:
    """Two sections numbered 3.2 are each reachable, and the bare key is refused."""
    assert cli.main(["spec", "parser", "shared-parsing-machinery/3.2"]) == 0
    resolution = capsys.readouterr().out
    assert "Closure-then-registry callee resolution" in resolution
    assert "classDiagram" not in resolution

    assert cli.main(["spec", "parser", "parser-architecture/3.2"]) == 0
    assert "classDiagram" in capsys.readouterr().out

    assert cli.main(["spec", "parser", "3.2"]) == 1
    assert "no section '3.2'" in capsys.readouterr().err


def test_spec_rejects_a_section_that_does_not_exist(capsys) -> None:
    """The refusal says what the document does have, so the next try can succeed."""
    assert cli.main(["spec", "dsl", "9.9"]) == 1
    error = capsys.readouterr().err

    assert "no section '9.9'" in error
    assert "silu" in error


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
        lambda source, analyses, as_json=False, dims=None: calls.append(
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
    does not -- checked over the default run, which is every analysis, so this is
    also where the text form's own shape is judged: the header a reader looks at
    first, the types, and one line per analysis that ran."""
    path = _write_module(tmp_path)

    assert cli.main(["analyze", f"{path}:Model", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert cli.main(["analyze", f"{path}:Model"]) == 0
    captured = capsys.readouterr()
    text = captured.out
    assert captured.err == ""

    assert payload["target"] == "cuda"
    assert payload["function"] == "main"
    assert payload["executed"] == ["compute-cost", "memory", "roofline", "timeline"]
    for level, value in payload["totals"]["traffic"].items():
        assert f"{level}=r{value['read_bytes']}/w{value['write_bytes']}" in text
    for item in payload["function_records"]["memory"]["footprint"]:
        assert f"{item['level']}={item['peak_bytes']}" in text
    assert f"by={payload['function_records']['roofline']['bound_by']}" in text

    assert text.startswith("# analysis target=cuda module=Model function=main")
    assert "type=Tensor[" in text
    # Every reported line comes off a record; the annotated body carries the
    # per-Call ones as comments.
    assert "compute-cost flops=f32:" in text
    assert "roofline bound=" in text
    assert "timeline units=168 waves=2" in text


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


def test_parse_dims_reads_one_extent_per_dimension() -> None:
    """Nothing stated is not the same as nothing chosen."""
    assert cli.parse_dims(None) is None
    assert cli.parse_dims([]) is None
    assert cli.parse_dims(["ctx_len=1024"]) == {"ctx_len": 1024}
    assert cli.parse_dims(["ctx_len=8", "seq_len=1"]) == {"ctx_len": 8, "seq_len": 1}


@pytest.mark.parametrize(
    "stated",
    [["ctx_len"], ["ctx_len="], ["=8"], ["ctx_len=eight"], ["ctx_len=1.5"]],
)
def test_parse_dims_rejects_an_argument_that_states_no_extent(stated) -> None:
    with pytest.raises(ValueError):
        cli.parse_dims(stated)


def test_parse_dims_rejects_one_dimension_stated_twice() -> None:
    """Repeating the flag states another dimension, not another value for one
    already stated.

    Taking the last would answer an ambiguous request by picking silently, and
    the caller would be told nothing -- which is the failure worth catching,
    because both numbers came from them.
    """
    with pytest.raises(ValueError, match="ctx_len was given twice"):
        cli.parse_dims(["ctx_len=8", "ctx_len=512"])
    # Repeating the same extent is still two statements of one dimension.
    with pytest.raises(ValueError, match="ctx_len was given twice"):
        cli.parse_dims(["ctx_len=8", "ctx_len=8"])


def test_the_cli_reports_a_duplicate_dimension_and_analyses_nothing(
    tmp_path, capsys
) -> None:
    """Through `main`, so the refusal is what a user actually meets."""
    source = tmp_path / "model.py"
    source.write_text("", encoding="utf-8")

    assert cli.main(["analyze", str(source), "--dim=ctx_len=8", "--dim=ctx_len=512"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ctx_len was given twice" in captured.err
