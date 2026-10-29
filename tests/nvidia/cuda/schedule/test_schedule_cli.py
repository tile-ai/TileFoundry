"""Scheduling CLI cases with fixture-owned behavioral guards."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pytest
import torch

from tests.core._corpus import Case, Problems, case, reload
from tests.nvidia.cuda.schedule._plain import SiteCounts, check_plain_report
from tilefoundry.cli import main as cli_main
from tilefoundry.ir.tir import PrimFunction


@dataclass(frozen=True)
class ScheduleCliCase(Case):
    path: Path = Path(__file__)
    argv: tuple[str, ...] = ()
    code: int = 0
    files: Mapping[str, Callable[[], torch.Tensor]] = field(default_factory=dict)


def _quiet_json(out: Path, captured: pytest.CaptureResult[str]) -> dict:
    assert captured == ("", "")
    return json.loads(out.read_text())


FACTS_GOLDEN = Path(__file__).parents[1] / "fixtures" / "schedule" / "h200.facts.txt"


CANDIDATE_GOLDEN = (
    Path(__file__).parents[1]
    / "fixtures"
    / "schedule"
    / "plain"
    / "gemm_8192x17408x5120_cta_grid.candidates.txt"
)


ANALYZED_GOLDEN = (
    Path(__file__).parents[1]
    / "fixtures"
    / "schedule"
    / "hir"
    / "gemm_8192x17408x5120_tma_store.analyzed.txt"
)


@case
class SelectedRows(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        records = _quiet_json(out, captured)
        gather = next(
            row["memory"]
            for row in [row for function in records["functions"] for row in function["calls"]]
            if any(
                operand["arg"] == 1 and operand["name"] == "idx"
                for operand in row["memory"]["operands"]
            )
        )
        assert gather["operands"][0]["read"] == 8192, "gather reads 64 selected bf16 rows"
        footprint = records["functions"][0]["function_records"]["memory"]["footprint"]
        assert footprint["buffers"]["kv"]["gmem"]["total"] == 8192
        assert footprint["buffers"]["kv"]["gmem"]["total"] <= gather["operands"][0]["read"]


@case
class ServiceWork(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        records = _quiet_json(out, captured)
        cost = records["functions"][0]["function_records"]["compute-cost"]
        assert not {"i64", "bool"}.intersection(cost["flops"]), "services are not flops"
        assert cost["other_ops"]["integer"]["total"] > 0, "i64 arithmetic counts as integer work"
        assert cost["other_ops"]["predicate"]["total"] > 0, "predicates count separately"


@case
class GmemGatherRefusal(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        report = _quiet_json(out, captured)
        indexed = next(row for row in report["lines"] if row["op"] == "tf.index_select")
        assert not indexed["candidates"] and [item["id"] for item in indexed["refused"]] == [
            "T.copy_async"
        ], "gather keeps its gmem result"
        assert indexed["refused"][0]["reason"].startswith("mesh:")
        assert 'Topology("cta"' in indexed["refused"][0]["reason"]


@case
class SmemPlacement(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        """The second 64x64 bf16 tile ends at 8192 + 8192, the analyzed peak."""
        assert captured == ("", "")
        lines = out.read_text().splitlines()
        key_view = next(i for i, line in enumerate(lines) if line.strip() == "ks = T.tensor_view(")
        assert lines[key_view + 1].strip().split(",", 1)[0] == "0", (
            "the lowered key tile starts at 0 bytes"
        )


def _attention_tensor(name: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(9)
    q = torch.randn(1, 2, 128, 64, generator=generator).to(torch.bfloat16)
    kv = torch.randn(1, 256, 1, 64, generator=generator).to(torch.bfloat16)
    idx = torch.arange(64, dtype=torch.int64) * 3
    idx[1], idx[17] = -1, 256
    live = (idx >= 0) & (idx < 256)
    rows = torch.zeros(64, 64, dtype=torch.bfloat16)
    rows[live] = kv.reshape(256, 64)[idx[live]]
    scores = q[0, 1, 32:96].float() @ rows.float().t()
    scores[:, ~live] = -torch.inf
    probabilities = torch.softmax(scores, dim=1).to(torch.bfloat16)
    reference = (probabilities.float() @ rows.float()).to(torch.bfloat16)
    return {"q": q, "kv": kv, "idx": idx, "expected": reference}[name]


@case
class Attention(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        report = _quiet_json(out, captured)
        comparison = report["runs"][0]["outputs"][0]["fns"][0]
        assert comparison["fn"] == "allclose"
        assert comparison["rtol"] == 2**-7 and comparison["atol"] == 0
        assert report["passed"] and comparison["passed"], (
            "gather fill, mask/key alignment and natural output heads match torch"
        )


def _e4m3_gemm_tensor(name: str) -> torch.Tensor:
    """Integer inputs in [-4, 4] keep every partial sum of the product exact."""
    generator = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (16, 128), generator=generator).float().to(torch.float8_e4m3fn)
    weight = torch.randint(-4, 5, (8, 128), generator=generator).float().to(torch.float8_e4m3fn)
    return {"a": a, "b": weight.t(), "expected": a.float() @ weight.float().t()}[name]


@case
class E4M3Gemm(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        report = _quiet_json(out, captured)
        comparison = report["runs"][0]["outputs"][0]["fns"][0]
        assert comparison["fn"] == "allclose"
        assert comparison["rtol"] == 0 and comparison["atol"] == 0
        assert report["passed"] and comparison["passed"], (
            "K slicing, the (K, N) view of B and the accumulation across slices match torch"
        )


@case
class Golden(ScheduleCliCase):
    golden: Path = Path()

    def guard(self, state, problems):
        out, captured = state
        golden = self.golden
        assert captured == ("", "")
        assert out.read_bytes() == golden.read_bytes()


@case
class FactsRefusal(ScheduleCliCase):
    message: str = ""

    def guard(self, state, problems):
        out, captured = state
        message = self.message
        assert captured.out == ""
        assert captured.err.startswith("tilefoundry: error: ")
        assert message in captured.err
        assert not out.exists()


@case
class FinalizedRows(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        assert captured == ("", "")
        lowered = reload(out.read_text(), "RowCopy").entry_function()
        assert isinstance(lowered, PrimFunction)
        assert lowered.params[-1].type.shape == (128,)


@case
class FinalizeUnboundDim(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        assert captured.out == ""
        assert "schedule finalize" in captured.err
        assert "bind it with --dim" in captured.err
        assert not out.exists()


@case
class ReshardLocations(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        rows = _quiet_json(out, captured)["functions"][0]["calls"]
        rows = {row["value"].rsplit(":", 1)[1]: row["memory"] for row in rows}
        for label, byte_count in (("12", 256), ("20", 224)):
            memory = rows[label]
            assert memory["operands"][0]["read"] == byte_count
            assert memory["operands"][1]["write"] == byte_count
        assert rows["20"]["buffer_bytes"] == 112
        assert rows["20"]["offsets"]
        for label in ("14", "15", "16", "17", "18"):
            memory = rows[label]
            assert memory["buffer_bytes"] is None
            assert memory["offsets"] == []
            assert memory["traffic"] == {"communication": {}, "storage": {}}


@case
class ReshapeMatchesTorch(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        report = _quiet_json(out, captured)
        assert report["passed"]
        comparison = report["runs"][0]["outputs"][0]["fns"][0]
        assert comparison["fn"] == "allclose" and comparison["passed"]


@case
class CrossingAxesRefusal(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        assert captured.out == ""
        assert "layout positions (2, 8, 2, 4, 2, 4, 8) cross logical axes of (64, 128)" in captured.err
        assert "layout_crossing_axes.py:13:" in captured.err
        assert not out.exists()


@case
class SelectorHelp(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        assert captured.err == ""
        assert "model.py[:Module[.function]]" in captured.out
        assert "child_module" not in captured.out


def _reshape_tensor(expected: bool = False) -> torch.Tensor:
    x = torch.arange(128, dtype=torch.float32).reshape(1, 8, 1, 16)
    return x.reshape(8, 16) if expected else x


@case
class PlainCandidates(ScheduleCliCase):
    source: str = ""
    sites: SiteCounts | None = None

    def guard(self, state, problems):
        out, captured = state
        source = self.source
        sites = self.sites
        assert captured == ("", "")
        check_plain_report(json.loads(out.read_text()), source, sites)


@case
class InvalidChunks(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        assert "[1, 32]" in captured.err
        assert not out.exists()


@case
class UnboundChunks(ScheduleCliCase):

    def guard(self, state, problems):
        out, captured = state
        assert "chunks is declared as [1, 32]" in captured.err
        assert "bind it with --dim" in captured.err
        assert not out.exists()


FIXTURES = (
    ReshardLocations(
        name='reshard is charged only when an element moves',
        argv=(
            'analyze',
            'tests/nvidia/cuda/fixtures/schedule/plain/reshard_locations.py',
            '{out}',
            '--memory',
            '--json',
        ),
    ),
    ReshapeMatchesTorch(
        name='check binds two DimVars in a reshape',
        argv=(
            'check',
            'tests/nvidia/cuda/fixtures/placed/two_dimvar_reshape.py:TwoDimVarReshape',
            '--inputs',
            'files:{tmp}/x.pt',
            '--expected',
            '{tmp}/expected.pt',
            '--dim',
            'T=8',
            '--dim',
            'C=16',
            '--device',
            'cpu',
            '--out',
            'output',
            '--fn',
            'allclose',
            '--atol',
            '0',
            '--rtol',
            '0',
            '--json',
            '{out}',
        ),
        files={'x.pt': _reshape_tensor, 'expected.pt': partial(_reshape_tensor, True)},
    ),
    CrossingAxesRefusal(
        name='finalize refuses a layout crossing logical axes where it is declared',
        argv=(
            'schedule',
            'finalize',
            'tests/nvidia/cuda/fixtures/diagnostics/layout_crossing_axes.py',
            '{out}',
        ),
        code=1,
    ),
    SelectorHelp(
        name='analyze help names only Module.function',
        argv=('analyze', '--help'),
    ),
    FinalizedRows(
        name='finalize binds an open row dimension',
        argv=(
            'schedule',
            'finalize',
            'tests/nvidia/cuda/fixtures/schedule/hir/dim/row_copy.py',
            '{out}',
            '--dim=rows=128',
        ),
    ),
    FinalizeUnboundDim(
        name='finalize refuses an unbound row dimension without output',
        argv=('schedule', 'finalize', 'tests/nvidia/cuda/fixtures/schedule/hir/dim/row_copy.py', '{out}'),
        code=1,
    ),
    SelectedRows(
        name='gather reads selected rows and caps the footprint',
        argv=(
            'analyze',
            'tests/nvidia/cuda/fixtures/schedule/plain/sparse_decode.py',
            '{out}',
            '--memory',
            '--json',
        ),
    ),
    ServiceWork(
        name='integer and predicate work uses services',
        argv=(
            'analyze',
            'tests/nvidia/cuda/fixtures/schedule/plain/sparse_decode.py',
            '{out}',
            '--compute-cost',
            '--json',
        ),
    ),
    GmemGatherRefusal(
        name='gather into smem is refused for a gmem result',
        argv=(
            'schedule',
            'candidates',
            'tests/nvidia/cuda/fixtures/schedule/plain/sparse_decode.py',
            '{out}',
            '--json',
        ),
    ),
    SmemPlacement(
        name='analyzed smem peak matches the lowered placement',
        argv=(
            'schedule',
            'finalize',
            'tests/nvidia/cuda/fixtures/schedule/hir/sparse_decode.py',
            '{out}',
        ),
    ),
    *(
        Attention(
            name=name,
            argv=(
                'check',
                f'tests/nvidia/cuda/fixtures/schedule/{stage}/sparse_decode.py:SparseDecode',
                '--inputs',
                'files:{tmp}/q.pt,{tmp}/kv.pt,{tmp}/idx.pt',
                '--expected',
                '{tmp}/expected.pt',
                '--device',
                'cpu',
                '--out',
                'output',
                '--fn',
                'allclose',
                '--rtol',
                str(2 ** (-7)),
                '--atol',
                '0',
                '--json',
                '{out}',
            ),
            files={f'{key}.pt': partial(_attention_tensor, key) for key in ('q', 'kv', 'idx', 'expected')},
        )
        for (name, stage) in (
            ('plain sparse attention matches torch', 'plain'),
            ('scheduled sparse attention matches torch', 'hir'),
        )
    ),
    *(
        E4M3Gemm(
            name=name,
            argv=(
                'check',
                f'tests/nvidia/cuda/fixtures/schedule/{stage}/sm89_e4m3_gemm.py:SM89_E4M3_GEMM',
                '--inputs',
                'files:{tmp}/a.pt,{tmp}/b.pt',
                '--expected',
                '{tmp}/expected.pt',
                '--device',
                'cpu',
                '--out',
                'output',
                '--fn',
                'allclose',
                '--atol',
                '0',
                '--rtol',
                '0',
                '--json',
                '{out}',
            ),
            files={f'{key}.pt': partial(_e4m3_gemm_tensor, key) for key in ('a', 'b', 'expected')},
        )
        for (name, stage) in (('plain e4m3 gemm matches torch', 'plain'), ('scheduled e4m3 gemm matches torch', 'hir'))
    ),
    Golden(
        name='facts describe all H200 instructions',
        argv=('schedule', 'facts', '--target', 'nvidia.h200_sxm', '{out}'),
        golden=FACTS_GOLDEN,
    ),
    *(
        FactsRefusal(
            name=name,
            argv=('schedule', 'facts', instruction, '--target', target, '{out}'),
            code=1,
            message=message,
        )
        for (name, instruction, target, message) in (
            (
                'facts refuse an unknown instruction without output',
                'T.cuda.sm90.Unknown',
                'nvidia.h200_sxm',
                "unknown instruction 'T.cuda.sm90.Unknown'",
            ),
            (
                'facts refuse an unknown target without output',
                'T.cuda.sm90.Wgmma',
                'nvidia.unknown',
                "unknown target identity 'nvidia.unknown'",
            ),
        )
    ),
    Golden(
        name='candidates write the canonical report',
        argv=(
            'schedule',
            'candidates',
            f"tests/nvidia/cuda/fixtures/schedule/plain/{CANDIDATE_GOLDEN.name.removesuffix('.candidates.txt')}.py",
            '{out}',
        ),
        golden=CANDIDATE_GOLDEN,
    ),
    Golden(
        name='analyze writes the memory report',
        argv=(
            'analyze',
            'tests/nvidia/cuda/fixtures/schedule/hir/gemm_8192x17408x5120_tma_store.py',
            '{out}',
            '--memory',
        ),
        golden=ANALYZED_GOLDEN,
    ),
    PlainCandidates(
        name='candidate sites at chunks=32',
        argv=(
            'schedule',
            'candidates',
            'tests/nvidia/cuda/fixtures/schedule/plain/chunk_rmsnorm.py',
            '{out}',
            '--json',
            '--dim=chunks=32',
        ),
        source='tests/nvidia/cuda/fixtures/schedule/plain/chunk_rmsnorm.py',
        sites=SiteCounts(0, 7, detailed=True, binaries=5),
    ),
    InvalidChunks(
        name='candidates refuses chunks outside declared range',
        argv=(
            'schedule',
            'candidates',
            'tests/nvidia/cuda/fixtures/schedule/plain/chunk_rmsnorm.py',
            '{out}',
            '--dim=chunks=33',
        ),
        code=1,
    ),
    UnboundChunks(
        name='candidates requires chunks binding',
        argv=(
            'schedule',
            'candidates',
            'tests/nvidia/cuda/fixtures/schedule/plain/chunk_rmsnorm.py',
            '{out}',
        ),
        code=1,
    ),
)


def test_candidates_nk_layout(tmp_path, capsys):
    variant = Path(__file__).parents[1] / "fixtures/schedule/variants/gemm_relu_gemm_untiled_nk.py"
    out = tmp_path / "gemm_relu_gemm_untiled.json"
    assert cli_main(["schedule", "candidates", str(variant), str(out), "--json"]) == 0
    case = PlainCandidates(source=str(variant), sites=SiteCounts(2, 0, detailed=True))
    problems = Problems(case)
    problems.run("guard", case.guard, (out, capsys.readouterr()), problems)
    problems.dump()


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case.name)
def test_cli_case(case: ScheduleCliCase, tmp_path, capsys):
    problems = Problems(case)

    def invoke():
        for name, make in case.files.items():
            torch.save(make(), tmp_path / name)
        out = tmp_path / "out"
        argv = [arg.format(out=out, tmp=tmp_path) for arg in case.argv]
        try:
            code = cli_main(argv)
        except SystemExit as error:
            code = error.code
        captured = capsys.readouterr()
        assert code == case.code, captured
        return out, captured

    state = problems.run("cli", invoke)
    problems.run("guard", case.guard, state, problems)
    problems.dump()
