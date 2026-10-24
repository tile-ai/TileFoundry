"""Scheduling schedule cli checks."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest
import torch

from tests.core._source import import_dsl
from tests.core.schedule._fixtures import (
    CliCase,
    _quiet_json,
    check_cli_case,
    check_facts,
)
from tests.nvidia.cuda.schedule._plain import SiteCounts, check_plain_report
from tilefoundry.cli import main as cli_main
from tilefoundry.ir.core import (
    op_identifier,
    supported_op_capabilities,
)
from tilefoundry.ir.tir import PrimFunction
from tilefoundry_nvidia import CudaTarget

FACT_INSTRUCTIONS = sorted(
    {
        op_identifier(declaration)
        for op_type, capability in supported_op_capabilities(CudaTarget("nvidia.h200_sxm"))
        for declaration in (
            (op_type, capability.declaration) if capability.attribute is not None else (op_type,)
        )
    }
)


WGMMA_FACTS = Path(__file__).parents[1] / "fixtures" / "schedule" / "Wgmma.facts.txt"


_H200_FACTS_TEXT = """\
target nvidia.h200_sxm
instructions
  T.arange                     all targets
  T.binary                     all targets
  T.cast                       all targets
  T.clamp                      all targets
  T.unary                      all targets
  T.where                      all targets
  T.copy_async                 cp.async
  T.copy_async_bulk            cp.async.bulk
  T.copy_async_tensor          cp.async.bulk.tensor
  T.copy_reduce_async_tensor   cp.async.bulk.tensor
  T.copy                       all targets
  T.ldmatrix                   ldmatrix
  T.stmatrix                   stmatrix
  T.relu                       all targets
  T.tiled_mma                  wgmma.mma_async, mma.sync
  T.mbarrier_arrive_expect_tx  mbarrier.expect_tx
  T.mbarrier_wait_parity       mbarrier.try_wait
  T.reduce                     all targets
"""


_H200_FACTS_JSON = {
    "target": "nvidia.h200_sxm",
    "instructions": [
        {"id": "T.arange", "capability": None},
        {"id": "T.binary", "capability": None},
        {"id": "T.cast", "capability": None},
        {"id": "T.clamp", "capability": None},
        {"id": "T.unary", "capability": None},
        {"id": "T.where", "capability": None},
        {"id": "T.copy_async", "capability": "cp.async"},
        {"id": "T.copy_async_bulk", "capability": "cp.async.bulk"},
        {"id": "T.copy_async_tensor", "capability": "cp.async.bulk.tensor"},
        {"id": "T.copy_reduce_async_tensor", "capability": "cp.async.bulk.tensor"},
        {"id": "T.copy", "capability": None},
        {"id": "T.ldmatrix", "capability": "ldmatrix"},
        {"id": "T.stmatrix", "capability": "stmatrix"},
        {"id": "T.relu", "capability": None},
        {"id": "T.tiled_mma", "capability": ["wgmma.mma_async", "mma.sync"]},
        {"id": "T.mbarrier_arrive_expect_tx", "capability": "mbarrier.expect_tx"},
        {"id": "T.mbarrier_wait_parity", "capability": "mbarrier.try_wait"},
        {"id": "T.reduce", "capability": None},
    ],
}


_SM89_FACTS_TEXT = """\
target nvidia.rtx_6000_ada
instructions
  T.arange      all targets
  T.binary      all targets
  T.cast        all targets
  T.clamp       all targets
  T.unary       all targets
  T.where       all targets
  T.copy_async  cp.async
  T.copy        all targets
  T.ldmatrix    ldmatrix
  T.relu        all targets
  T.tiled_mma   mma.sync
  T.reduce      all targets
"""


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


def _check_selected_rows(out: Path, captured: pytest.CaptureResult[str]) -> None:
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


def _check_service_work(out: Path, captured: pytest.CaptureResult[str]) -> None:
    records = _quiet_json(out, captured)
    cost = records["functions"][0]["function_records"]["compute-cost"]
    assert not {"i64", "bool"}.intersection(cost["flops"]), "services are not flops"
    assert cost["other_ops"]["integer"]["total"] > 0, "i64 arithmetic counts as integer work"
    assert cost["other_ops"]["predicate"]["total"] > 0, "predicates count separately"


def _check_gmem_gather_refusal(out: Path, captured: pytest.CaptureResult[str]) -> None:
    report = _quiet_json(out, captured)
    indexed = next(row for row in report["lines"] if row["op"] == "tf.index_select")
    assert not indexed["candidates"] and [item["id"] for item in indexed["refused"]] == [
        "T.copy_async"
    ], "gather keeps its gmem result"
    assert indexed["refused"][0]["reason"].startswith("mesh:")
    assert 'Topology("cta"' in indexed["refused"][0]["reason"]


def _check_smem_placement(out: Path, captured: pytest.CaptureResult[str]) -> None:
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


def _check_attention(out: Path, captured: pytest.CaptureResult[str]) -> None:
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


def _check_e4m3_gemm(out: Path, captured: pytest.CaptureResult[str]) -> None:
    report = _quiet_json(out, captured)
    comparison = report["runs"][0]["outputs"][0]["fns"][0]
    assert comparison["fn"] == "allclose"
    assert comparison["rtol"] == 0 and comparison["atol"] == 0
    assert report["passed"] and comparison["passed"], (
        "K slicing, the (K, N) view of B and the accumulation across slices match torch"
    )


def _check_golden(out: Path, captured: pytest.CaptureResult[str], *, golden: Path) -> None:
    assert captured == ("", "")
    if golden.suffix == ".json":
        assert json.loads(out.read_text()) == json.loads(golden.read_text())
    else:
        assert out.read_bytes() == golden.read_bytes()


def _check_facts_declaration(
    out: Path,
    captured: pytest.CaptureResult[str],
    *,
    instruction: str,
) -> None:
    report = _quiet_json(out, captured)
    assert report["id"] == instruction
    assert report["target"] == "nvidia.h200_sxm"
    assert report["execution_mesh"]


def _check_facts_refusal(out: Path, captured: pytest.CaptureResult[str], *, message: str) -> None:
    assert captured.out == ""
    assert captured.err.startswith("tilefoundry: error: ")
    assert message in captured.err
    assert not out.exists()


def _check_finalized_rows(out: Path, captured: pytest.CaptureResult[str]) -> None:
    assert captured == ("", "")
    lowered = import_dsl(out.read_text())
    assert isinstance(lowered, PrimFunction)
    assert lowered.params[-1].type.shape == (128,)


def _check_finalize_unbound_dim(out: Path, captured: pytest.CaptureResult[str]) -> None:
    assert captured.out == ""
    assert "schedule finalize" in captured.err
    assert "bind it with --dim" in captured.err
    assert not out.exists()


def _check_reshard_locations(out: Path, captured: pytest.CaptureResult[str]) -> None:
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


def _check_reshape_matches_torch(out: Path, captured: pytest.CaptureResult[str]) -> None:
    report = _quiet_json(out, captured)
    assert report["passed"]
    comparison = report["runs"][0]["outputs"][0]["fns"][0]
    assert comparison["fn"] == "allclose" and comparison["passed"]


def _check_crossing_axes_refusal(out: Path, captured: pytest.CaptureResult[str]) -> None:
    assert captured.out == ""
    assert "layout positions (2, 8, 2, 4, 2, 4, 8) cross logical axes of (64, 128)" in captured.err
    assert "layout_crossing_axes.py:13:" in captured.err
    assert not out.exists()


def _check_selector_help(out: Path, captured: pytest.CaptureResult[str]) -> None:
    assert captured.err == ""
    assert "model.py[:Module[.function]]" in captured.out
    assert "child_module" not in captured.out


def _reshape_tensor(expected: bool = False) -> torch.Tensor:
    x = torch.arange(128, dtype=torch.float32).reshape(1, 8, 1, 16)
    return x.reshape(8, 16) if expected else x


FIXTURES = (
    CliCase(
        "reshard is charged only when an element moves",
        (
            "analyze",
            "tests/nvidia/cuda/fixtures/schedule/plain/reshard_locations.py",
            "{out}",
            "--memory",
            "--json",
        ),
        _check_reshard_locations,
    ),
    CliCase(
        "check binds two DimVars in a reshape",
        (
            "check",
            "tests/nvidia/cuda/fixtures/placed/two_dimvar_reshape.py:TwoDimVarReshape",
            "--inputs",
            "files:{tmp}/x.pt",
            "--expected",
            "{tmp}/expected.pt",
            "--dim",
            "T=8",
            "--dim",
            "C=16",
            "--device",
            "cpu",
            "--out",
            "output",
            "--fn",
            "allclose",
            "--atol",
            "0",
            "--rtol",
            "0",
            "--json",
            "{out}",
        ),
        _check_reshape_matches_torch,
        files={"x.pt": _reshape_tensor, "expected.pt": partial(_reshape_tensor, True)},
    ),
    CliCase(
        "finalize refuses a layout crossing logical axes where it is declared",
        (
            "schedule",
            "finalize",
            "tests/nvidia/cuda/fixtures/diagnostics/layout_crossing_axes.py",
            "{out}",
        ),
        _check_crossing_axes_refusal,
        code=1,
    ),
    CliCase(
        "analyze help names only Module.function",
        ("analyze", "--help"),
        _check_selector_help,
    ),
    CliCase(
        "finalize binds an open row dimension",
        (
            "schedule",
            "finalize",
            "tests/nvidia/cuda/fixtures/schedule/hir/dim/row_copy.py",
            "{out}",
            "--dim=rows=128",
        ),
        _check_finalized_rows,
    ),
    CliCase(
        "finalize refuses an unbound row dimension without output",
        (
            "schedule",
            "finalize",
            "tests/nvidia/cuda/fixtures/schedule/hir/dim/row_copy.py",
            "{out}",
        ),
        _check_finalize_unbound_dim,
        code=1,
    ),
    CliCase(
        "gather reads selected rows and caps the footprint",
        (
            "analyze",
            "tests/nvidia/cuda/fixtures/schedule/plain/sparse_decode.py",
            "{out}",
            "--memory",
            "--json",
        ),
        _check_selected_rows,
    ),
    CliCase(
        "integer and predicate work uses services",
        (
            "analyze",
            "tests/nvidia/cuda/fixtures/schedule/plain/sparse_decode.py",
            "{out}",
            "--compute-cost",
            "--json",
        ),
        _check_service_work,
    ),
    CliCase(
        "gather into smem is refused for a gmem result",
        (
            "schedule",
            "candidates",
            "tests/nvidia/cuda/fixtures/schedule/plain/sparse_decode.py",
            "{out}",
            "--json",
        ),
        _check_gmem_gather_refusal,
    ),
    CliCase(
        "analyzed smem peak matches the lowered placement",
        (
            "schedule",
            "finalize",
            "tests/nvidia/cuda/fixtures/schedule/hir/sparse_decode.py",
            "{out}",
        ),
        _check_smem_placement,
    ),
    *(
        CliCase(
            name,
            (
                "check",
                f"tests/nvidia/cuda/fixtures/schedule/{stage}/sparse_decode.py:SparseDecode",
                "--inputs",
                "files:{tmp}/q.pt,{tmp}/kv.pt,{tmp}/idx.pt",
                "--expected",
                "{tmp}/expected.pt",
                "--device",
                "cpu",
                "--out",
                "output",
                "--fn",
                "allclose",
                "--rtol",
                str(2**-7),
                "--atol",
                "0",
                "--json",
                "{out}",
            ),
            _check_attention,
            files={
                f"{key}.pt": partial(_attention_tensor, key)
                for key in ("q", "kv", "idx", "expected")
            },
        )
        for name, stage in (
            ("plain sparse attention matches torch", "plain"),
            ("scheduled sparse attention matches torch", "hir"),
        )
    ),
    *(
        CliCase(
            name,
            (
                "check",
                f"tests/nvidia/cuda/fixtures/schedule/{stage}/sm89_e4m3_gemm.py:SM89_E4M3_GEMM",
                "--inputs",
                "files:{tmp}/a.pt,{tmp}/b.pt",
                "--expected",
                "{tmp}/expected.pt",
                "--device",
                "cpu",
                "--out",
                "output",
                "--fn",
                "allclose",
                "--atol",
                "0",
                "--rtol",
                "0",
                "--json",
                "{out}",
            ),
            _check_e4m3_gemm,
            files={f"{key}.pt": partial(_e4m3_gemm_tensor, key) for key in ("a", "b", "expected")},
        )
        for name, stage in (
            ("plain e4m3 gemm matches torch", "plain"),
            ("scheduled e4m3 gemm matches torch", "hir"),
        )
    ),
    CliCase(
        "facts write the Wgmma declaration",
        ("schedule", "facts", "T.cuda.sm90.Wgmma", "--target", "nvidia.h200_sxm", "{out}"),
        partial(_check_golden, golden=WGMMA_FACTS),
    ),
    CliCase(
        "facts write the tiled MMA atom alternatives",
        ("schedule", "facts", "T.tiled_mma", "--target", "nvidia.h200_sxm", "{out}"),
        partial(_check_golden, golden=WGMMA_FACTS.with_name("tiled_mma.facts.txt")),
    ),
    *(
        CliCase(
            f"facts describe {instruction}",
            (
                "schedule",
                "facts",
                instruction,
                "--target",
                "nvidia.h200_sxm",
                "{out}",
                "--json",
            ),
            partial(_check_facts_declaration, instruction=instruction),
        )
        for instruction in FACT_INSTRUCTIONS
        if instruction not in {"T.cuda.sm90.Wgmma", "T.tiled_mma"}
    ),
    *(
        CliCase(
            name,
            ("schedule", "facts", "--target", target, "{out}", *options),
            partial(check_facts, expected=expected),
        )
        for name, target, options, expected in (
            ("facts list CUDA instructions as text", "nvidia.h200_sxm", (), _H200_FACTS_TEXT),
            (
                "facts list CUDA instructions as JSON",
                "nvidia.h200_sxm",
                ("--json",),
                _H200_FACTS_JSON,
            ),
            (
                "sm_89 facts list no WGMMA or TMA instruction",
                "nvidia.rtx_6000_ada",
                (),
                _SM89_FACTS_TEXT,
            ),
        )
    ),
    *(
        CliCase(
            name,
            ("schedule", "facts", instruction, "--target", target, "{out}"),
            partial(_check_facts_refusal, message=message),
            code=1,
        )
        for name, instruction, target, message in (
            (
                "facts refuse an unknown instruction without output",
                "T.cuda.sm90.Unknown",
                "nvidia.h200_sxm",
                "unknown instruction 'T.cuda.sm90.Unknown'",
            ),
            (
                "facts refuse an unknown target without output",
                "T.cuda.sm90.Wgmma",
                "nvidia.unknown",
                "unknown target identity 'nvidia.unknown'",
            ),
        )
    ),
    CliCase(
        "candidates write the canonical report",
        (
            "schedule",
            "candidates",
            f"tests/nvidia/cuda/fixtures/schedule/plain/{CANDIDATE_GOLDEN.name.removesuffix('.candidates.txt')}.py",
            "{out}",
        ),
        partial(_check_golden, golden=CANDIDATE_GOLDEN),
    ),
    CliCase(
        "analyze writes the memory report",
        (
            "analyze",
            "tests/nvidia/cuda/fixtures/schedule/hir/gemm_8192x17408x5120_tma_store.py",
            "{out}",
            "--memory",
        ),
        partial(_check_golden, golden=ANALYZED_GOLDEN),
    ),
)


def _check_plain_candidates(out, captured, *, source, sites):
    assert captured == ("", "")
    check_plain_report(json.loads(out.read_text()), source, sites)


def _check_invalid_chunks(out, captured):
    assert "[1, 32]" in captured.err
    assert not out.exists()


def _check_unbound_chunks(out, captured):
    assert "chunks is declared as [1, 32]" in captured.err
    assert "bind it with --dim" in captured.err
    assert not out.exists()


FIXTURES += (
    CliCase(
        "candidate sites at chunks=32",
        (
            "schedule",
            "candidates",
            "tests/nvidia/cuda/fixtures/schedule/plain/chunk_rmsnorm.py",
            "{out}",
            "--json",
            "--dim=chunks=32",
        ),
        partial(
            _check_plain_candidates,
            source="tests/nvidia/cuda/fixtures/schedule/plain/chunk_rmsnorm.py",
            sites=SiteCounts(0, 7, detailed=True, binaries=5),
        ),
    ),
    CliCase(
        "candidates refuses chunks outside declared range",
        (
            "schedule",
            "candidates",
            "tests/nvidia/cuda/fixtures/schedule/plain/chunk_rmsnorm.py",
            "{out}",
            "--dim=chunks=33",
        ),
        _check_invalid_chunks,
        code=1,
    ),
    CliCase(
        "candidates requires chunks binding",
        (
            "schedule",
            "candidates",
            "tests/nvidia/cuda/fixtures/schedule/plain/chunk_rmsnorm.py",
            "{out}",
        ),
        _check_unbound_chunks,
        code=1,
    ),
)


def test_candidates_nk_layout(tmp_path, capsys):
    source = Path("tests/nvidia/cuda/fixtures/schedule/plain/gemm_relu_gemm_untiled.py")
    variant = tmp_path / "gemm_relu_gemm_untiled_nk.py"
    variant.write_text(
        source.read_text()
        .replace("tf.matmul(a, b)", 'tf.matmul(a, b, b_layout="NK")')
        .replace("tf.matmul(y, c)", 'tf.matmul(y, c, b_layout="NK")')
    )
    out = tmp_path / "gemm_relu_gemm_untiled.json"
    assert cli_main(["schedule", "candidates", str(variant), str(out), "--json"]) == 0
    _check_plain_candidates(
        out, capsys.readouterr(), source=str(variant), sites=SiteCounts(2, 0, detailed=True)
    )


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case.name)
def test_cli_case(case: CliCase, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    check_cli_case(case, tmp_path, capsys)
