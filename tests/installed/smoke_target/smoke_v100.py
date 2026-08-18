"""External CUDA documents can analyse and schedule a copied shipped model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _v100_qwen(tf, tmp_path: Path) -> Path:
    located = tf("models", "qwen3_1_7b", "--source")
    assert located.returncode == 0, located.stderr
    source = Path(located.stdout.splitlines()[0])
    assert source.is_absolute()
    copied = tmp_path / "qwen3_1_7b_v100"
    shutil.copytree(source, copied)
    shutil.copytree(Path(__file__).parent / "hw", copied / "hw")

    model = copied / "model.py"
    model.write_text(
        model.read_text(encoding="utf-8")
        .replace(
            "from tilefoundry.target import CudaTarget",
            "from pathlib import Path\nfrom tilefoundry.target import CudaTarget",
        )
        .replace(
            'CudaTarget("nvidia.h200_sxm")',
            'CudaTarget(\n'
            '    Path(__file__).parent / "hw" / "vendor_v100_sxm2_32gb.toml",\n'
            '    Path(__file__).parent / "hw" / "vendor_sm70.toml",\n'
            ')',
        ),
        encoding="utf-8",
    )
    config = copied / "config.json"
    configured = json.loads(config.read_text(encoding="utf-8"))
    configured["torch_dtype"] = "float16"
    config.write_text(json.dumps(configured), encoding="utf-8")
    return model


def test_external_v100_documents_analyse_a_copied_installed_model(
    tf, tmp_path
) -> None:
    model = _v100_qwen(tf, tmp_path)
    done = tf(
        "analyze",
        f"{model}:Qwen3_1_7B.layer0.placed_mlp",
        "--compute-cost",
        "--memory",
        "--roofline",
        "--performance",
        "--json",
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)

    assert report["target"] == "vendor.v100_sxm2_32gb"
    assert report["executed"] == ["compute-cost", "memory", "roofline", "performance"]
    assert report["totals"]["flops"]["f16"] > 0
    assert report["function_records"]["roofline"]["ideal_ns"] > 0
    gmem = next(
        item for item in report["function_records"]["memory"]["footprint"]
        if item["level"] == "gmem"
    )
    assert gmem["peak_bytes"] < 32_000_000_000
    record = report["function_records"]["performance"]
    assert record["waves"] == 2
    assert report["function_records"]["memory"]["allocation"]["solver_status"] in (
        "optimal",
        "feasible",
    )
    call_records = [
        call["performance"]["timeline"] for call in report["calls"] if "performance" in call
    ]
    assert call_records
    assert all(
        set(call) == {"start_ns", "end_ns", "trips", "stride_ns"}
        for call in call_records
    )
    assert all(call["end_ns"] > call["start_ns"] for call in call_records)
    assert record["timeline"] == {
        "start_ns": 0,
        "end_ns": 2 * max(call["end_ns"] for call in call_records),
        "trips": 1,
        "stride_ns": 0,
    }

    scheduled = tf(
        "schedule",
        f"{model}:Qwen3_1_7B.layer0.mlp",
        "--topology",
        "thread",
        "--solver-timeout=60",
        "--solver-workers=4",
        "--first-plan",
    )
    assert scheduled.returncode == 0, scheduled.stderr
