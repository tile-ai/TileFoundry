"""An installed external Target can analyse a copied shipped model."""

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
    provider = Path(__file__).with_name("v100.py")
    shutil.copy2(provider, copied / "v100.py")

    model = copied / "model.py"
    model.write_text(
        model.read_text(encoding="utf-8")
        .replace("from tilefoundry.target import CudaTarget", "from v100 import V100Target")
        .replace('CudaTarget("nvidia.h200_sxm")', "V100Target()"),
        encoding="utf-8",
    )
    config = copied / "config.json"
    configured = json.loads(config.read_text(encoding="utf-8"))
    configured["torch_dtype"] = "float16"
    config.write_text(json.dumps(configured), encoding="utf-8")
    return model


def test_external_v100_target_analyses_a_copied_installed_model(
    tf, tmp_path
) -> None:
    model = _v100_qwen(tf, tmp_path)
    done = tf(
        "analyze",
        f"{model}:Qwen3_1_7B.layer0.mlp",
        "--compute-cost",
        "--memory",
        "--roofline",
        "--timeline",
        "--json",
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)

    assert report["target"] == "nvidia.v100_sxm2_32gb"
    assert report["executed"] == ["compute-cost", "memory", "roofline", "timeline"]
    assert report["totals"]["flops"]["f16"] > 0
    assert report["function_records"]["roofline"]["ideal_ns"] > 0
    gmem = next(
        item for item in report["function_records"]["memory"]["footprint"]
        if item["level"] == "gmem"
    )
    assert gmem["peak_bytes"] < 32_000_000_000
    timeline = report["function_records"]["timeline"]
    assert timeline["grid_units"] == 1
    call_timelines = [call["timeline"] for call in report["calls"]]
    assert call_timelines
    assert {call["grid_units"] for call in call_timelines} == {1}
    assert {call["waves"] for call in call_timelines} == {1}
    assert timeline["waves"] == sum(call["waves"] for call in call_timelines)

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

    service_calls = (model.parent / "v100_service_calls.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert {
        "analyzer:compute-cost",
        "analyzer:memory",
        "analyzer:roofline",
        "analyzer:timeline",
        "scheduler:thread",
    } <= set(service_calls)
