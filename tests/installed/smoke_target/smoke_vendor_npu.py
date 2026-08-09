"""A document-free Target provider drives standard analysis and its scheduler."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _vendor_npu_model(cmine: Path, tmp_path: Path) -> Path:
    provider = tmp_path / "vendor_npu"
    shutil.copytree(Path(__file__).parent / "vendor_npu", provider)
    model = tmp_path / "vendor_npu_model.py"
    model.write_text(
        cmine.read_text(encoding="utf-8")
        .replace(
            "from tilefoundry.target import CudaTarget",
            "from vendor_npu import VendorNpuTarget",
        )
        .replace(
            '@module(entry="root", target=CudaTarget("nvidia.h200_sxm"), '
            'topologies=(Topology("cta", 1), Topology("thread", 128)))',
            '@module(entry="root", target=VendorNpuTarget(), topologies=(Topology("core", 1),))',
        ),
        encoding="utf-8",
    )
    return model


def test_document_free_target_analyses_and_selects_its_scheduler(
    tf, cmine, tmp_path, monkeypatch
) -> None:
    model = _vendor_npu_model(cmine, tmp_path)
    scheduler_calls = tmp_path / "scheduler_calls.txt"
    monkeypatch.setenv("TF_VENDOR_NPU_SCHEDULER_CALLS", str(scheduler_calls))
    analyzed = tf(
        "analyze",
        f"{model}:CMine.root",
        "--compute-cost",
        "--memory",
        "--roofline",
        "--timeline",
        "--json",
    )
    assert analyzed.returncode == 0, analyzed.stderr
    report = json.loads(analyzed.stdout)
    assert report["target"] == "vendor.npu"
    assert report["executed"] == ["compute-cost", "memory", "roofline", "timeline"]
    assert report["function_records"]["roofline"]["ideal_ns"] > 0
    assert report["function_records"]["timeline"]["grid_units"] == 1

    scheduled = tf(
        "schedule",
        f"{model}:CMine.root",
        "--topology",
        "core",
        "--json",
    )
    assert scheduled.returncode == 0, scheduled.stderr
    assert json.loads(scheduled.stdout) == {"topology": "core", "extent": 1}
    assert scheduler_calls.read_text(encoding="utf-8") == "core\n"
