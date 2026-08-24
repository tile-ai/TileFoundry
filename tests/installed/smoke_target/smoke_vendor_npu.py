"""A document-free Target provider drives standard analysis."""

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


def test_document_free_target_analyzes_with_its_declared_target(
    tf, cmine, tmp_path
) -> None:
    model = _vendor_npu_model(cmine, tmp_path)
    analyzed = tf(
        "analyze",
        f"{model}:CMine.root",
        str(tmp_path / "report.json"),
        "--compute-cost",
        "--memory",
        "--roofline",
        "--json",
    )
    assert analyzed.returncode == 0, analyzed.stderr
    assert analyzed.stdout == ""
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["target"] == "vendor.npu"
    assert report["executed"] == ["compute-cost", "memory", "roofline"]
    assert report["function_records"]["roofline"]["ideal_ns"] > 0

    rejected = tf("analyze", f"{model}:CMine.root", str(tmp_path / "rejected.py"), "--performance")
    assert rejected.returncode == 1
    assert "performance:" in rejected.stderr
    assert "has no core execution domain" in rejected.stderr
