"""A document-free Target provider drives standard analysis."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _vendor_npu_model(tmp_path: Path) -> Path:
    provider = tmp_path / "vendor_npu"
    provider_fixture = (
        Path(__file__).parents[4] / "core" / "fixtures" / "target_plugins" / "vendor_npu"
    )
    shutil.copytree(provider_fixture, provider)
    model = tmp_path / "vendor_npu_model.py"
    shutil.copyfile(Path(__file__).parents[2] / "fixtures/cli/vendor_npu/model.py", model)
    return model


def test_document_free_target_analyzes_with_its_declared_target(tf, tmp_path) -> None:
    model = _vendor_npu_model(tmp_path)
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
    assert report["functions"][0]['function_records']["roofline"]["ideal_ns"] > 0

    rejected = tf("analyze", f"{model}:CMine.root", str(tmp_path / "rejected.py"), "--performance")
    assert rejected.returncode == 1
    assert "performance:" in rejected.stderr
