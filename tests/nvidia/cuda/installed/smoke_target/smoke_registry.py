"""Persistent target entries drive every installed command from one registry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def test_registered_targets_drive_commands(
    tf, installation, tmp_path
) -> None:
    default_registry = installation / "share" / "tilefoundry" / "registry.toml"
    default_before = default_registry.read_bytes() if default_registry.exists() else None
    registry = tmp_path / "registry.toml"
    fixtures = Path(__file__).parent
    architecture = tmp_path / "vendor_sm70.toml"
    device = tmp_path / "vendor_v100_sxm2_32gb.toml"
    provider = tmp_path / "vendor_npu.py"
    shutil.copyfile(fixtures / "hw" / architecture.name, architecture)
    shutil.copyfile(fixtures / "hw" / device.name, device)
    provider_fixture = (
        Path(__file__).parents[4] / "core" / "fixtures" / "target_plugins" / "vendor_npu"
    )
    shutil.copyfile(provider_fixture / "__init__.py", provider)
    models = Path(__file__).parents[2] / "fixtures" / "cli"

    for source, document in ((architecture, True), (device, True), (provider, False)):
        arguments = ["--registry", registry, "target", "add"]
        if document:
            arguments.append("--document")
        arguments.append(source)
        added = tf(*arguments)
        assert added.returncode == 0, added.stderr

    listed = tf("--registry", registry, "target", "list")
    assert listed.returncode == 0, listed.stderr
    assert 'CudaTarget("vendor.v100_sxm2_32gb")' in listed.stdout
    assert "identity: vendor.v100_sxm2_32gb   added" in listed.stdout
    assert "VendorNpuTarget()" in listed.stdout
    assert "identity: vendor.npu   added" in listed.stdout
    shown = tf("--registry", registry, "target", "show", "vendor.v100_sxm2_32gb")
    assert shown.returncode == 0, shown.stderr
    assert "device: vendor.v100_sxm2_32gb" in shown.stdout
    assert shown.stdout.count("  digest: ") == 2

    analyzed_npu = tf(
        "--registry",
        registry,
        "analyze",
        f"{models / 'registered_npu_installed.py'}:Model",
        str(tmp_path / "npu.json"),
        "--compute-cost",
        "--memory",
        "--roofline",
        "--json",
    )
    assert analyzed_npu.returncode == 0, analyzed_npu.stderr
    assert analyzed_npu.stdout == ""
    npu_report = json.loads((tmp_path / "npu.json").read_text(encoding="utf-8"))
    assert npu_report["target"] == "vendor.npu"
    assert npu_report["executed"] == ["compute-cost", "memory", "roofline"]
    analyzed_cuda = tf(
        "--registry",
        registry,
        "analyze",
        f"{models / 'registered_cuda_installed.py'}:Model",
        str(tmp_path / "cuda.json"),
        "--compute-cost",
        "--json",
    )
    assert analyzed_cuda.returncode == 0, analyzed_cuda.stderr
    assert analyzed_cuda.stdout == ""
    assert json.loads((tmp_path / "cuda.json").read_text(encoding="utf-8"))["target"] == "vendor.v100_sxm2_32gb"

    removed = tf("--registry", registry, "target", "remove", "vendor.npu")
    assert removed.returncode == 0, removed.stderr
    assert "identities: ['vendor.npu']" in removed.stdout
    after = tf("--registry", registry, "target", "list")
    assert after.returncode == 0, after.stderr
    assert "VendorNpuTarget()" not in after.stdout
    assert 'CudaTarget("vendor.v100_sxm2_32gb")' in after.stdout

    if default_before is None:
        assert not default_registry.exists()
    else:
        assert default_registry.read_bytes() == default_before
