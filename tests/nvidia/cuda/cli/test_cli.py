from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import tilefoundry.cli.analyze as analyze_cli
import tilefoundry.cli.target as target_cli
from tests.core.cli._targets import listed_identities
from tilefoundry import cli
from tilefoundry.utils.units import format_bytes

_SMOKE_TARGET = Path(__file__).parents[1] / "installed" / "smoke_target"
FIXTURES = Path(__file__).parents[1] / "fixtures"


def _run_cli(registry: Path, cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tilefoundry.cli",
            "--registry",
            str(registry),
            *arguments,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _copy(source: Path, destination: Path) -> Path:
    destination.write_bytes(source.read_bytes())
    return destination


@pytest.mark.parametrize(
    ("root", "flags", "expected"),
    [
        ("Sound", (), 0),
        ("Sound", ("--compute-cost",), 0),
        ("Sound", ("--compute-cost", "--json"), 0),
        ("Unsound", ("--compute-cost",), 1),
    ],
)
def test_analyze_disarms_its_watchdog_on_every_exit(
    tmp_path, monkeypatch, root, flags, expected
) -> None:
    """Neither successful returns nor raised errors leave a timer armed in the caller."""
    source = FIXTURES / "cli/neighbours.py"
    disarmed = analyze_cli._watch(analyze_cli._ANALYSIS_TIMEOUT_SECONDS)
    monkeypatch.setattr(analyze_cli, "_watch", lambda limit: disarmed)
    assert not disarmed.is_set()

    assert (
        cli.main(["analyze", f"{source}:{root}", str(tmp_path / "report.txt"), *flags])
        == expected
    )
    assert disarmed.is_set()


def test_naming_one_root_does_not_ask_about_the_rest_of_its_file(tmp_path, capsys) -> None:
    """A selector names one root, and the rest of the file is a different question.

    The unsound root here states an entry naming no function it collected, and it
    is written first, so a reading that only survives when the broken one comes
    last is one that got away with it. Asking about the sound root has to answer
    about the sound root; asking about the file still has to refuse, because then
    the broken root is one of the programs that was asked for.
    """
    source = FIXTURES / "cli/neighbours.py"

    report_path = tmp_path / "sound.json"
    assert (
        cli.main(["analyze", f"{source}:Sound", str(report_path), "--compute-cost", "--json"]) == 0
    )
    assert capsys.readouterr().out == ""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["module"] == "Sound"
    assert report["functions"][0]['totals']["flops"]

    assert (
        cli.main(["analyze", str(source), str(tmp_path / "bad.json"), "--compute-cost", "--json"])
        == 1
    )
    assert "nope" in capsys.readouterr().err


def test_models_source_names_checkout_directory(capsys) -> None:
    assert cli.main(["models", "qwen3_1_7b", "--source"]) == 0
    source = capsys.readouterr()
    assert Path(source.out.splitlines()[0]).is_dir()
    assert not source.err
    assert "manifest" not in source.out


def test_target_list_expressions_execute_and_show_accepts_their_identities(
    capsys, monkeypatch
) -> None:
    identities = listed_identities(capsys, monkeypatch)
    assert {
        "cpu",
        "nvidia.b200_sxm",
        "nvidia.h200_sxm",
    } < set(identities)


def test_analysis_reports_distinguish_cuda_products(tmp_path, capsys) -> None:
    reports = {}
    for device in ("nvidia.h200_sxm", "nvidia.b200_sxm"):
        source = FIXTURES / f"cli/products/{device.rsplit('.', 1)[1]}.py"
        report_path = tmp_path / f"{device.rsplit('.', 1)[1]}.json"
        assert (
            cli.main(
                [
                    "analyze",
                    f"{source}:Model",
                    str(report_path),
                    "--compute-cost",
                    "--json",
                ]
            )
            == 0
        )
        assert capsys.readouterr().out == ""
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports[device] = report["target"]

    assert reports == {
        "nvidia.h200_sxm": "nvidia.h200_sxm",
        "nvidia.b200_sxm": "nvidia.b200_sxm",
    }


def test_persisted_targets_drive_every_command_without_touching_the_default_registry(
    tmp_path,
) -> None:
    registry = tmp_path / "registry.toml"
    default = target_cli.registry_path()
    default_before = default.read_bytes() if default.exists() else None
    architecture = _copy(
        _SMOKE_TARGET / "hw" / "vendor_sm70.toml",
        tmp_path / "vendor_sm70.toml",
    )
    device = _copy(
        _SMOKE_TARGET / "hw" / "vendor_v100_sxm2_32gb.toml",
        tmp_path / "vendor_v100_sxm2_32gb.toml",
    )
    provider = _copy(
        Path(__file__).parents[3]
        / "core"
        / "fixtures"
        / "target_plugins"
        / "vendor_npu"
        / "__init__.py",
        tmp_path / "vendor_npu.py",
    )

    added_architecture = _run_cli(
        registry, tmp_path, "target", "add", "--document", str(architecture)
    )
    assert added_architecture.returncode == 0, added_architecture.stderr
    assert "vendor.sm70" in added_architecture.stdout
    added_device = _run_cli(registry, tmp_path, "target", "add", "--document", str(device))
    assert added_device.returncode == 0, added_device.stderr
    assert 'CudaTarget("vendor.v100_sxm2_32gb")' in added_device.stdout
    added_provider = _run_cli(registry, tmp_path, "target", "add", str(provider))
    assert added_provider.returncode == 0, added_provider.stderr
    assert "VendorNpuTarget()" in added_provider.stdout
    assert "executed on every tilefoundry run" in added_provider.stdout

    stored = tomllib.loads(registry.read_text(encoding="utf-8"))
    assert [item["source"] for item in stored["document"]] == [
        str(architecture),
        str(device),
    ]
    assert stored["module"] == [{"source": str(provider)}]
    assert all(len(item["digest"]) == 64 for item in stored["document"])

    listed = _run_cli(registry, tmp_path, "target", "list")
    assert listed.returncode == 0, listed.stderr
    assert 'CudaTarget("vendor.v100_sxm2_32gb")' in listed.stdout
    assert "VendorNpuTarget()" in listed.stdout
    assert "identity: vendor.v100_sxm2_32gb   added" in listed.stdout
    assert "identity: vendor.npu   added" in listed.stdout
    assert "Added to this environment:" in listed.stdout
    assert "from vendor_npu import VendorNpuTarget" in listed.stdout

    reconstructed = subprocess.run(
        [
            sys.executable,
            "-c",
            "namespace = {}\n"
            "exec('from vendor_npu import VendorNpuTarget', namespace)\n"
            "target = eval('VendorNpuTarget()', namespace)\n"
            "assert target.identity == 'vendor.npu'\n",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert reconstructed.returncode == 0, reconstructed.stderr

    alternate_provider = tmp_path / "alternate" / provider.name
    alternate_provider.parent.mkdir()
    _copy(provider, alternate_provider)
    duplicate_module_name = _run_cli(registry, tmp_path, "target", "add", str(alternate_provider))
    assert duplicate_module_name.returncode == 1
    assert "module name 'vendor_npu' is already added from" in duplicate_module_name.stderr
    assert str(provider) in duplicate_module_name.stderr
    assert str(alternate_provider) not in registry.read_text(encoding="utf-8")

    shown = _run_cli(registry, tmp_path, "target", "show", "vendor.v100_sxm2_32gb")
    assert shown.returncode == 0, shown.stderr
    assert "architecture: vendor.sm70" in shown.stdout
    assert "device: vendor.v100_sxm2_32gb" in shown.stdout
    assert shown.stdout.count("  digest: ") == 2

    analyzed_npu = _run_cli(
        registry,
        tmp_path,
        "analyze",
        f"{FIXTURES / 'cli/registered_npu_source.py'}:Model",
        "--compute-cost",
        "--memory",
        "--roofline",
        str(tmp_path / "npu.json"),
        "--json",
    )
    assert analyzed_npu.returncode == 0, analyzed_npu.stderr
    assert analyzed_npu.stdout == ""
    assert json.loads((tmp_path / "npu.json").read_text(encoding="utf-8"))["target"] == "vendor.npu"
    analyzed_cuda = _run_cli(
        registry,
        tmp_path,
        "analyze",
        f"{FIXTURES / 'cli/registered_cuda_source.py'}:Model",
        "--compute-cost",
        str(tmp_path / "cuda.json"),
        "--json",
    )
    assert analyzed_cuda.returncode == 0, analyzed_cuda.stderr
    assert analyzed_cuda.stdout == ""
    assert (
        json.loads((tmp_path / "cuda.json").read_text(encoding="utf-8"))["target"]
        == "vendor.v100_sxm2_32gb"
    )

    removed_module = _run_cli(registry, tmp_path, "target", "remove", "vendor.npu")
    assert removed_module.returncode == 0, removed_module.stderr
    assert "identities: ['vendor.npu']" in removed_module.stdout
    removed_device = _run_cli(registry, tmp_path, "target", "remove", "vendor.v100_sxm2_32gb")
    assert removed_device.returncode == 0, removed_device.stderr
    after = _run_cli(registry, tmp_path, "target", "list")
    assert after.returncode == 0, after.stderr
    assert "VendorNpuTarget()" not in after.stdout
    assert 'CudaTarget("vendor.v100_sxm2_32gb")' not in after.stdout

    if default_before is None:
        assert not default.exists()
    else:
        assert default.read_bytes() == default_before


def test_registration_diagnostics_isolate_bad_entries_and_identity_sources(
    tmp_path,
) -> None:
    source_architecture = (_SMOKE_TARGET / "hw" / "vendor_sm70.toml").read_text(encoding="utf-8")
    source_device = (_SMOKE_TARGET / "hw" / "vendor_v100_sxm2_32gb.toml").read_text(
        encoding="utf-8"
    )

    missing_registry = tmp_path / "missing-registry.toml"
    missing_device = tmp_path / "missing_architecture.toml"
    missing_device.write_text(
        source_device.replace("vendor.sm70", "vendor.absent_sm70"),
        encoding="utf-8",
    )
    missing = _run_cli(
        missing_registry,
        tmp_path,
        "target",
        "add",
        "--document",
        str(missing_device),
    )
    assert missing.returncode == 1
    assert "needs architecture 'vendor.absent_sm70'; add it first" in missing.stderr
    assert not missing_registry.exists()

    registry = tmp_path / "registry.toml"
    architecture = tmp_path / "vendor_sm70.toml"
    architecture.write_text(source_architecture, encoding="utf-8")
    first = _run_cli(registry, tmp_path, "target", "add", "--document", str(architecture))
    assert first.returncode == 0, first.stderr
    duplicate = _run_cli(registry, tmp_path, "target", "add", "--document", str(architecture))
    assert duplicate.returncode == 1
    assert "hardware document 'vendor.sm70' is already registered" in duplicate.stderr

    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    json_provider = _copy(FIXTURES / "target_plugins/shadow/json.py", shadow_dir / "json.py")
    shadowed = _run_cli(
        tmp_path / "shadow-registry.toml",
        tmp_path,
        "target",
        "add",
        str(json_provider),
    )
    assert shadowed.returncode == 1
    assert "module name 'json' is already taken by" in shadowed.stderr
    assert "json/__init__.py" in shadowed.stderr
    assert not (tmp_path / "shadow-registry.toml").exists()

    collision_provider = _copy(FIXTURES / "target_plugins/collision.py", tmp_path / "collision.py")
    collision = _run_cli(registry, tmp_path, "target", "add", str(collision_provider))
    assert collision.returncode == 1
    assert "target identity 'nvidia.h200_sxm' is already occupied by" in collision.stderr
    assert 'CudaTarget("nvidia.h200_sxm")' in collision.stderr
    stored = tomllib.loads(registry.read_text(encoding="utf-8"))
    assert "module" not in stored

    named_provider = _copy(FIXTURES / "target_plugins/named_provider.py", tmp_path / "named_provider.py")
    named_registry = tmp_path / "named-registry.toml"
    named = _run_cli(named_registry, tmp_path, "target", "add", "named_provider")
    assert named.returncode == 0, named.stderr
    assert "NamedTarget()" in named.stdout
    named_stored = tomllib.loads(named_registry.read_text(encoding="utf-8"))
    assert named_stored == {"module": [{"name": "named_provider"}]}
    named_list = _run_cli(named_registry, tmp_path, "target", "list")
    assert named_list.returncode == 0, named_list.stderr
    assert "identity: vendor.named   added" in named_list.stdout
    _copy(FIXTURES / "target_plugins/changed_provider.py", named_provider)
    named_removed = _run_cli(named_registry, tmp_path, "target", "remove", "named_provider")
    assert named_removed.returncode == 0, named_removed.stderr
    assert "module 'named_provider' failed: provider changed" in named_removed.stderr
    assert "identities: []" in named_removed.stdout

    bad_architecture = tmp_path / "vendor_bad_sm70.toml"
    bad_architecture.write_text(
        source_architecture.replace('id = "vendor.sm70"', 'id = "vendor.bad_sm70"'),
        encoding="utf-8",
    )
    good_device = tmp_path / "vendor_v100_sxm2_32gb.toml"
    good_device.write_text(source_device, encoding="utf-8")
    added_bad = _run_cli(registry, tmp_path, "target", "add", "--document", str(bad_architecture))
    assert added_bad.returncode == 0, added_bad.stderr
    added_good = _run_cli(registry, tmp_path, "target", "add", "--document", str(good_device))
    assert added_good.returncode == 0, added_good.stderr
    bad_architecture.write_text(
        bad_architecture.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    listed = _run_cli(registry, tmp_path, "target", "list")
    assert listed.returncode == 0
    assert "document 'vendor.bad_sm70' content changed" in listed.stderr
    assert 'CudaTarget("vendor.v100_sxm2_32gb")' in listed.stdout
    analyzed = _run_cli(
        registry,
        tmp_path,
        "analyze",
        f"{FIXTURES / 'cli/registered_cuda_source.py'}:Model",
        "--compute-cost",
        str(tmp_path / "repaired.json"),
        "--json",
    )
    assert analyzed.returncode == 0
    assert "document 'vendor.bad_sm70' content changed" in analyzed.stderr
    assert analyzed.stdout == ""
    assert (
        json.loads((tmp_path / "repaired.json").read_text(encoding="utf-8"))["target"]
        == "vendor.v100_sxm2_32gb"
    )

    repaired = _run_cli(registry, tmp_path, "target", "add", "--document", str(bad_architecture))
    assert repaired.returncode == 0, repaired.stderr
    repaired_list = _run_cli(registry, tmp_path, "target", "list")
    assert repaired_list.returncode == 0, repaired_list.stderr
    assert "vendor.bad_sm70" in repaired_list.stdout


def test_analyze_binds_an_extent_on_a_root_that_reaches_a_child(tmp_path, capsys) -> None:
    """Choosing a size for a composed root keeps its child call activation-only."""
    source = FIXTURES / "cli/composed_leaf_n_cli.py"

    report_path = tmp_path / "composed_report.py"
    assert (
        cli.main(["analyze", f"{source}:Composed.root", str(report_path), "--dim", "n_cli=4"]) == 0
    )
    assert capsys.readouterr().out == ""
    expanded = report_path.read_text(encoding="utf-8")
    assert "def root(" in expanded
    assert "w: ConstTensor" in expanded
    assert "leaf(x)" in expanded


def test_analyze_reports_retained_occurrences_from_one_rendering(tmp_path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "placed" / "moe_mega_kernel.py"
    selector = f"{source}:MoEMegaKernel"
    flags = ["--compute-cost", "--memory", "--roofline", "--performance"]

    json_path = tmp_path / "report.json"
    first_path = tmp_path / "first.py"
    second_path = tmp_path / "second.py"
    operands_path = tmp_path / "operands.py"
    operands_json_path = tmp_path / "operands.json"
    assert cli.main(["analyze", selector, str(json_path), *flags, "--json"]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert cli.main(["analyze", selector, str(first_path), *flags]) == 0
    first = first_path.read_text(encoding="utf-8")
    assert cli.main(["analyze", selector, str(second_path), *flags]) == 0
    second = second_path.read_text(encoding="utf-8")
    assert first == second

    assert cli.main(["analyze", selector, str(operands_path), *flags, "--operands"]) == 0
    asked = operands_path.read_text(encoding="utf-8")
    assert "operands=" not in first
    assert "operands=0:r30.00KiB/w0;result:r0/w30.00KiB" in asked
    assert (
        cli.main(["analyze", selector, str(operands_json_path), *flags, "--operands", "--json"])
        == 0
    )
    assert json.loads(operands_json_path.read_text(encoding="utf-8")) == payload

    header, annotated = first.split("\n\n", 1)
    lines = annotated.splitlines()
    assert (
        payload["requested"]
        == payload["executed"]
        == [
            "compute-cost",
            "memory",
            "roofline",
            "performance",
        ]
    )
    assert set(payload["functions"][0]["function_records"]) == {
        "compute-cost",
        "memory",
        "roofline",
        "performance",
    }
    assert "v0 = routed_expert(tokens)" in annotated
    assert "v1 = shared_expert(tokens)" in annotated
    assert "function=routed_expert" in annotated
    assert "function=shared_expert" in annotated
    assert annotated.count("reshard(tokens") == 2
    assert "local = tf.reshard(tokens" in annotated
    assert annotated.count("local = tf.reshard(tokens") == 2
    assert "with mesh[:120] as cta:" in annotated
    assert "with mesh[120:] as cta:" in annotated

    summary = payload["functions"][0]["function_records"]["performance"]
    cost = payload["functions"][0]["function_records"]["compute-cost"]
    moved = payload["functions"][0]["function_records"]["memory"]
    bound = payload["functions"][0]["function_records"]["roofline"]
    assert {name: held["gmem"]["logical"] for name, held in moved["footprint"]["buffers"].items()} == {
        "tokens": 2560,
        "routed_expert::placed": 30976,
        "shared_expert::placed": 33280,
        "v2": 30720,
    }
    assert "cache" not in payload
    assert (
        not {
            "cache_level",
            "cache_capacity_bytes",
            "wave_units",
            "declared_units",
        }
        & moved.keys()
    )
    assert header.splitlines() == [
        "# analysis target=nvidia.h200_sxm module=MoEMegaKernel function=experts "
        f"topology={payload['topology']}",
        f"# selection requested={','.join(payload['requested'])} "
        f"executed={','.join(payload['executed'])}",
        "# compute-cost "
        f"flops=f32:{cost['flops']['f32']['logical']}@logical,"
        f"{cost['flops']['f32']['total']}@total,"
        f"{cost['flops']['f32']['per_unit'][0]}@{payload['topology']} precision=exact",
        "# memory traffic=gmem:r120.00KiB/w90.00KiB@logical,"
        "r7.79MiB/w3.93MiB@total,r62.75KiB/w32.75KiB@cta "
        "footprint=tokens:2.50KiB;routed_expert::placed:30.25KiB;shared_expert::placed:32.50KiB;v2:30.00KiB "
        "footprint-precision=exact peak=gmem:90.00KiB persistent=gmem:30.00KiB",
        f"# roofline ideal-ns={bound['ideal_ns']} bound-by={bound['bound_by']}",
        "# performance root=MoEMegaKernel::experts "
        f"predicted-ns={summary['timeline']['end_ns']} wave=132/132",
    ]
    assert payload["functions"][0]["totals"]["flops"] == {
        name: spread["total"] for name, spread in cost["flops"].items()
    }
    assert payload["functions"][0]["totals"]["traffic"] == {
        name: value["total"] for name, value in moved["traffic"]["storage"].items()
    }

    hoisted = {line.split(" = ", 1)[0] for line in lines if " = Mesh((Topology(" in line}
    scoped = {
        line.rsplit(" as ", 1)[1].split(":", 1)[0]
        for line in lines
        if line.lstrip().startswith("with ")
    }
    assert hoisted == set()
    assert scoped == {"cta"}
    annotated_types = [
        line.split("  # ", 1)[1].split("; ", 1)[0] for line in lines if "  # Tensor[" in line
    ]
    assert 'Tensor[(120, 64), "f32", (120 @ cta.tile, 64)]' in annotated_types
    assert 'Tensor[(120, 64), "f32", (12 @ cta.tile, 10, 64)]' in annotated_types
    assert 'Tensor[(120, 64), "f32"]' in annotated_types

    rows = [row for function in payload["functions"] for row in function["calls"]]
    assert len(rows) == 9
    cost_rows = [row for row in rows if "compute-cost" in row]
    assert len(cost_rows) == 9
    assert all(
        set(row) - {"performance"} == {"value", "compute-cost", "memory", "roofline"}
        for row in cost_rows
    )
    assert [row["value"] for row in payload["functions"][0]["calls"][:2]] == ["v0:9", "v1:10"]
    timed = [index for index, row in enumerate(rows) if "performance" in row]
    assert timed == [0, 1, 2, 4, 7]
    rendered = {
        function["function"]: block.rstrip().splitlines()
        for function, block in zip(
            payload["functions"],
            re.split(r"(?m)(?=^from __future__ import annotations$)", annotated)[1:],
            strict=True,
        )
    }
    for function, row in (
        (function, row) for function in payload["functions"] for row in function["calls"]
    ):
        value, line_text = row["value"].rsplit(":", 1)
        body = rendered[function["function"]]
        start = int(line_text)
        assert body[start - 1].lstrip().startswith(f"{value} = ")
        later = [
            int(other["value"].rsplit(":", 1)[1]) - 1
            for other in function["calls"]
            if int(other["value"].rsplit(":", 1)[1]) > start
        ]
        stop = min(later, default=len(body))
        statement = "\n".join(body[start - 1 : stop])
        if "performance" in row:
            timeline = row["performance"]["timeline"]
            assert f"performance=[{timeline['start_ns']},{timeline['end_ns']})" in statement
        else:
            assert "; performance=" not in statement
        annotated_meshes = set(re.findall(r"@ (\w+)\.", statement.split("  # ", 1)[1]))
        assert annotated_meshes <= scoped | {"mesh"}
        placed_meshes = set(re.findall(r"mesh=(\w+),", statement))
        assert placed_meshes <= scoped | {"mesh"}
    assert len([line for line in lines if "; performance=" in line]) == len(timed)
    assert len(
        {
            (function["function"], row["value"])
            for function in payload["functions"]
            for row in function["calls"]
        }
    ) == len(rows)
    assert "units=" not in annotated


def test_analyze_without_a_path_writes_the_report_to_stdout(capsys) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "placed" / "rmsnorm.py"
    assert cli.main(["analyze", f"{source}:RmsnormModule", "--memory"]) == 0
    assert "# memory traffic=" in capsys.readouterr().out


def test_analyze_memory_lines_state_json_totals(tmp_path) -> None:
    """Each Function's # memory @total is that Function's JSON totals."""
    source = Path(__file__).parents[1] / "fixtures" / "placed" / "moe_mega_kernel.py"
    selector = f"{source}:MoEMegaKernel"
    flags = ["--compute-cost", "--memory"]

    json_path = tmp_path / "report.json"
    text_path = tmp_path / "report.py"
    assert cli.main(["analyze", selector, str(json_path), *flags, "--json"]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert cli.main(["analyze", selector, str(text_path), *flags]) == 0
    text = text_path.read_text(encoding="utf-8")

    memory_lines = {}
    for line in text.splitlines():
        if identity := re.match(r"# analysis .* module=(\S+) function=(\S+)", line):
            current = ".".join(identity.groups())
        elif line.startswith("# memory traffic="):
            memory_lines[current] = line
    functions = payload["functions"]
    assert len(functions) > 1, "one Function has no other record to be confused with"
    assert set(memory_lines) == {row["function"] for row in functions}
    for row in functions:
        line = memory_lines[row["function"]]
        traffic = re.search(r"traffic=(\S+)", line).group(1)
        segments = {
            segment.split(":", 1)[0]: segment.split(":", 1)[1]
            for segment in traffic.split(";")
        }
        for level, moved in row["totals"]["traffic"].items():
            total = f"r{format_bytes(moved['read'])}/w{format_bytes(moved['write'])}@total"
            assert total in segments[level].split(",")
    entry = functions[0]
    for row in functions[1:]:
        for level, moved in row["totals"]["traffic"].items():
            whole = entry["totals"]["traffic"][level]
            assert whole["read"] >= moved["read"] and whole["write"] >= moved["write"]
