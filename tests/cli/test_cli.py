from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import tilefoundry.cli.target as target_cli
from tests.fixtures.shapes.composed_leaf_source import composed_leaf_source
from tilefoundry import cli
from tilefoundry.cli.source import (
    load_authored_ir,
    one_extent_per_dim,
)
from tilefoundry.target import CpuTarget, registered_targets


class ListedCpuTarget(CpuTarget):
    name = "tests.cli.listed_cpu"


_SMOKE_TARGET = Path(__file__).parents[1] / "installed" / "smoke_target"


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


def _write_registered_model(
    path: Path, *, target: str, topology: str
) -> Path:
    path.write_text(
        "import json\n"
        "from tilefoundry import func\n"
        "from tilefoundry.dsl import Tensor, Topology, tf\n"
        "from tilefoundry.target import CudaTarget, registered_targets\n"
        "_extent = json.loads('{\"extent\": 1}')[\"extent\"]\n"
        f"_target = {target}\n"
        f"@func(target=_target, topologies=(Topology('{topology}', _extent),))\n"
        "def model(source: Tensor[(8,), 'f32']):\n"
        "    return tf.add(source, source)\n",
        encoding="utf-8",
    )
    return path


def test_parse_dims_reads_one_extent_per_dimension() -> None:
    """Nothing stated is not the same as nothing chosen."""
    assert cli.parse_dims(None) is None
    assert cli.parse_dims([]) == {}
    assert cli.parse_dims(["ctx_len=1024"]) == {"ctx_len": (1024,)}
    assert cli.parse_dims(["ctx_len=8", "seq_len=1"]) == {"ctx_len": (8,), "seq_len": (1,)}
    assert cli.parse_dims(["ctx_len=0,1,37"]) == {"ctx_len": (0, 1, 37)}

    assert one_extent_per_dim(cli.parse_dims(None)) is None
    assert one_extent_per_dim(cli.parse_dims([])) == {}
    assert one_extent_per_dim(cli.parse_dims(["ctx_len=1024"])) == {"ctx_len": 1024}
    with pytest.raises(ValueError, match="asking several EXTENTs together is for check"):
        one_extent_per_dim(cli.parse_dims(["ctx_len=0,1,37"]))


@pytest.mark.parametrize(
    "stated",
    [["=8"], ["ctx_len=1.5"]],
)
def test_parse_dims_rejects_an_argument_that_states_no_extent(stated) -> None:
    with pytest.raises(ValueError):
        cli.parse_dims(stated)


def test_parse_dims_rejects_one_dimension_stated_twice() -> None:
    """Repeating the flag states another dimension, not another value for one already stated.

    Repeating the flag states another dimension, not another value for one
    already stated.

    Taking the last would answer an ambiguous request by picking silently, and
    the caller would be told nothing -- which is the failure worth catching,
    because both numbers came from them.
    """
    with pytest.raises(ValueError, match="ctx_len was given twice"):
        cli.parse_dims(["ctx_len=8", "ctx_len=512"])

    with pytest.raises(ValueError, match="ctx_len was given twice"):
        cli.parse_dims(["ctx_len=8", "ctx_len=8"])


def test_schedule_rejects_several_extents_per_dimension(capsys) -> None:
    argv = ["schedule", "missing.py", "--topology", "cta", "--dim", "ctx_len=0,1"]
    assert cli.main(argv) == 1
    refused = capsys.readouterr().err
    assert "ctx_len takes one EXTENT at a time" in refused
    assert "asking several EXTENTs together is for check" in refused


_NEIGHBOURS = """from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.target import CudaTarget

N = 132 * 128
_H200 = CudaTarget('nvidia.h200_sxm')


@module(entry='nope', target=_H200, topologies=(Topology('cta', 132),))
class Unsound:
    @func
    def kernel(x: Tensor[(N,), 'f32']) -> Tensor[(N,), 'f32']:
        return tf.square(x)


@module(entry='kernel', target=_H200, topologies=(Topology('cta', 132),))
class Sound:
    @func
    def kernel(x: Tensor[(N,), 'f32']) -> Tensor[(N,), 'f32']:
        with Mesh(('cta',), layout=(132,), names=('block',)) as m:
            placed = tf.reshard(x, (N @ m.block,), 'gmem')
            return tf.reshard(tf.square(placed), (N @ m.block,), 'gmem')
"""


def test_naming_one_root_does_not_ask_about_the_rest_of_its_file(tmp_path, capsys) -> None:
    """A selector names one root, and the rest of the file is a different question.

    The unsound root here states an entry naming no function it collected, and it
    is written first, so a reading that only survives when the broken one comes
    last is one that got away with it. Asking about the sound root has to answer
    about the sound root; asking about the file still has to refuse, because then
    the broken root is one of the programs that was asked for.
    """
    source = tmp_path / "neighbours.py"
    source.write_text(_NEIGHBOURS, encoding="utf-8")

    assert cli.main(["analyze", f"{source}:Sound", "--compute-cost", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["module"] == "Sound"
    assert report["totals"]["flops"]

    assert cli.main(["analyze", str(source), "--compute-cost", "--json"]) == 1
    assert "nope" in capsys.readouterr().err


def test_analyze_help_explains_topology_effects_and_assumptions(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["analyze", "--help"])

    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for family in ("compute-cost", "memory", "roofline", "performance"):
        assert family in help_text
    assert "flops_per_unit" in help_text
    assert "per-unit traffic" in help_text
    assert "global traffic is the device's and counted once" in help_text
    assert "is an observation, not a bound" in help_text


def test_analyze_json_without_a_requested_root_is_a_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["analyze", "missing.py", "--json"])

    assert stopped.value.code == 2
    refused = capsys.readouterr()
    assert refused.out == ""
    assert refused.err.startswith(
        "tilefoundry analyze: error: --json requires at least one analysis flag:"
    )
    assert "usage: tilefoundry analyze" in refused.err
    assert "source file not found" not in refused.err


def test_target_list_expressions_execute_and_show_accepts_their_identities(
    capsys, monkeypatch
) -> None:
    builtins = target_cli.available_targets()
    monkeypatch.setattr(
        target_cli,
        "available_targets",
        lambda: (*builtins, ListedCpuTarget()),
    )
    assert cli.main(["target", "list"]) == 0
    listed = capsys.readouterr()
    namespace: dict[str, object] = {}
    for import_line in (
        line for line in listed.out.splitlines() if line.startswith("from ")
    ):
        exec(import_line, namespace)

    rows = [line.strip() for line in listed.out.splitlines() if "  identity: " in line]
    identities = []
    for row in rows:
        expression, identity = row.rsplit("  identity: ", 1)
        target = eval(expression.rstrip(), namespace)
        assert target.identity == identity
        identities.append(identity)

        assert cli.main(["target", "show", identity]) == 0
        shown = capsys.readouterr()
        assert shown.err == ""
        if "device:" in shown.out:
            assert f"device: {identity}\n" in shown.out
            assert shown.out.count("  digest: ") == 2
        else:
            assert shown.out == (
                f"identity: {identity}\n{expression.rstrip()}\n"
                "facts: unavailable\n"
            )

    assert {
        "apple.m2_pro",
        "cpu",
        "nvidia.b200_sxm",
        "nvidia.h200_sxm",
    } < set(identities)
    assert "tests.cli.listed_cpu" in identities


def test_analysis_reports_distinguish_cuda_products(tmp_path, capsys) -> None:
    reports = {}
    for device in ("nvidia.h200_sxm", "nvidia.b200_sxm"):
        source = tmp_path / f"{device.rsplit('.', 1)[1]}.py"
        source.write_text(
            "from tilefoundry import func\n"
            "from tilefoundry.dsl import Tensor, Topology, tf\n"
            "from tilefoundry.target import CudaTarget\n"
            f"@func(target=CudaTarget('{device}'), "
            "topologies=(Topology('cta', 1),))\n"
            "def model(source: Tensor[(8,), 'f32']):\n"
            "    return tf.add(source, source)\n",
            encoding="utf-8",
        )
        assert cli.main(["analyze", f"{source}:model", "--compute-cost", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        reports[device] = report["target"]

    assert reports == {
        "nvidia.h200_sxm": "nvidia.h200_sxm",
        "nvidia.b200_sxm": "nvidia.b200_sxm",
    }


def test_repeated_source_loads_keep_one_logical_target_registration(tmp_path) -> None:
    (tmp_path / "provider.py").write_text(
        "from tilefoundry.target import CpuTarget, register_target\n"
        "@register_target\n"
        "class ReloadTarget(CpuTarget):\n"
        "    name = 'tests.cli.reload_target'\n",
        encoding="utf-8",
    )
    source = tmp_path / "model.py"
    source.write_text(
        "from tilefoundry import module\n"
        "from provider import ReloadTarget\n"
        "@module(target=ReloadTarget())\n"
        "class Model:\n"
        "    def forward(self):\n"
        "        return None\n",
        encoding="utf-8",
    )

    first = load_authored_ir(f"{source}:Model")
    second = load_authored_ir(f"{source}:Model")

    assert type(first.target) is not type(second.target)
    assert registered_targets()["tests.cli.reload_target"] is type(first.target)


def test_named_provider_registers_only_decorated_targets_and_replays_them(
    tmp_path, monkeypatch, capsys
) -> None:
    provider_name = "decorated_target_provider"
    (tmp_path / f"{provider_name}.py").write_text(
        "from dataclasses import dataclass\n"
        "from tilefoundry.target import Target, register_target\n"
        "@dataclass(frozen=True)\n"
        "class _VendorBase(Target):\n"
        "    topology_levels = ('core',)\n"
        "@register_target\n"
        "@dataclass(frozen=True)\n"
        "class VendorOne(_VendorBase):\n"
        "    name = 'tests.cli.vendor_one'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = tmp_path / "registry.toml"
    prefix = ["--registry", str(registry), "target"]

    assert cli.main([*prefix, "add", provider_name]) == 0
    assert "VendorOne()" in capsys.readouterr().out
    assert "tests.cli.vendor_one" in registered_targets()
    assert all(
        target_type.__name__ != "_VendorBase"
        for target_type in registered_targets().values()
    )

    assert cli.main([*prefix, "remove", provider_name]) == 0
    assert "tests.cli.vendor_one" not in registered_targets()
    capsys.readouterr()

    assert cli.main([*prefix, "add", provider_name]) == 0
    assert "VendorOne()" in capsys.readouterr().out
    assert "tests.cli.vendor_one" in registered_targets()


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
        _SMOKE_TARGET / "vendor_npu" / "__init__.py",
        tmp_path / "vendor_npu.py",
    )

    added_architecture = _run_cli(
        registry, tmp_path, "target", "add", "--document", str(architecture)
    )
    assert added_architecture.returncode == 0, added_architecture.stderr
    assert "vendor.sm70" in added_architecture.stdout
    added_device = _run_cli(
        registry, tmp_path, "target", "add", "--document", str(device)
    )
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
    duplicate_module_name = _run_cli(
        registry, tmp_path, "target", "add", str(alternate_provider)
    )
    assert duplicate_module_name.returncode == 1
    assert "module name 'vendor_npu' is already added from" in duplicate_module_name.stderr
    assert str(provider) in duplicate_module_name.stderr
    assert str(alternate_provider) not in registry.read_text(encoding="utf-8")

    shown = _run_cli(registry, tmp_path, "target", "show", "vendor.v100_sxm2_32gb")
    assert shown.returncode == 0, shown.stderr
    assert "architecture: vendor.sm70" in shown.stdout
    assert "device: vendor.v100_sxm2_32gb" in shown.stdout
    assert shown.stdout.count("  digest: ") == 2

    npu_model = _write_registered_model(
        tmp_path / "npu_model.py",
        target="registered_targets()['vendor.npu']()",
        topology="core",
    )
    analyzed_npu = _run_cli(
        registry,
        tmp_path,
        "analyze",
        f"{npu_model}:model",
        "--compute-cost",
        "--memory",
        "--roofline",
        "--json",
    )
    assert analyzed_npu.returncode == 0, analyzed_npu.stderr
    assert json.loads(analyzed_npu.stdout)["target"] == "vendor.npu"
    unplaced_npu = _run_cli(
        registry,
        tmp_path,
        "analyze",
        f"{npu_model}:model",
        "--performance",
    )
    assert unplaced_npu.returncode == 1
    assert "has no core execution domain" in unplaced_npu.stderr
    scheduled_npu = _run_cli(
        registry,
        tmp_path,
        "schedule",
        f"{npu_model}:model",
        "--topology",
        "core",
        "--json",
    )
    assert scheduled_npu.returncode == 0, scheduled_npu.stderr
    assert json.loads(scheduled_npu.stdout) == {"topology": "core", "extent": 1}

    cuda_model = _write_registered_model(
        tmp_path / "cuda_model.py",
        target='CudaTarget("vendor.v100_sxm2_32gb")',
        topology="cta",
    )
    analyzed_cuda = _run_cli(
        registry,
        tmp_path,
        "analyze",
        f"{cuda_model}:model",
        "--compute-cost",
        "--json",
    )
    assert analyzed_cuda.returncode == 0, analyzed_cuda.stderr
    assert json.loads(analyzed_cuda.stdout)["target"] == "vendor.v100_sxm2_32gb"

    removed_module = _run_cli(registry, tmp_path, "target", "remove", "vendor.npu")
    assert removed_module.returncode == 0, removed_module.stderr
    assert "identities: ['vendor.npu']" in removed_module.stdout
    removed_device = _run_cli(
        registry, tmp_path, "target", "remove", "vendor.v100_sxm2_32gb"
    )
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
    source_architecture = (_SMOKE_TARGET / "hw" / "vendor_sm70.toml").read_text(
        encoding="utf-8"
    )
    source_device = (
        _SMOKE_TARGET / "hw" / "vendor_v100_sxm2_32gb.toml"
    ).read_text(encoding="utf-8")

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
    first = _run_cli(
        registry, tmp_path, "target", "add", "--document", str(architecture)
    )
    assert first.returncode == 0, first.stderr
    duplicate = _run_cli(
        registry, tmp_path, "target", "add", "--document", str(architecture)
    )
    assert duplicate.returncode == 1
    assert "hardware document 'vendor.sm70' is already registered" in duplicate.stderr

    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    json_provider = shadow_dir / "json.py"
    json_provider.write_text(
        "from tilefoundry.target import Target, register_target\n"
        "@register_target\n"
        "class JsonTarget(Target):\n"
        "    name = 'vendor.json'\n",
        encoding="utf-8",
    )
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

    collision_provider = tmp_path / "collision.py"
    collision_provider.write_text(
        "from tilefoundry.target import Target, register_target\n"
        "@register_target\n"
        "class CollisionTarget(Target):\n"
        "    name = 'nvidia.h200_sxm'\n",
        encoding="utf-8",
    )
    collision = _run_cli(registry, tmp_path, "target", "add", str(collision_provider))
    assert collision.returncode == 1
    assert "target identity 'nvidia.h200_sxm' is already occupied by" in collision.stderr
    assert 'CudaTarget("nvidia.h200_sxm")' in collision.stderr
    stored = tomllib.loads(registry.read_text(encoding="utf-8"))
    assert "module" not in stored

    named_provider = tmp_path / "named_provider.py"
    named_provider.write_text(
        "from tilefoundry.target import Target, register_target\n"
        "@register_target\n"
        "class NamedTarget(Target):\n"
        "    name = 'vendor.named'\n",
        encoding="utf-8",
    )
    named_registry = tmp_path / "named-registry.toml"
    named = _run_cli(named_registry, tmp_path, "target", "add", "named_provider")
    assert named.returncode == 0, named.stderr
    assert "NamedTarget()" in named.stdout
    named_stored = tomllib.loads(named_registry.read_text(encoding="utf-8"))
    assert named_stored == {"module": [{"name": "named_provider"}]}
    named_list = _run_cli(named_registry, tmp_path, "target", "list")
    assert named_list.returncode == 0, named_list.stderr
    assert "identity: vendor.named   added" in named_list.stdout
    named_provider.write_text("raise RuntimeError('provider changed')\n", encoding="utf-8")
    named_removed = _run_cli(
        named_registry, tmp_path, "target", "remove", "named_provider"
    )
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
    added_bad = _run_cli(
        registry, tmp_path, "target", "add", "--document", str(bad_architecture)
    )
    assert added_bad.returncode == 0, added_bad.stderr
    added_good = _run_cli(
        registry, tmp_path, "target", "add", "--document", str(good_device)
    )
    assert added_good.returncode == 0, added_good.stderr
    bad_architecture.write_text(
        bad_architecture.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    listed = _run_cli(registry, tmp_path, "target", "list")
    assert listed.returncode == 0
    assert "document 'vendor.bad_sm70' content changed" in listed.stderr
    assert 'CudaTarget("vendor.v100_sxm2_32gb")' in listed.stdout
    good_model = _write_registered_model(
        tmp_path / "good_model.py",
        target='CudaTarget("vendor.v100_sxm2_32gb")',
        topology="cta",
    )
    analyzed = _run_cli(
        registry,
        tmp_path,
        "analyze",
        f"{good_model}:model",
        "--compute-cost",
        "--json",
    )
    assert analyzed.returncode == 0
    assert "document 'vendor.bad_sm70' content changed" in analyzed.stderr
    assert json.loads(analyzed.stdout)["target"] == "vendor.v100_sxm2_32gb"

    repaired = _run_cli(
        registry, tmp_path, "target", "add", "--document", str(bad_architecture)
    )
    assert repaired.returncode == 0, repaired.stderr
    repaired_list = _run_cli(registry, tmp_path, "target", "list")
    assert repaired_list.returncode == 0, repaired_list.stderr
    assert "vendor.bad_sm70" in repaired_list.stdout


def test_analyze_binds_an_extent_on_a_root_that_reaches_a_child(tmp_path, capsys) -> None:
    """Choosing a size for a composed root keeps its child call activation-only."""
    source = tmp_path / "composed.py"
    source.write_text(composed_leaf_source("n_cli"), encoding="utf-8")

    assert cli.main(["analyze", f"{source}:Composed.root", "--dim", "n_cli=4"]) == 0
    expanded = capsys.readouterr().out
    assert "def root(" in expanded
    assert "leaf_w: ConstTensor" in expanded
    assert "leaf(" not in expanded


def test_analyze_reports_the_inlined_mega_kernel_from_one_rendering(capsys) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "placed" / "moe_mega_kernel.py"
    selector = f"{source}:MoEMegaKernel"
    flags = ["--compute-cost", "--memory", "--roofline", "--performance"]

    assert cli.main(["analyze", selector, *flags, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert cli.main(["analyze", selector, *flags]) == 0
    first = capsys.readouterr().out
    assert cli.main(["analyze", selector, *flags]) == 0
    second = capsys.readouterr().out
    assert first == second

    assert cli.main(["analyze", selector, *flags, "--operands"]) == 0
    asked = capsys.readouterr().out
    assert "operands=" not in first
    assert "operands=0:r30720/w0,result:r0/w30720" in asked
    assert cli.main(["analyze", selector, *flags, "--operands", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == payload

    header, annotated = first.split("\n\n", 1)
    lines = annotated.splitlines()
    assert payload["requested"] == payload["executed"] == [
        "compute-cost",
        "memory",
        "roofline",
        "performance",
    ]
    assert set(payload["function_records"]) == {
        "compute-cost",
        "traffic",
        "memory",
        "roofline",
        "performance",
    }
    assert "routed_expert(" not in annotated
    assert "shared_expert(" not in annotated
    assert annotated.count("reshard(tokens") == 2
    assert "v0 = reshard(tokens" in annotated
    assert "v3 = reshard(tokens" in annotated
    assert "offset=0" in annotated
    assert "offset=120" in annotated

    summary = payload["function_records"]["performance"]
    cost = payload["function_records"]["compute-cost"]
    moved = payload["function_records"]["traffic"]
    peak = payload["function_records"]["memory"]["footprint"]
    bound = payload["function_records"]["roofline"]
    assert header.splitlines() == [
        "# analysis target=nvidia.h200_sxm module=MoEMegaKernel function=experts "
        f"topology={payload['topology']}",
        f"# selection requested={','.join(payload['requested'])} "
        f"executed={','.join(payload['executed'])}",
        "# compute-cost "
        f"flops=f32:{cost['flops']['f32']}@{cost['flops_per_unit']['f32']}",
        "# traffic "
        f"traffic=gmem:r{moved['whole']['gmem']['read']}"
        f"/w{moved['whole']['gmem']['write']}"
        f"@r{moved['per_unit']['gmem']['read']}"
        f"/w{moved['per_unit']['gmem']['write']}",
        f"# peak-footprint=gmem:{peak[0]['peak_bytes']}",
        f"# roofline ideal-ns={bound['ideal_ns']} bound-by={bound['bound_by']}",
        "# performance root=MoEMegaKernel::experts "
        f"predicted-ns={summary['timeline']['end_ns']} "
        f"waves={summary['waves']}",
    ]
    assert payload["totals"]["flops"] == cost["flops"]
    assert payload["totals"]["traffic"] == moved["whole"]

    hoisted = {
        line.split(" = ", 1)[0] for line in lines if " = Mesh((Topology(" in line
    }
    assert hoisted == {"cta", "cta_2", "cta_3", "cta_4"}
    annotated_types = [
        line.split("  # ", 1)[1].split("; ", 1)[0] for line in lines if "  # Tensor[" in line
    ]
    assert 'Tensor[(120, 64), "f32", (120 @ cta_2.tile, 64)]' in annotated_types
    assert 'Tensor[(120, 64), "f32", (12 @ cta_3.tile, 10, 64)]' in annotated_types

    rows = payload["calls"]
    assert len(rows) == 7
    assert all(
        set(row) - {"performance"} == {"value", "compute-cost", "traffic", "roofline"}
        for row in rows
    )
    timed = [index for index, row in enumerate(rows) if "performance" in row]
    assert timed == [1, 4, 6]
    starts = []
    for row in rows:
        value, line_text = row["value"].rsplit(":", 1)
        line = int(line_text)
        starts.append(line)
        assert lines[line - 1].lstrip().startswith(f"{value} = ")

    for index, (row, start) in enumerate(zip(rows, starts)):
        stop = starts[index + 1] - 1 if index + 1 < len(starts) else len(lines)
        statement = "\n".join(lines[start - 1 : stop])
        if "performance" in row:
            timeline = row["performance"]["timeline"]
            assert f"performance=[{timeline['start_ns']},{timeline['end_ns']})" in statement
        else:
            assert "; performance=" not in statement
        annotated_meshes = set(re.findall(r"@ (\w+)\.", statement.split("  # ", 1)[1]))
        assert annotated_meshes <= hoisted
        placed_meshes = set(re.findall(r"mesh=(\w+),", statement))
        if annotated_meshes and placed_meshes:
            assert annotated_meshes == placed_meshes
    assert len([line for line in lines if "; performance=" in line]) == len(timed)
    assert len({row["value"] for row in rows}) == len(rows)
    assert "units=" not in annotated
