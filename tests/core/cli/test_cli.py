"""Target-neutral tests and helpers."""

from __future__ import annotations

import pytest

from tilefoundry import cli
from tilefoundry.cli.source import (
    load_authored_ir,
    one_extent_per_dim,
)
from tilefoundry.target import registered_targets


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


def test_analyze_help_explains_topology_effects_and_assumptions(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["analyze", "--help"])

    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for family in ("compute-cost", "memory", "roofline", "performance"):
        assert family in help_text
    assert "logical" in help_text and "per-unit share" in help_text
    assert "traffic" in help_text
    assert "logical traffic omits loop replication" in help_text
    assert "total traffic counts every executed occurrence" in help_text
    assert "is an observation, not a bound" in help_text


def test_cli_overviews_and_version_are_parser_surfaces(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])
    assert stopped.value.code == 0
    help_page = capsys.readouterr()

    assert cli.main([]) == 0
    bare = capsys.readouterr()
    assert bare.out == help_page.out
    assert bare.err == help_page.err == ""
    assert "TileFoundry — A tile-based, agentic platform" in help_page.out

    with pytest.raises(SystemExit) as stopped:
        cli.main(["nope"])
    assert stopped.value.code == 2
    refused = capsys.readouterr()
    assert refused.out == ""
    assert refused.err.startswith("usage: tilefoundry")
    assert "argument <command>: invalid choice: 'nope'" in refused.err
    assert "TileFoundry — " not in refused.err

    with pytest.raises(SystemExit) as stopped:
        cli.main(["-h"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out == help_page.out

    with pytest.raises(SystemExit) as stopped:
        cli.main(["target", "--help"])
    assert stopped.value.code == 0
    target_help = capsys.readouterr()
    assert cli.main(["target"]) == 0
    target_bare = capsys.readouterr()
    assert target_bare.out == target_help.out
    assert target_bare.err == target_help.err == ""

    with pytest.raises(SystemExit) as stopped:
        cli.main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.startswith("tilefoundry ")


def test_spec_names_unknown_topic_and_available_aliases(capsys) -> None:
    assert cli.main(["spec", "nope"]) == 1
    refused = capsys.readouterr()
    assert "no spec named 'nope'; there are " in refused.err
    assert ".md" not in refused.err
    assert "TileFoundry" not in refused.err
    assert "dsl" in refused.err


def test_analyze_json_without_a_requested_root_is_a_usage_error(capsys, tmp_path) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["analyze", "missing.py", str(tmp_path / "report.json"), "--json"])

    assert stopped.value.code == 2
    refused = capsys.readouterr()
    assert refused.out == ""
    assert refused.err.startswith("usage: tilefoundry analyze")
    assert "tilefoundry analyze: error: --json requires at least one analysis flag:" in refused.err
    assert "source file not found" not in refused.err


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
        (
            'from tilefoundry.dsl import *\n'
            'from provider import ReloadTarget\n'
            '@module(target=ReloadTarget())\n'
            'class Model:\n'
            '    def forward(self):\n'
            '        return None\n'
        ),
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
        target_type.__name__ != "_VendorBase" for target_type in registered_targets().values()
    )

    assert cli.main([*prefix, "remove", provider_name]) == 0
    assert "tests.cli.vendor_one" not in registered_targets()
    capsys.readouterr()

    assert cli.main([*prefix, "add", provider_name]) == 0
    assert "VendorOne()" in capsys.readouterr().out
    assert "tests.cli.vendor_one" in registered_targets()
