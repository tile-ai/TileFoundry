"""CLI target listing reconstructs each advertised target value."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry import cli
from tilefoundry.cli.source import (
    _one_extent_per_dim,
    load_authored_ir,
)
from tilefoundry.target import registered_targets

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_parse_dims_reads_extents() -> None:
    """Nothing stated is not the same as nothing chosen."""
    assert cli.parse_dims(None) is None
    assert cli.parse_dims([]) == {}
    assert cli.parse_dims(["ctx_len=1024"]) == {"ctx_len": (1024,)}
    assert cli.parse_dims(["ctx_len=8", "seq_len=1"]) == {"ctx_len": (8,), "seq_len": (1,)}
    assert cli.parse_dims(["ctx_len=0,1,37"]) == {"ctx_len": (0, 1, 37)}

    assert _one_extent_per_dim(cli.parse_dims(None)) is None
    assert _one_extent_per_dim(cli.parse_dims([])) == {}
    assert _one_extent_per_dim(cli.parse_dims(["ctx_len=1024"])) == {"ctx_len": 1024}
    with pytest.raises(ValueError, match="asking several EXTENTs together is for check"):
        _one_extent_per_dim(cli.parse_dims(["ctx_len=0,1,37"]))


@pytest.mark.parametrize(
    "stated",
    [["=8"], ["ctx_len=1.5"]],
)
def test_parse_dims_rejects_missing_extent(stated) -> None:
    with pytest.raises(ValueError):
        cli.parse_dims(stated)


def test_parse_dims_rejects_repeated_dim() -> None:
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


def test_analyze_help_explains_traffic_assumptions(capsys) -> None:
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


def test_cli_prints_help_and_version(capsys) -> None:
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


def test_spec_refuses_unknown_topic(capsys) -> None:
    assert cli.main(["spec", "nope"]) == 1
    refused = capsys.readouterr()
    assert "no spec named 'nope'; there are " in refused.err
    assert ".md" not in refused.err
    assert "TileFoundry" not in refused.err
    assert "dsl" in refused.err


def test_analyze_json_requires_analysis_flag(capsys, tmp_path) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["analyze", "missing.py", str(tmp_path / "report.json"), "--json"])

    assert stopped.value.code == 2
    refused = capsys.readouterr()
    assert refused.out == ""
    assert refused.err.startswith("usage: tilefoundry analyze")
    assert "tilefoundry analyze: error: --json requires at least one analysis flag:" in refused.err
    assert "source file not found" not in refused.err


def test_reload_keeps_one_target_registration() -> None:
    source = FIXTURES / "cli/reload_target/model.py"

    first = load_authored_ir(f"{source}:Model")
    second = load_authored_ir(f"{source}:Model")

    assert type(first.target) is not type(second.target)
    assert registered_targets()["tests.cli.reload_target"] is type(first.target)


def test_provider_registers_decorated_targets(
    tmp_path, monkeypatch, capsys
) -> None:
    provider_name = "decorated_target_provider"
    monkeypatch.syspath_prepend(str(FIXTURES / "target_plugins"))
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
