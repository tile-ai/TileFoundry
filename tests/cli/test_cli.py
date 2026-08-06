from __future__ import annotations

import pytest

from tilefoundry import cli
from tilefoundry.cli.source import load_authored_ir, one_extent_per_dim
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
    [["ctx_len"], ["ctx_len="], ["=8"], ["ctx_len=eight"], ["ctx_len=1.5"]],
)
def test_parse_dims_rejects_an_argument_that_states_no_extent(stated) -> None:
    with pytest.raises(ValueError):
        cli.parse_dims(stated)


def test_parse_dims_rejects_one_dimension_stated_twice() -> None:
    """Repeating the flag states another dimension, not another value for one
    already stated.

    Taking the last would answer an ambiguous request by picking silently, and
    the caller would be told nothing -- which is the failure worth catching,
    because both numbers came from them.
    """
    with pytest.raises(ValueError, match="ctx_len was given twice"):
        cli.parse_dims(["ctx_len=8", "ctx_len=512"])
    # Repeating the same extent is still two statements of one dimension.
    with pytest.raises(ValueError, match="ctx_len was given twice"):
        cli.parse_dims(["ctx_len=8", "ctx_len=8"])


def test_schedule_rejects_several_extents_per_dimension(capsys) -> None:
    argv = ["schedule", "missing.py", "--topology", "cta", "--dim", "ctx_len=0,1"]
    assert cli.main(argv) == 1
    refused = capsys.readouterr().err
    assert "ctx_len takes one EXTENT at a time" in refused
    assert "asking several EXTENTs together is for check" in refused


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
