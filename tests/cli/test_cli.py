from __future__ import annotations

import sys

import pytest

from tilefoundry import cli
from tilefoundry.cli.source import load_namespace, one_extent_per_dim


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


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "missing.py", "--dim", "ctx_len=0,1"],
        ["schedule", "missing.py", "--topology", "cta", "--dim", "ctx_len=0,1"],
    ],
)
def test_analyze_and_schedule_reject_several_extents_per_dimension(argv, capsys) -> None:
    assert cli.main(argv) == 1
    refused = capsys.readouterr().err
    assert "ctx_len takes one EXTENT at a time" in refused
    assert "asking several EXTENTs together is for check" in refused


def test_an_authored_file_may_pair_deferred_annotations_with_a_dataclass(tmp_path) -> None:
    """The loaded module is registered while it runs, because `dataclasses` needs it.

    `from __future__ import annotations` makes every annotation a string, and
    `dataclasses` resolves one by looking the defining module up in `sys.modules`.
    A module loaded from a path and left unregistered is not there to be found, so
    the decorator raised `AttributeError: 'NoneType' object has no attribute
    '__dict__'` and the file could not be loaded at all -- which is how a shipped
    model that pairs the two became unanalysable by every command at once.

    Registered only for the load: the name is gone again afterwards, so one file
    cannot shadow a real module for whatever runs next.
    """
    source = tmp_path / "authored.py"
    source.write_text(
        "from __future__ import annotations\n"
        "\n"
        "from dataclasses import dataclass\n"
        "\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Shape:\n"
        "    width: int = 16\n"
        "\n"
        "\n"
        "STATED = Shape()\n",
        encoding="utf-8",
    )

    namespace, _selector = load_namespace(str(source))

    assert namespace["STATED"].width == 16
    assert "authored" not in sys.modules
