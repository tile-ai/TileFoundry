from __future__ import annotations

import pytest

from tilefoundry import cli


def test_parse_dims_reads_one_extent_per_dimension() -> None:
    """Nothing stated is not the same as nothing chosen."""
    assert cli.parse_dims(None) is None
    assert cli.parse_dims([]) is None
    assert cli.parse_dims(["ctx_len=1024"]) == {"ctx_len": 1024}
    assert cli.parse_dims(["ctx_len=8", "seq_len=1"]) == {"ctx_len": 8, "seq_len": 1}


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
