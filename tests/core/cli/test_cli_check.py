"""`check` through the command line, on targets the corpus already declares.

The command is the workflow: nothing else compares an implementation against a
reference, so every behaviour here is reached the way an agent reaches it.
"""

from __future__ import annotations

import json

import pytest
import torch

from tests.core.fixtures.logical import callee_dim
from tests.core.fixtures.placed import gqa_decode
from tilefoundry import cli

DISPATCHING = f"{gqa_decode.__file__}:GqaOnline.gqa_online_attend"


def test_inputs_must_be_stated(capsys) -> None:
    """Activations have no default form."""
    assert (
        cli.main(
            [
                "check",
                DISPATCHING,
                "--weights",
                "random",
                "--out",
                "output",
                "--fn",
                "nan_inf",
            ]
        )
        == 1
    )
    assert "no inputs stated" in capsys.readouterr().err


def test_a_json_target_without_a_parent_is_a_usage_error(capsys) -> None:
    """A misspelt report path is refused at parse time, before any comparison runs."""
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "check",
                DISPATCHING,
                "--inputs",
                "random",
                "--out",
                "output",
                "--fn",
                "nan_inf",
                "--json",
                "/nonexistent-dir/report.json",
            ]
        )

    assert stopped.value.code == 2
    refused = capsys.readouterr()
    assert refused.out == ""
    assert refused.err.startswith("usage: tilefoundry check")
    assert "/nonexistent-dir/report.json: its parent directory does not exist" in refused.err


def test_without_a_reference_only_a_one_sided_predicate_is_admitted(capsys) -> None:
    """Running the evaluator alone measures the candidate, and says only that."""
    assert (
        cli.main(
            [
                "check",
                DISPATCHING,
                "--inputs",
                "random",
                "--weights",
                "random",
                "--dim",
                "ctx_len=64",
                "--out",
                "output",
                "--fn",
                "rel_l2",
                "--max",
                "1",
            ]
        )
        == 1
    )
    assert "with no reference to compare against" in capsys.readouterr().err

    assert (
        cli.main(
            [
                "check",
                DISPATCHING,
                "--inputs",
                "random",
                "--weights",
                "random",
                "--dim",
                "ctx_len=64",
                "--out",
                "output",
                "--fn",
                "nan_inf",
            ]
        )
        == 0
    )
    reported = capsys.readouterr().out
    assert "reference: none" in reported
    assert "nan 0 inf 0" in reported


@pytest.mark.parametrize(
    ("selector", "dim", "refusal"),
    (
        ("Staged.main", "BT=8", None),
        ("Rooted.main", "BT=8", None),
        ("Staged.main", "BTT=8", "has no dimension named ['BTT']; the dimensions it reaches are BT (stage)"),
        ("DeadScratch.main", "BT=8", "no function it reaches declares a dimension"),
    ),
)
def test_a_dim_binds_where_a_reached_function_declares_it(
    selector, dim, refusal, tmp_path, capsys
) -> None:
    """--dim binds a helper's own axis, here or behind a module boundary.

    A name nothing reachable declares is refused with the functions that do
    declare dims; a binding the block dropped declares nothing.
    """
    x = torch.arange(4, dtype=torch.float32)
    torch.save(x, tmp_path / "x.pt")
    torch.save(x, tmp_path / "expected.pt")
    code = cli.main(
        [
            "check",
            f"{callee_dim.__file__}:{selector}",
            "--inputs",
            f"files:{tmp_path / 'x.pt'}",
            "--expected",
            str(tmp_path / "expected.pt"),
            "--out",
            "output",
            "--fn",
            "allclose",
            "--atol",
            "1e-6",
            "--rtol",
            "1e-6",
            "--dim",
            dim,
        ]
    )
    captured = capsys.readouterr()
    if refusal is None:
        assert code == 0 and "PASS" in captured.out
    else:
        assert code != 0 and refusal in captured.err


def test_a_dimension_left_as_a_range_is_refused(capsys) -> None:
    """An unstated range remains the caller's decision."""
    assert (
        cli.main(
            [
                "check",
                DISPATCHING,
                "--inputs",
                "random",
                "--weights",
                "random",
                "--out",
                "output",
                "--fn",
                "nan_inf",
            ]
        )
        == 1
    )
    refused = capsys.readouterr().err
    assert "still states DimVar(name='ctx_len'" in refused
    assert "bind it with --dim" in refused


def test_several_extents_check_the_dispatch_and_name_the_implementation(capsys, tmp_path) -> None:
    """Four lengths across the boundary reach both implementations, each named.

    The label is what a person reads and the canonical signature is what anything
    deciding reads, so both are reported and the text carries the label too.
    """
    assert (
        cli.main(
            [
                "check",
                DISPATCHING,
                "--inputs",
                "random",
                "--weights",
                "random",
                "--dim",
                "ctx_len=0,64,4096,32768",
                "--out",
                "output",
                "--fn",
                "nan_inf",
                "--json",
                str(tmp_path / "dispatch.json"),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    runs = json.loads((tmp_path / "dispatch.json").read_text(encoding="utf-8"))["runs"]

    assert [run["dims"]["ctx_len"] for run in runs] == [0, 64, 4096, 32768]
    assert [run["variant"]["display_name"] for run in runs] == [
        "head_on_cta",
        "head_on_cta",
        "head_on_cta",
        "ctx_split_kv",
    ]
    assert [run["variant"]["signature"] for run in runs] == [
        "ctx_len$0_4096",
        "ctx_len$0_4096",
        "ctx_len$0_4096",
        "ctx_len$4097_262144",
    ]

    assert (
        cli.main(
            [
                "check",
                DISPATCHING,
                "--inputs",
                "random",
                "--weights",
                "random",
                "--dim",
                "ctx_len=4096",
                "--out",
                "output",
                "--fn",
                "nan_inf",
            ]
        )
        == 0
    )
    assert (
        "variant:   head_on_cta  ctx_len$0_4096  (ctx_len in [0, 4096])"
        in capsys.readouterr().out
    )


def test_an_extent_outside_the_envelope_is_a_dispatch_hole_not_a_pass(capsys) -> None:
    """One past the envelope: no implementation claims it, and the answer says so.

    Naming the ranges that are covered is the point -- a hole is only actionable
    if the reader can see where the coverage stops.
    """
    assert (
        cli.main(
            [
                "check",
                DISPATCHING,
                "--inputs",
                "random",
                "--weights",
                "random",
                "--dim",
                "ctx_len=262145",
                "--out",
                "output",
                "--fn",
                "nan_inf",
            ]
        )
        == 1
    )
    refused = capsys.readouterr().err

    assert "declares no variant covering ctx_len=262145" in refused
    assert "4097, 262144]" in refused
