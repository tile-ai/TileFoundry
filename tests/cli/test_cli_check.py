"""`check` through the command line, on targets the corpus already declares.

The command is the workflow: nothing else compares an implementation against a
reference, so every behaviour here is reached the way an agent reaches it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy
import pytest
import torch
from safetensors.torch import save_file

from tests.fixtures import gqa_online
from tests.models.corpus import MODELS_ROOT
from tilefoundry import cli
from tilefoundry.cli.source import load_namespace, select_ir
from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.runtime import DictResource

#: Two outputs of different kinds from one call: routing weights and i64 indices.
#: A router that picked a different eight would be a different model even if every
#: number matched, so the indices are compared exactly. A child Module of the MoE
#: block, so this is also the real nested selector path.
ROUTING = f"{MODELS_ROOT / 'qwen3_5_35b_a3b' / 'model.py'}:Qwen3_5MoE.router.routing"

DISPATCHING = f"{gqa_online.__file__}:GqaOnline.gqa_online_attend"

@pytest.fixture(scope="module")
def routing(tmp_path_factory) -> dict[str, Path]:
    """One evaluator run of the MoE block's `router` child, as what a check reads.

    The inputs, a checkpoint holding the one weight that child declares, its two
    outputs, the same indices with one of them changed, and a zero reference.
    """
    where = tmp_path_factory.mktemp("routing")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    namespace, _ = load_namespace(ROUTING)
    parent = select_ir(namespace, "Qwen3_5MoE")
    leaf = next(child for child in parent.modules if child.name == "router")
    declared = leaf.lookup("routing")

    generator = torch.Generator(device=device).manual_seed(11)
    drawn = [
        torch.randn(tuple(param.type.shape), generator=generator, device=device).to(
            to_torch_dtype(param.type.dtype)
        )
        for param in declared.params
    ]
    tokens, w_router = drawn
    weights, indices = leaf.load(DictResource({"w_router": w_router})).routing(tokens)

    torch.save(tokens.cpu(), where / "tokens.pt")
    torch.save(weights.cpu(), where / "weights.pt")
    torch.save(torch.zeros_like(weights).cpu(), where / "zeros.pt")
    numpy.save(where / "indices.npy", indices.cpu().numpy())
    changed = indices.clone()
    changed[0, 0] = (changed[0, 0] + 1) % w_router.shape[-1]
    numpy.save(where / "one_off.npy", changed.cpu().numpy())
    # Exactly the leaf's own tensor, under the path the selector walks to it.
    save_file({"router.w_router": w_router.cpu()}, str(where / "model.safetensors"))
    return {
        "dir": where,
        "tokens": where / "tokens.pt",
        "weights": where / "weights.pt",
        "zeros": where / "zeros.pt",
        "indices": where / "indices.npy",
        "one_off": where / "one_off.npy",
    }


def _routing_argv(routing: dict[str, Path], indices: str, *comparison: str) -> list[str]:
    return [
        "check", ROUTING,
        "--input", str(routing["tokens"]),
        "--ckpt", str(routing["dir"]),
        "--expected", str(routing["weights"]),
        "--expected", str(routing[indices]),
        *comparison,
    ]


def test_each_output_is_judged_by_a_predicate_its_dtype_admits(routing, capsys) -> None:
    """One call, two kinds of output, each with its own comparison and verdict."""
    assert cli.main(_routing_argv(
        routing, "indices",
        "--out", "output[0]", "--fn", "allclose", "--atol", "1e-3", "--rtol", "4e-3",
        "--out", "output[1]", "--fn", "equal",
    )) == 0
    reported = capsys.readouterr().out

    assert "output[0]   bf16[1,8]" in reported and "output[1]   i64[1,8]" in reported
    assert "allclose(atol=0.001 rtol=0.004)" in reported
    assert "equal" in reported and "elements 8" in reported
    assert reported.rstrip().endswith("PASS") or "\nPASS" in reported


def test_one_wrong_index_fails_and_the_command_says_so(routing, capsys) -> None:
    """A single changed index is a total failure that no aggregate would see."""
    assert cli.main(_routing_argv(
        routing, "one_off",
        "--out", "output[0]", "--fn", "allclose", "--atol", "1e-3", "--rtol", "4e-3",
        "--out", "output[1]", "--fn", "equal",
    )) == 1
    reported = capsys.readouterr().out

    assert "mismatched 1" in reported
    assert "FAIL" in reported


def test_a_zero_reference_is_reported_rather_than_divided_by(routing, capsys) -> None:
    """`ref_norm` 0 and an absolute distance, not a number scaled by a clamp."""
    assert cli.main([
        "check", ROUTING,
        "--input", str(routing["tokens"]),
        "--ckpt", str(routing["dir"]),
        "--expected", str(routing["zeros"]),
        "--expected", str(routing["indices"]),
        "--out", "output[0]", "--fn", "rel_l2", "--max", "1e-3", "--fn", "cosine", "--min", "0.999",
        "--out", "output[1]", "--fn", "equal",
    ]) == 1
    reported = capsys.readouterr().out

    assert "ref_norm 0" in reported
    assert "absolute_l2" in reported
    assert "the reference norm is zero" in reported
    assert "one side is entirely zero" in reported
    # The old behaviour divided by a clamp and reported a number of that scale.
    assert "e+12" not in reported


def test_the_json_report_carries_the_same_facts_as_the_text(routing, capsys) -> None:
    """Including `ref_norm` and each predicate's own bound and value."""
    assert cli.main(_routing_argv(
        routing, "indices",
        "--out", "output[0]", "--fn", "rel_l2", "--max", "1e-3",
        "--out", "output[1]", "--fn", "equal",
        "--json",
    )) == 0
    reported = json.loads(capsys.readouterr().out)

    assert reported["passed"] is True
    outputs = reported["runs"][0]["outputs"]
    assert [output["path"] for output in outputs] == ["output[0]", "output[1]"]
    assert outputs[0]["ref_norm"] > 0
    assert outputs[0]["fns"][0] == {
        "fn": "rel_l2", "max": 1e-3, "rel_l2": outputs[0]["fns"][0]["rel_l2"], "passed": True
    }
    assert "verification" not in reported


@pytest.mark.parametrize(
    "comparison, refused",
    [
        pytest.param(
            ["--out", "output[0]", "--fn", "rel_l2", "--max", "1e-3",
             "--out", "output[1]", "--fn", "cosine", "--min", "0.99"],
            "output[1] is i64; cosine is not meaningful on a discrete output",
            id="an-aggregate-over-indices",
        ),
        pytest.param([], "no comparison requested", id="no-predicate-at-all"),
        pytest.param(
            ["--out", "output[0]", "--fn", "allclose", "--atol", "1e-3",
             "--out", "output[1]", "--fn", "equal"],
            "--fn allclose needs ['--rtol']",
            id="a-bound-left-out",
        ),
        pytest.param(
            ["--out", "output[0]", "--fn", "rel_l2", "--max", "1e-3"],
            "no comparison requested for output 'output[1]'",
            id="an-output-left-unjudged",
        ),
    ],
)
def test_check_refuses_what_it_cannot_answer(routing, capsys, comparison, refused) -> None:
    """Each refusal names what is missing; none of them has a default to fall back on."""
    assert cli.main(_routing_argv(routing, "indices", *comparison)) == 1

    assert refused in capsys.readouterr().err


def test_inputs_must_be_stated_and_weights_must_come_from_somewhere(routing, capsys) -> None:
    """Neither the inputs nor the weights have a default form."""
    assert cli.main([
        "check", ROUTING, "--out", "output[0]", "--fn", "nan_inf",
    ]) == 1
    assert "needs weights ['w_router']" in capsys.readouterr().err

    assert cli.main([
        "check", DISPATCHING, "--out", "output", "--fn", "nan_inf",
    ]) == 1
    assert "no inputs stated" in capsys.readouterr().err


def test_without_a_reference_only_a_one_sided_predicate_is_admitted(capsys) -> None:
    """Running the evaluator alone measures the candidate, and says only that."""
    assert cli.main([
        "check", DISPATCHING, "--inputs", "random", "--out", "output", "--fn", "rel_l2", "--max", "1",
    ]) == 1
    assert "with no reference to compare against" in capsys.readouterr().err

    assert cli.main([
        "check", DISPATCHING, "--inputs", "random", "--out", "output", "--fn", "nan_inf",
    ]) == 0
    reported = capsys.readouterr().out
    assert "reference: none" in reported
    assert "nan 0 inf 0" in reported


def test_a_dimension_left_as_a_range_is_reported_with_what_it_was_pinned_to(capsys) -> None:
    """The pin is a decision this run made, so it is said out loud, in both forms."""
    assert cli.main([
        "check", DISPATCHING, "--inputs", "random", "--out", "output", "--fn", "nan_inf",
    ]) == 0
    reported = capsys.readouterr().out
    assert "ctx_len is a range [0, 262144) that nothing bound; this run pinned it to 0" in reported
    # Both ways out of a pin: bind the size, or declare a variant that covers it.
    assert "--dim ctx_len=" in reported
    assert "`tilefoundry spec parser 1.1`" in reported

    assert cli.main([
        "check", DISPATCHING, "--inputs", "random", "--out", "output", "--fn", "nan_inf", "--json",
    ]) == 0
    pinned = json.loads(capsys.readouterr().out)["runs"][0]["pinned"]
    assert {entry["dim"]: entry["pinned"] for entry in pinned} == {"ctx_len": 0}


def test_several_extents_check_the_dispatch_and_name_the_implementation(capsys) -> None:
    """Four lengths across the boundary reach both implementations, each named.

    The label is what a person reads and the canonical signature is what anything
    deciding reads, so both are reported and the text carries the label too.
    """
    assert cli.main([
        "check", DISPATCHING, "--inputs", "random", "--dim", "ctx_len=0,64,4096,32768",
        "--out", "output", "--fn", "nan_inf", "--json",
    ]) == 0
    runs = json.loads(capsys.readouterr().out)["runs"]

    assert [run["dims"]["ctx_len"] for run in runs] == [0, 64, 4096, 32768]
    assert [run["variant"]["display_name"] for run in runs] == [
        "head_on_cta", "head_on_cta", "ctx_split_kv", "ctx_split_kv"
    ]
    assert [run["variant"]["signature"] for run in runs] == [
        "ctx_len$0_4096", "ctx_len$0_4096", "ctx_len$4096_262144", "ctx_len$4096_262144"
    ]

    assert cli.main([
        "check", DISPATCHING, "--inputs", "random", "--dim", "ctx_len=4096", "--out", "output", "--fn", "nan_inf",
    ]) == 0
    assert "variant:   ctx_split_kv  ctx_len$4096_262144" in capsys.readouterr().out


def test_an_extent_outside_the_envelope_is_a_dispatch_hole_not_a_pass(capsys) -> None:
    """One past the envelope: no implementation claims it, and the answer says so.

    Naming the ranges that are covered is the point -- a hole is only actionable
    if the reader can see where the coverage stops.
    """
    assert cli.main([
        "check", DISPATCHING, "--inputs", "random", "--dim", "ctx_len=262144", "--out", "output", "--fn", "nan_inf",
    ]) == 1
    refused = capsys.readouterr().err

    assert "declares no variant covering ctx_len=262144" in refused
    assert "4096, 262144)" in refused


def test_a_passing_check_carries_no_verification_ranking(routing, capsys) -> None:
    """A PASS says the candidate matches this Module, and claims nothing further.

    It used to append a ranking read out of the shipped catalog. That was a claim
    about tests, carried in data the command cannot check, and it outlived the test
    it named -- so it is gone, and the warnings that are about *this run* stay.
    """
    assert cli.main(_routing_argv(
        routing, "indices", "--out", "output[0]", "--fn", "rel_l2", "--max", "1e-3",
        "--out", "output[1]", "--fn", "equal",
    )) == 0
    reported = capsys.readouterr().out

    assert "PASS" in reported
    for absent in ("verification on record", "L1", "L2", "L3", "usable as an oracle"):
        assert absent not in reported
    assert "--inputs random makes each activation independently" not in reported
    assert "FAIL says the candidate and reference differ" not in reported

def test_a_random_input_fail_against_a_reference_states_its_limits(routing, capsys) -> None:
    """A failed random-input comparison qualifies both limits through the CLI."""
    argv = [
        "check", ROUTING,
        "--inputs", "random",
        "--expected", str(routing["zeros"]),
        "--expected", str(routing["indices"]),
        "--out", "output[0]", "--fn", "allclose", "--atol", "1e-3", "--rtol", "4e-3",
        "--out", "output[1]", "--fn", "equal",
    ]

    assert cli.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    reported = captured.out
    assert "--inputs random makes each activation independently" in reported
    assert "Rerun with --inputs real" in reported
    assert "FAIL says the candidate and reference differ" in reported
    assert "Establishing accuracy needs an independent high-precision" in reported
    assert "reference, which check does not run" in reported

    assert cli.main([*argv, "--json"]) == 1
    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert warnings == [
        "--inputs random makes each activation independently. A target that relies on "
        "semantic relationships between activations can differ at ulp scale without either "
        "implementation being wrong. Rerun with --inputs real to decide the comparison.",
        "FAIL says the candidate and reference differ, not which side is closer to truth. "
        "The reference may carry its own rounding; check compares only against it. "
        "Establishing accuracy needs an independent high-precision reference, which check "
        "does not run.",
    ]


def test_a_nested_child_reads_only_its_own_part_of_the_checkpoint(routing, capsys) -> None:
    """Reaching `router.routing` reads `router.w_router`: the checkpoint holds
    that one tensor and none of the eight the block around it declares."""
    assert cli.main(_routing_argv(
        routing, "indices",
        "--out", "output[0]", "--fn", "nan_inf", "--out", "output[1]", "--fn", "equal",
    )) == 0
    assert "weights the checkpoint" in capsys.readouterr().out
