"""`check` through the command line, on targets the corpus already declares.

The command is the workflow: nothing else compares an implementation against a
reference, so every behaviour here is reached the way an agent reaches it.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy
import pytest
import torch
from safetensors.torch import save_file

from tilefoundry import cli
from tilefoundry.cli.check import SEED
from tilefoundry.cli.source import load_namespace, select_ir
from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.runtime import DictResource

#: Two outputs of different kinds from one call: routing weights and i64 indices.
#: A router that picked a different eight would be a different model even if every
#: number matched, so the indices are compared exactly. A child Module of the MoE
#: block, so this is also the real nested selector path.
ROUTING = "tests/models/qwen3_5_35b_a3b/model.py:Qwen3_5MoE.router.routing"

DISPATCHING = "tests/fixtures/gqa_online.py:GqaOnline.gqa_online_attend"

_TWIN_SOURCE = """
from tilefoundry import module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, func, tf
from tilefoundry.runtime import RuntimeModule, runtime_func, runtime_module
from tilefoundry.target import CudaTarget


@module(entry="main", target=CudaTarget())
class Model:
    topologies = (Topology("cta", 168),)

    @func
    def main(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(Topology("cta", 168), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            squared = tf.square(x_local)
            return tf.reshard(squared, (168 @ cta.block,), "gmem")

    @func
    def zeroed(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(Topology("cta", 168), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            nothing = tf.sub(x_local, x_local)
            return tf.reshard(nothing, (168 @ cta.block,), "gmem")


@runtime_module(Model)
class Twin:
    @runtime_func
    def main(self, x):
        return x * x

    @runtime_func
    def zeroed(self, x):
        return x - x


@runtime_module(Model)
class Drifted:
    @runtime_func
    def main(self, x):
        return x * x + 0.5

    @runtime_func
    def zeroed(self, x):
        return x - x


@module(entry="scaled")
class Weighted:
    topologies = (Topology("cta", 168),)

    @func
    def scaled(
        x: Tensor[(168,), "f32"], w: ConstTensor[(168,), "f32"]
    ) -> Tensor[(168,), "f32"]:
        with Mesh(Topology("cta", 168), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            w_local = tf.reshard(w, (168 @ cta.block,), "rmem")
            weighted = tf.mul(x_local, w_local)
            return tf.reshard(weighted, (168 @ cta.block,), "gmem")


@runtime_module(Weighted)
class WeightedTwin:
    @runtime_func
    def scaled(self, x, w):
        return x * w


@module(entry="fused", target=CudaTarget())
class Fused:
    topologies = (Topology("cta", 168),)

    @func
    def fused(x: Tensor[(168,), "f32"]) -> Tensor[(168,), "f32"]:
        with Mesh(Topology("cta", 168), (168,), ("block",)) as cta:
            x_local = tf.reshard(x, (168 @ cta.block,), "rmem")
            squared = tf.square(x_local)
            shifted = tf.sub(squared, x_local)
            return tf.reshard(shifted, (168 @ cta.block,), "gmem")


@runtime_module(Fused)
class FusedTwin:
    @runtime_func
    def fused(self, x):
        return x * x - x


@module(target=CudaTarget())
class Nested:
    child = Weighted


@runtime_module(Nested)
class NestedTwin:
    child = WeightedTwin


class Handwritten(RuntimeModule):
    def __init__(self):
        super().__init__(name="handwritten")


class Mislabelled(RuntimeModule):
    module = "not a Module"

    def __init__(self):
        super().__init__(name="mislabelled")
"""


@pytest.fixture(scope="module")
def twin(tmp_path_factory) -> Path:
    """A file of somebody's own: a Module, its twin, and one that drifts."""
    path = tmp_path_factory.mktemp("authored") / "mine.py"
    path.write_text(textwrap.dedent(_TWIN_SOURCE), encoding="utf-8")
    return path


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
    assert reported["verification"] == {"model": "qwen3_5_35b_a3b", "level": "L1"}


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


def test_a_model_below_the_oracle_level_passes_with_a_warning(routing, capsys) -> None:
    """PASS against a Module is not agreement with what the Module describes."""
    assert cli.main(_routing_argv(
        routing, "indices", "--out", "output[0]", "--fn", "rel_l2", "--max", "1e-3",
        "--out", "output[1]", "--fn", "equal",
    )) == 0
    reported = capsys.readouterr().out

    assert "warning: qwen3_5_35b_a3b has no L3 verification on record" in reported
    assert "not\n           that the Module matches what it describes" in reported


def test_a_twin_is_compared_against_the_module_it_states(twin, capsys) -> None:
    """The target is any file, and naming the implementation reaches its reference."""
    assert cli.main([
        "check", f"{twin}:Twin.main", "--inputs", "random",
        "--out", "output", "--fn", "allclose", "--atol", "1e-6", "--rtol", "1e-6",
        "--fn", "cosine", "--min", "0.9999",
    ]) == 0
    reported = capsys.readouterr().out
    assert "reference: evaluator on Model.main" in reported
    assert "max_violation 0" in reported

    assert cli.main([
        "check", f"{twin}:Drifted.main", "--inputs", "random",
        "--out", "output", "--fn", "allclose", "--atol", "1e-6", "--rtol", "1e-6",
    ]) == 1
    assert "FAIL" in capsys.readouterr().out

    # The fused copy: what the file above spends two functions on, in one.
    assert cli.main([
        "check", f"{twin}:FusedTwin.fused", "--inputs", "random",
        "--out", "output", "--fn", "allclose", "--atol", "1e-6", "--rtol", "1e-6",
    ]) == 0
    assert "reference: evaluator on Fused.fused" in capsys.readouterr().out


def test_a_whole_module_is_checked_against_an_expected_output_file(twin, tmp_path, capsys) -> None:
    """An authored Module, activations from a file, and a file to match."""
    activation = torch.arange(168, dtype=torch.float32)
    torch.save(activation, tmp_path / "x.pt")
    torch.save(activation * activation, tmp_path / "expected.pt")

    assert cli.main([
        "check", f"{twin}:Model",
        "--input", str(tmp_path / "x.pt"),
        "--expected", str(tmp_path / "expected.pt"),
        "--out", "output", "--fn", "equal",
    ]) == 0
    reported = capsys.readouterr().out

    assert "expected.pt" in reported
    assert "elements 168" in reported


def test_the_inputs_are_exactly_one_form(twin, tmp_path, capsys) -> None:
    """Files and a draw are two answers to one question, so asking both is refused."""
    torch.save(torch.arange(168, dtype=torch.float32), tmp_path / "x.pt")

    for form in ("random", "real"):
        assert cli.main([
            "check", f"{twin}:Twin.main", "--input", str(tmp_path / "x.pt"),
            "--inputs", form, "--out", "output", "--fn", "nan_inf",
        ]) == 1
        assert "give exactly one form" in capsys.readouterr().err


def test_two_entirely_zero_sides_are_a_match_not_a_total_mismatch(twin, capsys) -> None:
    """Both sides zero: cosine is 1, which is what agreement between them means."""
    assert cli.main([
        "check", f"{twin}:Twin.zeroed", "--inputs", "random",
        "--out", "output", "--fn", "cosine", "--min", "0.999", "--fn", "rel_l2", "--max", "1e-6",
    ]) == 0
    reported = capsys.readouterr().out

    assert "cosine 1" in reported
    assert "both sides are entirely zero" in reported
    assert "ref_norm 0" in reported
    assert "PASS" in reported


def test_real_weights_come_from_the_checkpoint_and_activations_are_drawn(
    twin, tmp_path, capsys
) -> None:
    """The third input form: the weight is read, the activations are drawn.

    The report has to say which is which -- a run whose numbers came from a
    checkpoint and a run whose numbers were invented look the same otherwise.
    """
    save_file(
        {"w": torch.linspace(0.5, 2.0, 168)}, str(tmp_path / "model.safetensors")
    )

    assert cli.main([
        "check", f"{twin}:WeightedTwin.scaled", "--inputs", "real", "--ckpt", str(tmp_path),
        "--out", "output", "--fn", "allclose", "--atol", "1e-6", "--rtol", "1e-6",
    ]) == 0
    reported = capsys.readouterr().out

    assert "weights the checkpoint" in reported
    assert f"random, seed {SEED}" in reported
    assert "max_violation 0" in reported

    # And real weights without a checkpoint to read them from is refused.
    assert cli.main([
        "check", f"{twin}:WeightedTwin.scaled", "--inputs", "real",
        "--out", "output", "--fn", "allclose", "--atol", "1e-6", "--rtol", "1e-6",
    ]) == 1
    assert "needs --ckpt DIR" in capsys.readouterr().err


def test_a_nested_child_reads_only_its_own_part_of_the_checkpoint(routing, capsys) -> None:
    """Reaching `router.routing` reads `router.w_router`: the checkpoint holds
    that one tensor and none of the eight the block around it declares."""
    assert cli.main(_routing_argv(
        routing, "indices",
        "--out", "output[0]", "--fn", "nan_inf", "--out", "output[1]", "--fn", "equal",
    )) == 0
    assert "weights the checkpoint" in capsys.readouterr().out


def test_a_nested_twin_is_reached_through_the_child_it_is_declared_under(
    twin, capsys
) -> None:
    """Selector descent on a twin: the child segment moves the twin and the
    authored Module it is judged against, and the weight is read under the child's
    own name. Resolution and loading only."""
    assert cli.main([
        "check", f"{twin}:NestedTwin.child.scaled", "--inputs", "random",
        "--out", "output", "--fn", "allclose", "--atol", "1e-6", "--rtol", "1e-6",
    ]) == 0
    reported = capsys.readouterr().out

    assert "reference: evaluator on child.scaled" in reported
    assert "max_violation 0" in reported


def test_a_runtime_module_that_names_no_authored_module_is_refused(twin, capsys) -> None:
    """`check` reads the twin's accessor and takes what it says at its word.

    Both refusals are the CLI's, not the decorator's: `@runtime_module` rejects a
    written `module` outright, so only a hand-written subclass reaches here.
    """
    for target, refused in (
        ("Handwritten", "names no authored Module"),
        ("Mislabelled", "module must be Module or None, got str"),
    ):
        assert cli.main([
            "check", f"{twin}:{target}", "--inputs", "random",
            "--out", "output", "--fn", "nan_inf",
        ]) == 1
        assert refused in capsys.readouterr().err
