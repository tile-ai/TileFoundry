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
from safetensors.torch import load_file, save_file

from tests.core.cli._check import MESH_CHECK_ARGS
from tests.nvidia.cuda.models.corpus import MODELS_ROOT
from tilefoundry import cli
from tilefoundry.cli.source import load_authored_ir, load_namespace, select_ir
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.runtime import DictResource
from tilefoundry.runtime.resource import DrawnResource

ROUTING = f"{MODELS_ROOT / 'qwen3_5_35b_a3b' / 'model.py'}:Qwen3_5Router.routing"
LEAF_WEIGHTS = Path(__file__).parents[1] / "fixtures" / "placed" / "leaf_weights.py"


def cpu_gen() -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(0)


class RecordingResource:
    def __init__(self, inner):
        self.inner, self.asked = inner, []

    def load(self, name):
        self.asked.append(name)
        return self.inner.load(name)

    def load_group(self, name):
        return self.inner.load_group(name)

    def subtree(self, seg):
        return RecordingResource(self.inner.subtree(seg))


def test_load_reads_no_weights():
    Mod = load_authored_ir(f"{LEAF_WEIGHTS}:Mod")
    Small = load_authored_ir(f"{LEAF_WEIGHTS}:Small")
    SmallTwin = load_namespace(str(LEAF_WEIGHTS))[0]["SmallTwin"]
    for mod, mk in (
        (Mod, Mod.load),
        (Small, Small.load),
        (Small, SmallTwin().load),
    ):
        rec = RecordingResource(DrawnResource(mod, cpu_gen(), "cpu"))
        mk(rec)
        assert rec.asked == []


def test_evaluate_reads_own_consts():
    Small = load_authored_ir(f"{LEAF_WEIGHTS}:Small")
    D = load_namespace(str(LEAF_WEIGHTS))[0]["D"]
    rec = RecordingResource(DrawnResource(Small, cpu_gen(), "cpu"))
    evaluate(Small.load(rec).leaf, torch.zeros(1, D))
    assert rec.asked == []
    evaluate(Small.load(rec), torch.zeros(1, D))
    assert rec.asked == ["w"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_foreign_device_weight_named():
    Small = load_authored_ir(f"{LEAF_WEIGHTS}:Small")
    D = load_namespace(str(LEAF_WEIGHTS))[0]["D"]
    rec = RecordingResource(DrawnResource(Small, cpu_gen(), "cpu"))
    with pytest.raises(ValueError, match=r"'w'"):
        evaluate(
            Small.load(rec),
            torch.zeros(1, D, device="cuda"),
        )


def test_missing_weight_named():
    loaded = load_authored_ir(f"{LEAF_WEIGHTS}:Small").load(DictResource({}))
    with pytest.raises(KeyError, match=r"missing declared weight 'w'"):
        evaluate(loaded, torch.zeros(1, load_namespace(str(LEAF_WEIGHTS))[0]["D"]))


def test_twin_binds_weights_once():
    Small = load_authored_ir(f"{LEAF_WEIGHTS}:Small")
    leaf_weights = load_namespace(str(LEAF_WEIGHTS))[0]
    D = leaf_weights["D"]
    rec = RecordingResource(DrawnResource(Small, cpu_gen(), "cpu"))
    twin = leaf_weights["SmallTwin"]()
    twin.load(rec)
    twin.entry(torch.zeros(1, D))
    twin.entry(torch.zeros(1, D))
    assert rec.asked == ["w"]


def test_input_weight_axes_independent(tmp_path):
    from tilefoundry.cli import build_parser  # noqa: PLC0415 -- existing parser shape changes in M4

    source = f"{LEAF_WEIGHTS}:Small.entry"
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    for path in (a, b):
        torch.save(torch.zeros(1, load_namespace(str(LEAF_WEIGHTS))[0]["D"]), path)
    for inputs in ("random", f"files:{a},{b}"):
        for weights in ("random", f"ckpt:{tmp_path}"):
            build_parser().parse_args(
                [
                    "check",
                    source,
                    "--out",
                    "output",
                    "--fn",
                    "nan_inf",
                    "--inputs",
                    inputs,
                    "--weights",
                    weights,
                ]
            )


@pytest.fixture(scope="module")
def routing(tmp_path_factory) -> dict[str, Path]:
    """One evaluator run of the MoE block's `router` child, as what a check reads.

    The inputs, a checkpoint holding the one weight that child declares, its two
    outputs, the same indices with one of them changed, and a zero reference.
    """
    where = tmp_path_factory.mktemp("routing")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    namespace, _ = load_namespace(ROUTING)
    leaf = select_ir(namespace, "Qwen3_5Router")
    declared = leaf.lookup("routing")

    generator = torch.Generator(device=device).manual_seed(11)
    drawn = [
        torch.randn(tuple(param.type.shape), generator=generator, device=device).to(
            to_torch_dtype(param.type.dtype)
        )
        for param in declared.params
    ]
    tokens, w_router = drawn
    weights, indices = evaluate(
        leaf.load(DictResource({"w_router": w_router})).routing,
        tokens,
    )

    torch.save(tokens.cpu(), where / "tokens.pt")
    torch.save(weights.cpu(), where / "weights.pt")
    torch.save(torch.zeros_like(weights).cpu(), where / "zeros.pt")
    numpy.save(where / "indices.npy", indices.cpu().numpy())
    changed = indices.clone()
    changed[0, 0] = (changed[0, 0] + 1) % w_router.shape[-1]
    numpy.save(where / "one_off.npy", changed.cpu().numpy())

    save_file({"w_router": w_router.cpu()}, str(where / "model.safetensors"))
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
        "check",
        ROUTING,
        "--inputs",
        f"files:{routing['tokens']}",
        "--weights",
        f"ckpt:{routing['dir']}",
        "--expected",
        str(routing["weights"]),
        "--expected",
        str(routing[indices]),
        *comparison,
    ]


def test_outputs_judged_per_dtype(routing, capsys) -> None:
    """One call, two kinds of output, each with its own comparison and verdict."""
    assert (
        cli.main(
            _routing_argv(
                routing,
                "indices",
                "--out",
                "output[0]",
                "--fn",
                "allclose",
                "--atol",
                "1e-3",
                "--rtol",
                "4e-3",
                "--out",
                "output[1]",
                "--fn",
                "equal",
            )
        )
        == 0
    )
    reported = capsys.readouterr().out

    assert "output[0]   bf16[1,8]" in reported and "output[1]   i64[1,8]" in reported
    assert "allclose(atol=0.001 rtol=0.004)" in reported
    assert "equal" in reported and "elements 8" in reported
    assert reported.rstrip().endswith("PASS") or "\nPASS" in reported


def test_one_wrong_index_fails(routing, capsys) -> None:
    """A single changed index is a total failure that no aggregate would see."""
    assert (
        cli.main(
            _routing_argv(
                routing,
                "one_off",
                "--out",
                "output[0]",
                "--fn",
                "allclose",
                "--atol",
                "1e-3",
                "--rtol",
                "4e-3",
                "--out",
                "output[1]",
                "--fn",
                "equal",
            )
        )
        == 1
    )
    reported = capsys.readouterr().out

    assert "mismatched 1" in reported
    assert "FAIL" in reported


def test_zero_reference_reported(routing, capsys) -> None:
    """`ref_norm` 0 and an absolute distance, not a number scaled by a clamp."""
    assert (
        cli.main(
            [
                "check",
                ROUTING,
                    "--inputs",
                    f"files:{routing['tokens']}",
                    "--weights",
                    f"ckpt:{routing['dir']}",
                "--expected",
                str(routing["zeros"]),
                "--expected",
                str(routing["indices"]),
                "--out",
                "output[0]",
                "--fn",
                "rel_l2",
                "--max",
                "1e-3",
                "--fn",
                "cosine",
                "--min",
                "0.999",
                "--out",
                "output[1]",
                "--fn",
                "equal",
            ]
        )
        == 1
    )
    reported = capsys.readouterr().out

    assert "ref_norm 0" in reported
    assert "absolute_l2" in reported
    assert "the reference norm is zero" in reported
    assert "one side is entirely zero" in reported

    assert "e+12" not in reported


def test_json_report_matches_text(routing, capsys, tmp_path) -> None:
    """Including `ref_norm` and each predicate's own bound and value."""
    report_path = tmp_path / "report.json"
    assert (
        cli.main(
            _routing_argv(
                routing,
                "indices",
                "--out",
                "output[0]",
                "--fn",
                "rel_l2",
                "--max",
                "1e-3",
                "--out",
                "output[1]",
                "--fn",
                "equal",
                "--json",
                str(report_path),
            )
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    reported = json.loads(report_path.read_text(encoding="utf-8"))

    assert reported["passed"] is True
    outputs = reported["runs"][0]["outputs"]
    assert [output["path"] for output in outputs] == ["output[0]", "output[1]"]
    assert outputs[0]["ref_norm"] > 0
    assert outputs[0]["fns"][0] == {
        "fn": "rel_l2",
        "max": 1e-3,
        "rel_l2": outputs[0]["fns"][0]["rel_l2"],
        "passed": True,
    }
    assert "verification" not in reported


@pytest.mark.parametrize(
    "comparison, refused",
    [
        pytest.param(
            [
                "--out",
                "output[0]",
                "--fn",
                "rel_l2",
                "--max",
                "1e-3",
                "--out",
                "output[1]",
                "--fn",
                "cosine",
                "--min",
                "0.99",
            ],
            "output[1] is i64; cosine is not meaningful on a discrete output",
            id="an-aggregate-over-indices",
        ),
        pytest.param([], "no comparison requested", id="no-predicate-at-all"),
        pytest.param(
            [
                "--out",
                "output[0]",
                "--fn",
                "allclose",
                "--atol",
                "1e-3",
                "--out",
                "output[1]",
                "--fn",
                "equal",
            ],
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
def test_check_refuses_unanswerable_comparisons(routing, capsys, comparison, refused) -> None:
    """Each refusal names what is missing; none of them has a default to fall back on."""
    assert cli.main(_routing_argv(routing, "indices", *comparison)) == 1

    assert refused in capsys.readouterr().err


def test_passing_check_omits_ranking(routing, capsys) -> None:
    """A PASS says the candidate matches this Module, and claims nothing further.

    It used to append a ranking read out of the shipped catalog. That was a claim
    about tests, carried in data the command cannot check, and it outlived the test
    it named -- so it is gone, and the warnings that are about *this run* stay.
    """
    assert (
        cli.main(
            _routing_argv(
                routing,
                "indices",
                "--out",
                "output[0]",
                "--fn",
                "rel_l2",
                "--max",
                "1e-3",
                "--out",
                "output[1]",
                "--fn",
                "equal",
            )
        )
        == 0
    )
    reported = capsys.readouterr().out

    assert "PASS" in reported
    for absent in ("verification on record", "L1", "L2", "L3", "usable as an oracle"):
        assert absent not in reported
    assert "--inputs random makes each activation independently" not in reported
    assert "FAIL says the candidate and reference differ" not in reported


def test_random_input_failure_states_limits(
    routing, capsys, tmp_path
) -> None:
    """A failed random-input comparison qualifies both limits through the CLI."""
    argv = [
        "check",
        ROUTING,
        "--inputs",
        "random",
        "--weights",
        "random",
        "--expected",
        str(routing["zeros"]),
        "--expected",
        str(routing["indices"]),
        "--out",
        "output[0]",
        "--fn",
        "allclose",
        "--atol",
        "1e-3",
        "--rtol",
        "4e-3",
        "--out",
        "output[1]",
        "--fn",
        "equal",
    ]

    assert cli.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    reported = captured.out
    assert "--inputs random makes each activation independently" in reported
    assert "Rerun with --inputs files:" in reported
    assert "FAIL says the candidate and reference differ" in reported
    assert "Establishing accuracy needs an independent high-precision" in reported
    assert "reference, which check does not run" in reported

    report_path = tmp_path / "warnings.json"
    assert cli.main([*argv, "--json", str(report_path)]) == 1
    assert capsys.readouterr().out == ""
    warnings = json.loads(report_path.read_text(encoding="utf-8"))["warnings"]
    assert warnings == [
        "--inputs random makes each activation independently. A target that relies on "
        "semantic relationships between activations can differ at ulp scale without either "
        "implementation being wrong. Rerun with --inputs files:... to decide the comparison.",
        "FAIL says the candidate and reference differ, not which side is closer to truth. "
        "The reference may carry its own rounding; check compares only against it. "
        "Establishing accuracy needs an independent high-precision reference, which check "
        "does not run.",
    ]


def test_nested_child_reads_own_checkpoint(routing, tmp_path) -> None:
    """Reaching `router.routing` reads `router.w_router`.

    Reaching `router.routing` reads `router.w_router`: the checkpoint holds
    that one tensor and none of the eight the block around it declares.
    """
    source = Path(__file__).parents[1] / "fixtures/cli/nested_checkpoint.py"
    save_file(
        {"router.w_router": load_file(str(routing["dir"] / "model.safetensors"))["w_router"]},
        str(tmp_path / "model.safetensors"),
    )
    argv = _routing_argv(
        routing,
        "indices",
        "--out",
        "output[0]",
        "--fn",
        "nan_inf",
        "--out",
        "output[1]",
        "--fn",
        "equal",
    )
    argv[1] = f"{source}:Root.routing"
    argv[argv.index("--weights") + 1] = f"ckpt:{tmp_path}"
    assert cli.main(argv) == 0


def test_check_binds_composed_root_extent(capsys) -> None:
    """Binding an extent must not turn the child call back into an exact one."""
    source = Path(__file__).parents[1] / "fixtures/cli/composed_leaf_n_check.py"
    assert (
        cli.main(
            [
                "check",
                f"{source}:Composed.root",
                *MESH_CHECK_ARGS,
                "--dim",
                "n_check=4",
            ]
        )
        == 0
    )
    assert "PASS" in capsys.readouterr().out
