"""What one corpus case asks of the shipped source, one command per question.

Helper functions rather than a base class: every model states its own ``test_``
functions and calls these, so each test is visible in the file it fails in.

Every helper here runs exactly one command and judges exactly its result. The
machine is never passed in -- the copied ``model.py`` declares its own target and
its own topology levels, and the command reads them back out of the source. There
is no argument for a test to inject them by, which is the property being kept: a
level that stops being declared has nowhere to come from and the command fails.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors.torch import save_file

from tests.models.corpus import FunctionCase, ModelCase, states_execution_domain
from tests.models.decode_oracle import one_ulp_at
from tests.models.registry import cases_of
from tilefoundry.runtime import PREDICATES, DictResource, check

LOGICAL_FAMILIES = ("compute-cost", "memory", "roofline")
FAMILIES = (*LOGICAL_FAMILIES, "performance")


def model_cases(model: str) -> tuple[ModelCase, ...]:
    """Every case the named model states, in the order it states them.

    Read at import so each case is its own parametrized test: a model whose second
    case fails still gives a verdict on its third.
    """
    return cases_of(model)


def static(source: Path, case: ModelCase, selector: str) -> str:
    """The ``path:selector`` the command is given, from the non-empty segments.

    The root the source declares, the domain the case scopes to, and the function
    the case names. A case whose domain is the root itself contributes no segment.
    """
    reached = ".".join(part for part in (case.prototype.name, case.scope, selector) if part)
    return f"{source / 'model.py'}:{reached}"


def dim_args(dims: Mapping[str, int] | None) -> list[str]:
    """The extents a function left open, as the command takes them."""
    return [f"--dim={name}={extent}" for name, extent in (dims or {}).items()]


def _run_with_report(tf, arguments: Sequence[object], *, suffix: str):
    """Run a command whose report lives outside the installed command's stdout."""
    with tempfile.NamedTemporaryFile(suffix=suffix) as report_file:
        done = tf(*arguments, report_file.name)
        report_text = ""
        if done.returncode == 0:
            assert done.stdout == ""
            report_file.seek(0)
            report_text = report_file.read().decode()
        return done, report_text


def analysed(
    tf,
    source: Path,
    case: ModelCase,
    selector: str,
    family: str,
    dims: Mapping[str, int] | None = None,
    *,
    json_output: bool = False,
):
    """One ``analyze`` command for one family, held to succeeding."""
    arguments = [
        "analyze",
        static(source, case, selector),
        f"--{family}",
        *dim_args(dims),
    ]
    if json_output:
        arguments.append("--json")
    done, report_text = _run_with_report(
        tf, arguments, suffix=".json" if json_output else ".py"
    )
    assert done.returncode == 0, done.stderr
    return report_text


def reported(
    tf,
    source: Path,
    case: ModelCase,
    selector: str,
    families: tuple[str, ...],
    dims: Mapping[str, int] | None = None,
) -> dict:
    """The JSON report several families write about one function."""
    done, report_text = _run_with_report(
        tf,
        [
            "analyze",
            static(source, case, selector),
            *(f"--{family}" for family in families),
            *dim_args(dims),
            "--json",
        ],
        suffix=".json",
    )
    assert done.returncode == 0, done.stderr
    return json.loads(report_text)


def _compute_cost_evidence(report: dict) -> str | None:
    """What `compute-cost` must have counted: flops, bucketed by dtype."""
    flops = report["totals"]["flops"]
    if not flops or not any(count > 0 for count in flops.values()):
        return f"reported no flops at all ({flops!r})"
    return None


def _memory_evidence(report: dict) -> str | None:
    """What `memory` must have measured: bytes moved, lifetimes, and addresses.

    The bytes are the family's own record rather than a field of its footprint:
    what a program moves and what it must hold at once are two answers, reported
    as two.
    """
    record = report["function_records"]["memory"]
    placement = record.get("allocation")
    if placement is None or placement["solver_status"] not in ("optimal", "feasible"):
        return f"allocation is {placement!r}"
    gmem = report["function_records"]["traffic"]["whole"]["gmem"]
    if not gmem.get("read", 0) > 0:
        return f"reported no gmem read ({gmem!r})"
    if not record["lifetimes"]:
        return "reported no binding lifetimes"
    return None


def _roofline_evidence(report: dict) -> str | None:
    """What `roofline` must have decided: which side bounds, and the time it implies."""
    record = report["function_records"]["roofline"]
    if record["bound_by"] not in ("compute", "memory", "balanced"):
        return f"bound_by is {record['bound_by']!r}, no rated resource"
    if not record["ideal_ns"] > 0:
        return f"ideal_ns is {record['ideal_ns']!r}"
    return None


def _performance_evidence(report: dict) -> str | None:
    """What `performance` must have laid out and scaled by physical capacity."""
    record = report["function_records"]["performance"]
    envelope = record["timeline"]
    if envelope["start_ns"] != 0 or not envelope["end_ns"] > 0:
        return f"predicted interval is {envelope!r}"
    if not record["waves"] > 0:
        return f"waves is {record['waves']!r}"
    if envelope["end_ns"] % record["waves"]:
        return f"end_ns {envelope['end_ns']!r} is not {record['waves']!r} equal waves"
    return None


_EVIDENCE = {
    "compute-cost": _compute_cost_evidence,
    "memory": _memory_evidence,
    "roofline": _roofline_evidence,
    "performance": _performance_evidence,
}


def analysed_every_family(
    tf, source: Path, case: ModelCase, selector: str, dims: Mapping[str, int] | None = None
) -> dict:
    """Judge every family the selected function is ready to run.

    Which families those are is read off the program: one that runs something
    inside a CTA Mesh answers for all four, and one that runs nothing inside any
    Mesh keeps its logical three and must make a separate performance request
    fail for a missing execution domain. Asking a flag beside the case instead
    would let a program that gained a placement keep being asked the smaller
    question.
    """
    selected = [item for item in case.analyze if item.selector == selector]
    assert len(selected) == 1, f"{case.id}: analysis selector {selector!r} is not unique"
    owner, function = case.resolve(case.build(), selector)
    placed = states_execution_domain(owner, function, dims or selected[0].dims)
    families = FAMILIES if placed else LOGICAL_FAMILIES
    report = reported(tf, source, case, selector, families, dims)
    assert report["executed"] == list(families), (
        f"asked for {list(families)}, the command ran {report['executed']}"
    )
    failed = {
        family: complaint
        for family in families
        if (evidence := _EVIDENCE[family])
        if (complaint := evidence(report)) is not None
    }
    assert not failed, f"{selector} at {dict(dims or {})}: " + "; ".join(
        f"{family} {complaint}" for family, complaint in sorted(failed.items())
    )
    return report


def performance_refused(
    tf,
    source: Path,
    case: ModelCase,
    selected: FunctionCase,
) -> None:
    """One unplaced shipped-model function must identify the domain it lacks."""
    rejected, _report_text = _run_with_report(
        tf,
        [
            "analyze",
            static(source, case, selected.selector),
            "--performance",
            *dim_args(selected.dims),
        ],
        suffix=".py",
    )
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert "performance:" in rejected.stderr
    assert "has no" in rejected.stderr and "execution domain" in rejected.stderr


def lifetimes(
    tf, source: Path, case: ModelCase, selector: str, dims: Mapping[str, int]
) -> dict[str, int]:
    """What the memory analysis says each binding of one function costs."""
    report = reported(tf, source, case, selector, ("memory",), dims)
    return {
        item["binding"]: item["bytes"] for item in report["function_records"]["memory"]["lifetimes"]
    }


def traffic_read(tf, source: Path, case: ModelCase, selector: str, dims: Mapping[str, int]) -> int:
    """How many bytes the memory analysis says one function reads from gmem."""
    report = reported(tf, source, case, selector, ("memory",), dims)
    return report["function_records"]["traffic"]["whole"]["gmem"]["read"]


def one_rounding(want) -> tuple[str, dict[str, float]]:
    """One representable step at *want*'s own scale, for a single primitive boundary."""
    return "allclose", {"atol": one_ulp_at(want), "rtol": 0.0}


def three_roundings(want) -> tuple[str, dict[str, float]]:
    """Three, for a whole fused Function against the component it reproduces.

    One uniform contract for every model here, not per-model or depth-scaled: the
    Function rounds at each boundary it fuses.
    """
    return "allclose", {"atol": 3 * one_ulp_at(want), "rtol": 0.0}


def exactly() -> tuple[str, dict[str, float]]:
    """No tolerance. A gather or a copy reassociates nothing."""
    return "equal", {}


def _reached_through(case: ModelCase, selector: str) -> str:
    """The Modules between the root and the function, as the checkpoint keys them.

    ``check`` scopes the weights it reads to the child Modules the selector passes
    through, so a checkpoint written for one of them has to key its tensors the same
    way -- a bare weight name is not found at all.
    """
    inner = [part for part in (*case.scope.split("."), *selector.split(".")[:-1]) if part]
    return f"{'.'.join(inner)}." if inner else ""


def split_by_declaration(case: ModelCase, selector: str, args: Sequence):
    """One positional argument list, split the way the command takes it.

    An in-process call may hand every parameter over positionally, weights and all.
    ``check`` does not: ``--inputs files:...`` names the non-const parameters in
    declared order and the rest arrive as a checkpoint. Split here from the declaration itself, so
    a parameter that changes kind moves sides on its own.
    """
    _module, function = case.resolve(case.build(), selector)
    activations, weights = [], {}
    for parameter, value in zip(function.params, args, strict=True):
        if getattr(parameter, "is_const", False):
            weights[parameter.name] = value
        else:
            activations.append(value)
    return activations, weights


def nested_constants(loaded, prefix: str = "") -> dict:
    """Every weight a loaded Module holds, its children's included.

    A Module names only its own; a checkpoint has to carry the children's too, keyed
    by the path they are reached through, or the child is loaded with nothing.
    """
    found = {f"{prefix}{name}": loaded.resource.load(name) for name in loaded.module.weights}
    for child in loaded.modules:
        found.update(nested_constants(child, f"{prefix}{child.module.name}."))
    return found


def compared(
    tf,
    work: Path,
    source: Path,
    case: ModelCase,
    selector: str,
    *,
    activations: Sequence,
    weights: Mapping[str, object],
    expected: Sequence,
    held: Sequence[tuple[str, Mapping[str, float]]],
    dims: Mapping[str, int] | None = None,
    _refuse: bool = False,
):
    """Compare shipped HIR through the CLI, or host orchestration in this test.

    The oracle is written into *work*, so a stale artifact cannot hide disagreement.
    Weights travel as a checkpoint because ``check`` accepts non-const parameters
    only through ``--inputs files:...``. An empty selector names host orchestration;
    this test invokes it explicitly and applies the same public comparison layer.
    """
    if not selector:
        if _refuse:
            raise ValueError("an orchestration comparison cannot request a CLI refusal")
        module = case.build()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, (tuple, list)):
                return type(value)(move(item) for item in value)
            return value

        loaded = module.load(DictResource({name: value.to(device) for name, value in weights.items()}))
        got = loaded.forward(*(move(value) for value in activations))
        want = move(expected[0] if len(expected) == 1 else tuple(expected))
        expect = {
            ("output" if len(expected) == 1 else f"output[{position}]"): (
                PREDICATES[predicate](**bounds),
            )
            for position, (predicate, bounds) in enumerate(held)
        }
        report = check(lambda *_args: got, lambda *_args: want, tuple(activations), expect=expect)
        assert report.passed, report
        return report

    room = Path(tempfile.mkdtemp(dir=work))
    argv = ["check", static(source, case, selector)]
    input_paths = []
    for position, tensor in enumerate(activations):
        path = room / f"in{position}.pt"
        torch.save(tensor, path)
        input_paths.append(str(path))
    argv += ["--inputs", f"files:{','.join(input_paths)}"]

    if weights:
        save_file(
            {
                f"{_reached_through(case, selector)}{name}": value.contiguous()
                for name, value in weights.items()
            },
            str(room / "model.safetensors"),
        )
        argv += ["--weights", f"ckpt:{room}"]
    else:
        argv += ["--weights", "random"]
    argv += dim_args(dims)
    for position, tensor in enumerate(expected):
        path = room / f"want{position}.pt"
        torch.save(tensor, path)
        argv += ["--expected", str(path)]
    for position, (predicate, bounds) in enumerate(held):
        out = "output" if len(expected) == 1 else f"output[{position}]"
        argv += ["--out", out, "--fn", predicate]
        for bound, value in bounds.items():
            argv += [f"--{bound}", repr(value)]

    done = tf(*argv)
    if not _refuse:
        assert done.returncode == 0, done.stdout + done.stderr
    return done


def disagreed(tf, work: Path, source: Path, case: ModelCase, selector: str, **asked):
    """The same command, held to reporting FAIL.

    Perturbed runs must make parity refuse. They use the same bound as the passing
    run: a tighter bound can reject unperturbed output and falsely validate the
    perturbation. This previously hid five cases, including an identity cache
    permutation.
    """
    done = compared(tf, work, source, case, selector, _refuse=True, **asked)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "FAIL" in done.stdout, done.stdout
    return done
