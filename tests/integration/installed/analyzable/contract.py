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

from tests.models.corpus import FunctionCase, ModelCase
from tests.models.decode_oracle import one_ulp_at
from tests.models.registry import cases_of

#: The analyses the CLI offers as public flags, asked one at a time so each
#: family is its own verdict rather than one pass or fail for four questions.
FAMILIES = ("compute-cost", "memory", "roofline", "timeline")

#: The solver budget the in-process schedule witnesses used, stated to the command
#: rather than left to the library default: the default worker count sizes itself
#: to the machine, and several of these at once oversubscribes it so that none
#: returns an incumbent -- which reads as a model that cannot be scheduled.
#: ``--first-plan`` because what is asked is whether a plan exists at all.
SOLVER = ("--solver-timeout=60", "--solver-workers=4", "--first-plan")


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
    done = tf(
        "analyze",
        static(source, case, selector),
        f"--{family}",
        *(("--json",) if json_output else ()),
        *dim_args(dims),
    )
    assert done.returncode == 0, done.stderr
    return done


def reported(
    tf,
    source: Path,
    case: ModelCase,
    selector: str,
    families: tuple[str, ...],
    dims: Mapping[str, int] | None = None,
) -> dict:
    """The JSON report several families write about one function."""
    done = tf(
        "analyze",
        static(source, case, selector),
        *(f"--{family}" for family in families),
        "--json",
        *dim_args(dims),
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def scheduled(tf, source: Path, case: ModelCase, planned: FunctionCase, *, topology: str = ""):
    """One ``schedule`` command at a level the source has to declare itself."""
    level = topology or planned.topology
    done = tf(
        "schedule",
        static(source, case, planned.selector),
        "--topology",
        level,
        *dim_args(planned.dims),
        *SOLVER,
    )
    assert done.returncode == 0, done.stderr
    return done


def lifetimes(
    tf, source: Path, case: ModelCase, selector: str, dims: Mapping[str, int]
) -> dict[str, int]:
    """What the memory analysis says each binding of one function costs."""
    report = reported(tf, source, case, selector, ("memory",), dims)
    return {
        item["binding"]: item["bytes"]
        for item in report["function_records"]["memory"]["lifetimes"]
    }


def traffic_read(
    tf, source: Path, case: ModelCase, selector: str, dims: Mapping[str, int]
) -> int:
    """How many bytes the memory analysis says one function reads from gmem."""
    report = reported(tf, source, case, selector, ("memory",), dims)
    return report["function_records"]["memory"]["traffic"]["gmem"]["read"]


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
):
    """One ``check`` command: the shipped source against tensors produced here.

    The oracle is run by the caller and written into *work*, so nothing long-lived
    holds a frozen truth: a comparison that stopped agreeing cannot be made to pass
    by an artifact somebody recorded once.

    The weights travel as a checkpoint rather than as activations, because that is
    the only door ``check`` has for them -- ``--input`` names the non-const
    parameters, in the order the function declares them.
    """
    room = Path(tempfile.mkdtemp(dir=work))
    argv = ["check", static(source, case, selector)]
    for position, tensor in enumerate(activations):
        path = room / f"in{position}.pt"
        torch.save(tensor, path)
        argv += ["--input", str(path)]
    save_file(
        {f"{_reached_through(case, selector)}{name}": value.contiguous()
         for name, value in weights.items()},
        str(room / "model.safetensors"),
    )
    argv += ["--ckpt", str(room), *dim_args(dims)]
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
    assert done.returncode == 0, done.stdout + done.stderr
    return done


def capabilities(tf, source: Path, case: ModelCase, selector: str):
    """``inspect capabilities``, which nothing tells the target: the source states it."""
    done = tf("inspect", "capabilities", static(source, case, selector))
    assert done.returncode == 0, done.stderr
    return done
