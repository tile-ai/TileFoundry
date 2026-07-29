"""Command-line interface for authored TileFoundry HIR analysis."""

from __future__ import annotations

import argparse
import contextlib
import io
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from tilefoundry.analysis import AnalysisError, ExtractError
from tilefoundry.analysis.api import analyze
from tilefoundry.inspection import PythonPrintOptions, as_script
from tilefoundry.inspection.analysis_report import (
    render_json,
    render_text,
    report,
    selected_types,
)
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.module import Module, select
from tilefoundry.ir.hir.function import Function
from tilefoundry.schedule import ScheduleError, ScheduleOptions, schedule
from tilefoundry.target import CudaTarget
from tilefoundry.target.hardware import format_capabilities, hardware_documents

_HELP_SPEC_TOPICS = {
    "cli": "cli",
    "dsl": "hir",
}


def _source_spec_path(topic: str) -> Path:
    """Find one source-tree spec used by editable and direct invocations."""
    spec_name = _HELP_SPEC_TOPICS.get(topic, topic)
    return Path(__file__).resolve().parents[2] / "docs" / "spec" / f"{spec_name}.md"


def spec_path(topic: str) -> Path:
    """Return an installed spec path, falling back to the source tree."""
    spec_name = _HELP_SPEC_TOPICS.get(topic, topic)
    source_path = _source_spec_path(topic)
    if source_path.is_file():
        return source_path

    # setuptools data-files are placed below Python's installation data prefix.
    from sysconfig import get_path  # noqa: PLC0415

    installed = (
        Path(get_path("data"))
        / "share"
        / "tilefoundry"
        / "spec"
        / f"{spec_name}.md"
    )
    if installed.is_file():
        return installed
    raise FileNotFoundError(f"installed TileFoundry {spec_name} spec was not found")


def dsl_spec_path() -> Path:
    """Return the HIR spec exposed by the historical ``dsl`` help topic."""
    return spec_path("dsl")


def read_spec(topic: str) -> str:
    """Read the single source of truth for a `tilefoundry help` topic."""
    return spec_path(topic).read_text(encoding="utf-8")


def read_dsl_spec() -> str:
    """Read the HIR spec exposed by the historical ``dsl`` help topic."""
    return read_spec("dsl")


# The root analyses `analyze` can be asked for, in the order they are reported.
_ANALYSES = ("compute-cost", "memory", "roofline", "timeline")


def _split_source(source: str) -> tuple[Path, str | None]:
    path_text, separator, selector = source.partition(":")
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source file not found: {path}")
    if separator and not selector:
        raise ValueError("empty source selector after ':'")
    return path, selector or None


def _unique_values(namespace: dict[str, object], kind: type) -> tuple[object, ...]:
    values: list[object] = []
    seen: set[int] = set()
    for value in namespace.values():
        if isinstance(value, kind) and id(value) not in seen:
            seen.add(id(value))
            values.append(value)
    return tuple(values)


def _select_ir(namespace: dict[str, object], selector: str | None) -> Module:
    if selector is not None:
        # Validated before the join, which would turn `Root.` into the empty
        # path -- and that deliberately means the root itself.
        segments = selector.split(".")
        if any(not segment for segment in segments):
            raise ValueError(
                f"selector {selector!r}: an empty segment names nothing. A path is "
                f"its segments, so a leading, trailing or doubled dot would make "
                f"two different selectors name one node"
            )
        root_name, *path = segments
        selected = namespace.get(root_name)
        if selected is None:
            raise ValueError(f"selector {root_name!r} is not defined by the source")
        if not path:
            if isinstance(selected, Module):
                return selected
            raise TypeError(
                f"selector {root_name!r} resolves to "
                f"{type(selected).__name__}, expected a Module. A Function "
                "carries neither the Target its numbers are measured against "
                "nor the topology hierarchy they divide over; select the Module "
                "that declares it."
            )
        if not isinstance(selected, Module):
            raise TypeError(
                f"selector root {root_name!r} is {type(selected).__name__}, expected Module"
            )
        # The path below the root is resolved by the IR's own selector, so the
        # CLI and the corpus reach a nested kernel the same way.
        return select(selected, ".".join(path))

    modules = _unique_values(namespace, Module)
    if len(modules) == 1:
        return modules[0]  # type: ignore[return-value]
    if len(modules) > 1:
        names = ", ".join(sorted(module.name for module in modules))
        raise ValueError(f"source defines multiple Modules ({names}); add ':Module'")
    functions = _unique_values(namespace, Function)
    if functions:
        names = ", ".join(sorted(function.name for function in functions))
        raise ValueError(
            f"source defines no Module, only Functions ({names}). A Function "
            "carries neither a Target nor a topology hierarchy; declare the "
            "Module that owns it, or select one with ':Module.function'"
        )
    raise ValueError("source defines no TileFoundry Module")


def load_authored_ir(source: str) -> Module:
    """Execute one authored file and resolve its optional IR selector.

    The result is always a Module. A bare Function is rejected rather than
    resolved: it declares neither the Target its numbers would be measured
    against nor the topology hierarchy they divide over.
    """
    path, selector = _split_source(source)
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        namespace = runpy.run_path(str(path))
    return _select_ir(namespace, selector)


def _selected_target(ir: Module):
    """The Target the selection declares. Schedule and Analyze read hardware
    facts off it, so an undeclared Target is an authoring error rather than a
    cue to pick one: the selection must name the device it was written for."""
    if not isinstance(ir, Module):
        raise TypeError(
            f"expected a Module selection, got {type(ir).__name__}. A Function "
            "carries no Target; select the Module that declares it."
        )
    entry = ir.entry_function()
    if not isinstance(entry, Function):
        raise TypeError("capabilities requires a HIR Function entry")
    return ir.resolve_target()


def _grid_cta_count(ir: Module | Function) -> int | None:
    if not isinstance(ir, Module):
        return None
    counts = {
        topology.size
        for topology in ir.effective_topologies()
        if topology.name == "cta" and isinstance(topology.size, int)
    }
    return next(iter(counts)) if len(counts) == 1 else None


def parse_dims(stated: Sequence[str] | None) -> dict[str, int] | None:
    """``NAME=EXTENT`` arguments as the mapping the operations take.

    ``None`` when nothing was stated, which is not the same as an empty mapping:
    a caller who stated no size is asking about the program as authored, while an
    empty mapping is a caller who meant to choose sizes and named none.
    """
    if not stated:
        return None
    dims: dict[str, int] = {}
    for entry in stated:
        name, _, extent = entry.partition("=")
        if not name or not extent:
            raise ValueError(f"--dim takes NAME=EXTENT, got {entry!r}")
        # Repeating the flag states another dimension, not another value for one
        # already stated. Two extents for one dimension is a request with no
        # answer, and taking the last would silently pick one of them.
        if name in dims:
            raise ValueError(
                f"--dim {name} was given twice, as {dims[name]} and {extent}; "
                f"a dimension takes one extent"
            )
        try:
            dims[name] = int(extent)
        except ValueError:
            raise ValueError(
                f"--dim {name}: extent must be an integer, got {extent!r}"
            ) from None
    return dims


def run_authored_analysis(
    source: str,
    analyses: tuple[str, ...],
    *,
    as_json: bool = False,
    dims: Mapping[str, int] | None = None,
) -> int:
    """Analyse one authored HIR selection and print what was found.

    One public call per requested root, because the operation takes one root at
    a time. The renderings are composed from those results afterwards, so
    requesting two analyses cannot change what either of them reports.
    """
    module = load_authored_ir(source)
    function = module.entry_function()
    results = [
        analyze(module, function, analysis=name, dims=dims) for name in analyses
    ]
    data = report(results)
    if as_json:
        sys.stdout.write(f"{render_json(data)}\n")
        return 0
    # The annotated IR and the report are two views of one run, so they choose
    # what to show the same way: a dependency that ran unrequested is not
    # commented into the source either.
    annotated = as_script(
        module,
        options=PythonPrintOptions(
            show_types=True, comment_metadata_types=selected_types(results)
        ),
    )
    sys.stdout.write(f"{render_text(data)}\n\n{annotated}")
    return 0


def _entry_function(ir: Module | Function) -> Function:
    """Resolve the HIR Function `schedule` runs its pipeline over -- the
    same Module -> entry_function() convention as `_selected_target`."""
    function = ir.entry_function() if isinstance(ir, Module) else ir
    if not isinstance(function, Function):
        raise TypeError(f"schedule requires a HIR Function entry, got {type(function).__name__}")
    return function


def run_schedule(
    source: str,
    topology: str,
    *,
    as_json: bool = False,
    dims: Mapping[str, int] | None = None,
    solver_timeout: float | None = None,
    solver_workers: int | None = None,
    first_plan: bool = False,
) -> int:
    """Schedule one authored Module through the public Schedule operation.

    The solver's budget is stated here rather than left to the library default
    because two things about it are the caller's to decide and neither is visible
    from inside. The worker count: the default lets the solver size itself to the
    machine, and several solvers each doing that on one machine oversubscribe it
    until none returns an answer, which looks like the model being unschedulable and
    is not. And whether the best plan is wanted at all: the search keeps improving
    until its limit, so a caller who needs a plan rather than the best one otherwise
    waits out the whole budget for an answer it had early.
    """
    ir = load_authored_ir(source)
    function = _entry_function(ir)
    options = None
    if solver_timeout is not None or solver_workers is not None or first_plan:
        options = ScheduleOptions(stop_at_first_solution=first_plan)
        if solver_timeout is not None:
            options = replace(options, timeout_seconds=solver_timeout)
        if solver_workers is not None:
            options = replace(options, workers=solver_workers)
    result = schedule(ir, function, topology=topology, dims=dims, options=options)
    sys.stdout.write((result.plan.to_json() if as_json else result.plan.render()) + "\n")
    return 0


def _add_source_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source",
        metavar="SOURCE",
        help="model.py[:Module[.child_module...][.function]]",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tilefoundry")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="type-check and analyze authored HIR")
    _add_source_argument(analyze)
    for analysis in _ANALYSES:
        analyze.add_argument(
            f"--{analysis}", action="store_true", help=f"run the {analysis} analysis"
        )
    analyze.add_argument(
        "--dim",
        action="append",
        metavar="NAME=EXTENT",
        help="bind one dimension the model left open, for example ctx_len=1024",
    )
    analyze.add_argument(
        "--json", action="store_true", help="print the report as JSON instead of text"
    )

    schedule = commands.add_parser(
        "schedule",
        help="schedule authored HIR at one declared topology level",
    )
    _add_source_argument(schedule)
    schedule.add_argument(
        "--topology",
        required=True,
        metavar="LEVEL",
        help="declared topology level to schedule (for example cta)",
    )
    schedule.add_argument(
        "--dim",
        action="append",
        metavar="NAME=EXTENT",
        help="bind one dimension the model left open, for example ctx_len=1024",
    )
    schedule.add_argument("--json", action="store_true", help="print the selected plan as JSON")
    schedule.add_argument(
        "--solver-timeout",
        type=float,
        metavar="SECONDS",
        help="how long the solver may search before it reports no answer",
    )
    schedule.add_argument(
        "--solver-workers",
        type=int,
        metavar="COUNT",
        help=(
            "how many search workers the solver may use; the default lets it "
            "size itself to the machine, which oversubscribes when several "
            "schedules run at once"
        ),
    )
    schedule.add_argument(
        "--first-plan",
        action="store_true",
        help="stop at the first plan that satisfies the constraints instead of "
        "searching the whole budget for the best one",
    )

    inspect = commands.add_parser("inspect", help="inspect installed target facts")
    inspect_commands = inspect.add_subparsers(dest="inspect_command", required=True)
    capabilities = inspect_commands.add_parser("capabilities", help="print target capabilities")
    _add_source_argument(capabilities)

    help_command = commands.add_parser("help", help="print installed reference material")
    help_command.add_argument("topic", choices=("dsl", "cli"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "help":
        sys.stdout.write(read_spec(args.topic))
        return 0
    if args.command == "inspect":
        try:
            ir = load_authored_ir(args.source)
            target = _selected_target(ir)
            if not isinstance(target, CudaTarget):
                raise ValueError(
                    f"no installed authored-analysis hardware spec for target {target.name!r}"
                )
            sys.stdout.write(
                format_capabilities(
                    hardware_documents(target),
                    grid_cta_count=_grid_cta_count(ir),
                )
                + "\n"
            )
            return 0
        except Exception as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
    if args.command == "schedule":
        try:
            return run_schedule(
                args.source,
                args.topology,
                as_json=args.json,
                dims=parse_dims(args.dim),
                solver_timeout=args.solver_timeout,
                solver_workers=args.solver_workers,
                first_plan=args.first_plan,
            )
        except (
            ExtractError,
            ScheduleError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1

    analyses = tuple(
        name for name in _ANALYSES if getattr(args, name.replace("-", "_"))
    )
    if not analyses:
        analyses = _ANALYSES
    try:
        return run_authored_analysis(
            args.source, analyses, as_json=args.json, dims=parse_dims(args.dim)
        )
    except (AnalysisError, VerifyError, OSError, TypeError, ValueError) as error:
        print(f"tilefoundry: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
