"""Command-line interface for authored TileFoundry HIR analysis."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import runpy
import sys
from pathlib import Path
from typing import Sequence

from tilefoundry.analysis import AnalysisError, ExtractError, TileGraph, extract
from tilefoundry.analysis.api import analyze
from tilefoundry.inspection import PythonPrintOptions, as_script
from tilefoundry.inspection.analysis_report import (
    render_json,
    render_text,
    report,
    selected_types,
)
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.schedule.kernel_schedule import KernelScheduleError, build_schedule_tree
from tilefoundry.schedule.render import EmitScaffoldError, HoleContract, emit_scaffold
from tilefoundry.schedule.select_atoms import AtomSelectionError, select_atoms
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


def _detached_selection(module: Module, entry: str) -> Module:
    """*module* re-entried at *entry*.

    ``replace`` rebuilds the value without its owner backlink, so a child
    selected out of a tree would lose the Target and hierarchy it inherits.
    The copy therefore carries the context it resolved through.
    """
    try:
        target = module.resolve_target()
    except ValueError:
        target = module.target
    return dataclasses.replace(
        module,
        entry=entry,
        target=target,
        topologies=module.effective_topologies(),
    )


def _select_ir(namespace: dict[str, object], selector: str | None) -> Module:
    if selector is not None:
        root_name, *path = selector.split(".")
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
        # Walk the child Modules the path names, so a leaf deep in the tree is
        # selected through the owners it inherits its context from.
        for index, name in enumerate(path):
            children = {child.name: child for child in selected.modules}
            if name in children:
                selected = children[name]
                continue
            if index != len(path) - 1:
                raise ValueError(
                    f"selector {selector!r}: Module {selected.name!r} has no "
                    f"child module {name!r}"
                )
            function = selected.lookup(name)
            if not isinstance(function, Function):
                raise TypeError(
                    f"selector {selector!r} resolves to {type(function).__name__}, "
                    "expected HIR Function"
                )
            # Naming a function selects it without losing the Target and
            # Topologies of the Module it runs against.
            return _detached_selection(selected, name)
        return selected

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


def run_authored_analysis(
    source: str, analyses: tuple[str, ...], *, as_json: bool = False
) -> int:
    """Analyse one authored HIR selection and print what was found.

    One public call per requested root, because the operation takes one root at
    a time. The renderings are composed from those results afterwards, so
    requesting two analyses cannot change what either of them reports.
    """
    module = load_authored_ir(source)
    function = module.entry_function()
    results = [analyze(module, function, analysis=name) for name in analyses]
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


def _decisions_of(solved: TileGraph) -> dict:
    """The resource decisions `select_atoms` records on its output."""
    return solved.decisions


def _hole_contract_line(contract: HoleContract) -> str:
    op_name = type(contract.op_ref.target).__name__
    inputs = ",".join(view.tensor_name for view in contract.inputs)
    coords = ",".join(contract.coords)
    return (
        f"hole={contract.name} op={op_name} coords={coords} "
        f"inputs={inputs} output={contract.output.tensor_name}"
    )


def _stage_or_die(target, stage: str) -> str:
    """``stage`` if the resolved target owns a level by that name. A target
    that enumerates its levels can name the alternatives; one that does not
    leaves the service lookup inside `select_atoms` to report the mismatch."""
    levels = getattr(target, "topology_levels", ())
    if levels and stage not in levels:
        raise ValueError(
            f"target {target.name!r} has no topology level {stage!r}; "
            f"--stage must be one of {', '.join(levels)}"
        )
    return stage


def run_schedule(source: str, stage: str) -> int:
    """Model, schedule, select atoms for, and scaffold one authored HIR
    Function at one of its target's topology levels -- the schedule-path
    analogue of `run_authored_analysis`: same source loading, ``#``-headed
    machine-parsable summary style. The target comes from the selected Module's
    resolved Target, not from a flag: a kernel is authored against one."""
    ir = load_authored_ir(source)
    function = _entry_function(ir)
    resolved_target = _selected_target(ir)
    stage = _stage_or_die(resolved_target, stage)

    tg = extract(function)
    solved = select_atoms(build_schedule_tree(tg), target=resolved_target, stage=stage)
    skeleton, swimlane, contracts = emit_scaffold(solved)
    decisions = _decisions_of(solved)

    header = [
        f"schedule target={resolved_target.name} stage={stage} function={function.name} "
        f"statements={','.join(unit.name for unit in tg.units)}",
        f"decisions status={decisions['status']} makespan={decisions['makespan']}",
    ]
    for name, stmt in decisions["statements"].items():
        header.append(
            f"decisions statement={name} atom={stmt['atom'] or 'none'} "
            f"place={stmt['place']} start={stmt['start']} end={stmt['end']}"
        )
    if solved.ring:
        header.append(
            "ring " + " ".join(f"{buf}={depth}" for buf, depth in sorted(solved.ring.items()))
        )
    summary = "\n".join(f"# {line}" for line in header)
    holes = "\n".join(f"# {_hole_contract_line(contract)}" for contract in contracts)

    sys.stdout.write(
        f"{summary}\n\n"
        f"# skeleton\n{skeleton.text}\n"
        f"# swimlane\n{swimlane.text}\n\n"
        f"# holes\n{holes}\n"
    )
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
        "--json", action="store_true", help="print the report as JSON instead of text"
    )

    schedule = commands.add_parser(
        "schedule",
        help="schedule authored HIR at one topology level into an agent-fillable scaffold",
    )
    _add_source_argument(schedule)
    schedule.add_argument(
        "--stage",
        required=True,
        metavar="LEVEL",
        help="topology level to schedule at (a level the Module's target owns, e.g. core)",
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
            return run_schedule(args.source, args.stage)
        except (
            ExtractError,
            EmitScaffoldError,
            AtomSelectionError,
            KernelScheduleError,
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
        return run_authored_analysis(args.source, analyses, as_json=args.json)
    except (AnalysisError, VerifyError, OSError, TypeError, ValueError) as error:
        print(f"tilefoundry: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
