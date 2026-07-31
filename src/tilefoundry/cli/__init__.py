"""Command-line interface for authored TileFoundry HIR analysis."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from tilefoundry.analysis import AnalysisError, ExtractError
from tilefoundry.cli.analyze import ANALYSES, run_authored_analysis
from tilefoundry.cli.check import add_arguments as add_check_arguments
from tilefoundry.cli.check import guidance, run_check
from tilefoundry.cli.inspect import run_capabilities
from tilefoundry.cli.models import run_models
from tilefoundry.cli.schedule import run_schedule
from tilefoundry.cli.source import load_authored_ir, parse_dims
from tilefoundry.cli.spec import read_spec, run_spec, spec_path
from tilefoundry.cli.tutorial import PAGES, run_tutorial
from tilefoundry.ir.core import VerifyError
from tilefoundry.schedule import ScheduleError

_ANALYSES = ANALYSES

#: Every command and its one-line description, in the order an agent meets them
#: rather than alphabetically -- the order itself is meant to read as the workflow.
#: One table, so the parser and the overview cannot describe different surfaces.
_COMMANDS = {
    "models": "list the described models, or show one of them",
    "spec": "read one specification: its sections, or one of them",
    "tutorial": "learn the two-step workflow: its pages, or one of them",
    "check": "compare an implementation against its reference, output by output",
    "analyze": "type-check and analyze authored HIR",
    "schedule": "schedule authored HIR at one declared topology level",
    "inspect": "inspect installed target facts",
}

_INSPECT_COMMANDS = {
    "capabilities": (
        "the facts a selection's target was composed from, or the installed "
        "hardware documents there are"
    ),
}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self._print_message(f"{self.prog}: error: {message}\n\n", sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2)


def _project_summary() -> str:
    """The packaged one-line description of the project.

    Read from installed metadata rather than restated here, so there is one copy
    of the sentence and no second one to drift.
    """
    from importlib.metadata import metadata  # noqa: PLC0415

    return metadata("tilefoundry")["Summary"].rstrip(".")


def overview() -> str:
    """What a bare invocation prints: what this is, and how to ask it something."""
    width = max(len(name) for name in _COMMANDS)
    commands = "\n".join(
        f"  {name:<{width}}  {description}" for name, description in _COMMANDS.items()
    )
    return (
        f"TileFoundry — {_project_summary()}\n"
        f"\n"
        f"Usage:\n"
        f"  tilefoundry <command> [options]\n"
        f"\n"
        f"Common commands:\n"
        f"{commands}\n"
        f"\n"
        f"Options:\n"
        f"  -h, --help  print this, or a command's own help after the command\n"
    )


def _inspect_overview() -> str:
    """What ``tilefoundry inspect`` prints without a subcommand."""
    width = max(len(name) for name in _INSPECT_COMMANDS)
    commands = "\n".join(
        f"  {name:<{width}}  {description}"
        for name, description in _INSPECT_COMMANDS.items()
    )
    return (
        f"tilefoundry inspect — {_COMMANDS['inspect']}\n"
        f"\n"
        f"Usage:\n"
        f"  tilefoundry inspect <command> [options]\n"
        f"\n"
        f"Commands:\n"
        f"{commands}\n"
        f"\n"
        f"Options:\n"
        f"  -h, --help  print this, or a command's own help after the command\n"
    )


def _add_source_argument(
    parser: argparse.ArgumentParser, *, optional: bool = False
) -> None:
    arguments = {
        "metavar": "SOURCE",
        "help": "model.py[:Module[.child_module...][.function]]",
    }
    if optional:
        arguments["nargs"] = "?"
    parser.add_argument(
        "source",
        **arguments,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="tilefoundry")
    # Not required: naming no command is how the overview is asked for.
    commands = parser.add_subparsers(dest="command", parser_class=_Parser)

    models = commands.add_parser("models", help=_COMMANDS["models"])
    models.add_argument(
        "name",
        nargs="?",
        metavar="NAME",
        help="which model; with none, list the models there are",
    )
    models.add_argument(
        "--source", action="store_true", help="print the model's authored source instead"
    )

    tutorial = commands.add_parser("tutorial", help=_COMMANDS["tutorial"])
    tutorial.add_argument(
        "page",
        nargs="?",
        choices=PAGES[1:],
        metavar="PAGE",
        help="which page; with none, the overview and the pages there are",
    )

    spec = commands.add_parser("spec", help=_COMMANDS["spec"])
    spec.add_argument(
        "topic",
        nargs="?",
        metavar="TOPIC",
        help="which document; with none, list the documents there are",
    )
    spec.add_argument(
        "section",
        nargs="?",
        metavar="SECTION",
        help="one section's key, as the outline prints it; with none, print the outline",
    )

    check = commands.add_parser(
        "check",
        help=_COMMANDS["check"],
        epilog=guidance(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_check_arguments(check)

    analyze = commands.add_parser("analyze", help=_COMMANDS["analyze"])
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

    schedule = commands.add_parser("schedule", help=_COMMANDS["schedule"])
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

    inspect = commands.add_parser("inspect", help=_COMMANDS["inspect"])
    inspect_commands = inspect.add_subparsers(dest="inspect_command", parser_class=_Parser)
    capabilities = inspect_commands.add_parser(
        "capabilities", help=_INSPECT_COMMANDS["capabilities"]
    )
    _add_source_argument(capabilities, optional=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        sys.stdout.write(overview())
        return 0
    if args.command == "models":
        try:
            return run_models(args.name, source=args.source)
        except (OSError, ValueError) as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
    if args.command == "spec":
        try:
            return run_spec(args.topic, args.section)
        except (OSError, ValueError) as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
    if args.command == "tutorial":
        try:
            return run_tutorial(args.page)
        except (OSError, ValueError) as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
    if args.command == "check":
        try:
            return run_check(args)
        except Exception as error:
            print(f"tilefoundry check: error: {error}", file=sys.stderr)
            return 1
    if args.command == "inspect":
        if args.inspect_command is None:
            sys.stdout.write(_inspect_overview())
            return 0
        try:
            return run_capabilities(args.source)
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


__all__ = [
    "build_parser",
    "load_authored_ir",
    "main",
    "overview",
    "parse_dims",
    "read_spec",
    "spec_path",
]


if __name__ == "__main__":
    raise SystemExit(main())
