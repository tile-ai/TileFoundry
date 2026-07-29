"""Command-line interface for authored TileFoundry HIR analysis."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from tilefoundry.analysis import AnalysisError, ExtractError
from tilefoundry.cli.analyze import ANALYSES, run_authored_analysis
from tilefoundry.cli.inspect import run_capabilities
from tilefoundry.cli.schedule import run_schedule
from tilefoundry.cli.source import load_authored_ir, parse_dims
from tilefoundry.cli.spec import (
    dsl_spec_path,
    read_dsl_spec,
    read_spec,
    spec_path,
)
from tilefoundry.ir.core import VerifyError
from tilefoundry.schedule import ScheduleError

_ANALYSES = ANALYSES


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
    "dsl_spec_path",
    "load_authored_ir",
    "main",
    "parse_dims",
    "read_dsl_spec",
    "read_spec",
    "spec_path",
]


if __name__ == "__main__":
    raise SystemExit(main())
