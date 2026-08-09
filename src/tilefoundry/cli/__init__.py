"""Command-line interface for authored TileFoundry HIR analysis."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from tilefoundry.analysis import AnalysisError, ExtractError
from tilefoundry.cli.analyze import ANALYSES, EVIDENCE, run_authored_analysis
from tilefoundry.cli.analyze import guidance as analyze_guidance
from tilefoundry.cli.check import add_arguments as add_check_arguments
from tilefoundry.cli.check import guidance as check_guidance
from tilefoundry.cli.check import run_check
from tilefoundry.cli.models import run_models
from tilefoundry.cli.schedule import guidance as schedule_guidance
from tilefoundry.cli.schedule import run_schedule
from tilefoundry.cli.source import load_authored_ir, one_extent_per_dim, parse_dims
from tilefoundry.cli.spec import read_spec, run_spec, spec_path
from tilefoundry.cli.target import load_registrations, registry_path
from tilefoundry.cli.target import run_add_document as run_target_add_document
from tilefoundry.cli.target import run_add_module as run_target_add_module
from tilefoundry.cli.target import run_list as run_target_list
from tilefoundry.cli.target import run_remove as run_target_remove
from tilefoundry.cli.target import run_show as run_target_show
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
    # Named by their evidence, not shared authored HIR input, so a reader who finished
    # HIR does not mistake Analyze or Schedule for a command belonging to that step.
    "analyze": "report what a program costs: flops, traffic, bounds, timing",
    "schedule": "propose a plan for one topology level: placement and timing",
    "target": "list, show, add, or remove compilation targets",
}

_TARGET_COMMANDS = {
    "list": "list every available target as reconstructing Python",
    "show": "show the documents retained by one target identity",
    "add": "add one Target provider or hardware document",
    "remove": "remove one entry shown by target list",
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
        f"  tilefoundry [--registry PATH] <command> [options]\n"
        f"\n"
        f"Common commands:\n"
        f"{commands}\n"
        f"\n"
        f"Options:\n"
        f"  --registry PATH  use this installation registry instead\n"
        f"  -h, --help  print this, or a command's own help after the command\n"
    )


def _target_overview() -> str:
    """What ``tilefoundry target`` prints without a subcommand."""
    width = max(len(name) for name in _TARGET_COMMANDS)
    commands = "\n".join(
        f"  {name:<{width}}  {description}" for name, description in _TARGET_COMMANDS.items()
    )
    return (
        f"tilefoundry target — {_COMMANDS['target']}\n"
        f"\n"
        f"Usage:\n"
        f"  tilefoundry [--registry PATH] target <command> [options]\n"
        f"\n"
        f"Commands:\n"
        f"{commands}\n"
        f"\n"
        f"Options:\n"
        f"  --registry PATH  use this installation registry instead\n"
        f"  -h, --help  print this, or a command's own help after the command\n"
    )


def _add_source_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source",
        metavar="SOURCE",
        help="model.py[:Module[.child_module...][.function]]",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="tilefoundry")
    parser.add_argument(
        "--registry",
        metavar="PATH",
        help="override this installation's target registry",
    )
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
        choices=(*PAGES[1:], "orchestrator"),
        metavar="PAGE",
        help="which page; with none, the overview and the pages there are",
    )
    tutorial.add_argument(
        "family",
        nargs="?",
        metavar="FAMILY",
        help="which orchestrator family to show",
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
        epilog=check_guidance(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_check_arguments(check)

    analyze = commands.add_parser(
        "analyze",
        help=_COMMANDS["analyze"],
        epilog=analyze_guidance(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_source_argument(analyze)
    for analysis in _ANALYSES:
        analyze.add_argument(f"--{analysis}", action="store_true", help=EVIDENCE[analysis])
    analyze.add_argument(
        "--topology",
        metavar="LEVEL",
        help=(
            "topology level whose unit the per-unit figures describe; defaults "
            "to the module's coarsest declared level"
        ),
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
        help=_COMMANDS["schedule"],
        epilog=schedule_guidance(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    target = commands.add_parser("target", help=_COMMANDS["target"])
    target_commands = target.add_subparsers(dest="target_command", parser_class=_Parser)
    target_commands.add_parser("list", help=_TARGET_COMMANDS["list"])
    target_show = target_commands.add_parser("show", help=_TARGET_COMMANDS["show"])
    target_show.add_argument("identity", metavar="IDENTITY")
    target_add = target_commands.add_parser("add", help=_TARGET_COMMANDS["add"])
    target_add.add_argument(
        "--document",
        action="store_true",
        help="add a hardware document instead of a Target provider module",
    )
    target_add.add_argument("source", metavar="MODULE|PATH")
    target_remove = target_commands.add_parser("remove", help=_TARGET_COMMANDS["remove"])
    target_remove.add_argument("name", metavar="NAME")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        sys.stdout.write(overview())
        return 0
    try:
        registrations = load_registrations(registry_path(args.registry))
    except (OSError, TypeError, ValueError) as error:
        print(f"tilefoundry: error: {error}", file=sys.stderr)
        return 1
    for warning in registrations.warnings:
        print(f"tilefoundry: warning: {warning}", file=sys.stderr)
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
            return run_tutorial(args.page, args.family)
        except (OSError, ValueError) as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
    if args.command == "check":
        try:
            return run_check(args)
        except Exception as error:
            print(f"tilefoundry check: error: {error}", file=sys.stderr)
            return 1
    if args.command == "target":
        if args.target_command is None:
            sys.stdout.write(_target_overview())
            return 0
        try:
            if args.target_command == "list":
                return run_target_list(registrations)
            if args.target_command == "show":
                return run_target_show(args.identity)
            if args.target_command == "add":
                if args.document:
                    return run_target_add_document(args.source, registrations)
                return run_target_add_module(args.source, registrations)
            return run_target_remove(args.name, registrations)
        except Exception as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1
    if args.command == "schedule":
        try:
            return run_schedule(
                args.source,
                args.topology,
                as_json=args.json,
                dims=one_extent_per_dim(parse_dims(args.dim)),
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

    analyses = tuple(name for name in _ANALYSES if getattr(args, name.replace("-", "_")))
    if not analyses:
        analyses = _ANALYSES
    try:
        return run_authored_analysis(
            args.source,
            analyses,
            topology=args.topology,
            as_json=args.json,
            dims=one_extent_per_dim(parse_dims(args.dim)),
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
