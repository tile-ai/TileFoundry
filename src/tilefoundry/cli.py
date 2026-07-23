"""Command-line interface for authored TileFoundry HIR analysis."""

from __future__ import annotations

import argparse
import contextlib
import io
import runpy
import sys
from pathlib import Path
from typing import Sequence

from tilefoundry.analysis import AnalysisError, AnalysisOptions, analyze
from tilefoundry.inspection import PythonPrintOptions, as_script
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.target import CudaTarget, default_target
from tilefoundry.target.hardware import format_capabilities, load_hardware_spec

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


def _select_ir(namespace: dict[str, object], selector: str | None) -> Module | Function:
    if selector is not None:
        module_name, dot, function_name = selector.partition(".")
        selected = namespace.get(module_name)
        if selected is None:
            raise ValueError(f"selector {module_name!r} is not defined by the source")
        if not dot:
            if isinstance(selected, (Module, Function)):
                return selected
            raise TypeError(
                f"selector {module_name!r} resolves to {type(selected).__name__}, "
                "expected Module or Function"
            )
        if not isinstance(selected, Module):
            raise TypeError(
                f"selector root {module_name!r} is {type(selected).__name__}, expected Module"
            )
        function = selected.lookup(function_name)
        if not isinstance(function, Function):
            raise TypeError(
                f"selector {selector!r} resolves to {type(function).__name__}, "
                "expected HIR Function"
            )
        return function

    modules = _unique_values(namespace, Module)
    if len(modules) == 1:
        return modules[0]  # type: ignore[return-value]
    if len(modules) > 1:
        names = ", ".join(sorted(module.name for module in modules))
        raise ValueError(f"source defines multiple Modules ({names}); add ':Module'")
    functions = _unique_values(namespace, Function)
    if len(functions) == 1:
        return functions[0]  # type: ignore[return-value]
    if not functions:
        raise ValueError("source defines no TileFoundry Module or HIR Function")
    names = ", ".join(sorted(function.name for function in functions))
    raise ValueError(f"source defines multiple Functions ({names}); add ':Function'")


def load_authored_ir(source: str) -> Module | Function:
    """Execute one authored file and resolve its optional IR selector."""
    path, selector = _split_source(source)
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        namespace = runpy.run_path(str(path))
    return _select_ir(namespace, selector)


def _selected_target(ir: Module | Function):
    if isinstance(ir, Function):
        return ir.target or default_target()
    entry = ir.entry_function()
    if not isinstance(entry, Function):
        raise TypeError("capabilities requires a HIR Function entry")
    return entry.target or default_target()


def _grid_cta_count(ir: Module | Function) -> int | None:
    function = ir.entry_function() if isinstance(ir, Module) else ir
    if not isinstance(function, Function):
        return None
    counts = {
        topology.size
        for topology in function.topologies
        if topology.name == "cta" and isinstance(topology.size, int)
    }
    return next(iter(counts)) if len(counts) == 1 else None


def run_authored_analysis(source: str, analyses: tuple[str, ...]) -> int:
    """Load, type-infer, analyze, and print one authored HIR selection."""
    ir = load_authored_ir(source)
    selected = set(analyses)
    result = analyze(
        ir,
        options=AnalysisOptions(
            roofline="roofline" in selected,
            footprint="footprint" in selected,
            timeline="timeline" in selected,
        ),
    )
    summary = "\n".join(f"# {line}" for line in result.summary_lines)
    annotated = as_script(
        result.ir,
        options=PythonPrintOptions(
            show_types=True,
            comment_metadata_types=result.metadata_types,
        ),
    )
    sys.stdout.write(f"{summary}\n\n{annotated}")
    return 0


def _add_source_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", metavar="SOURCE", help="model.py[:Module[.function]]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tilefoundry")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="type-check and analyze authored HIR")
    _add_source_argument(analyze)
    for analysis in ("roofline", "footprint", "timeline"):
        analyze.add_argument(f"--{analysis}", action="store_true", help=f"print {analysis}")

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
                    load_hardware_spec(target),
                    grid_cta_count=_grid_cta_count(ir),
                )
                + "\n"
            )
            return 0
        except Exception as error:
            print(f"tilefoundry: error: {error}", file=sys.stderr)
            return 1

    analyses = tuple(
        name for name in ("roofline", "footprint", "timeline") if getattr(args, name)
    )
    if not analyses:
        analyses = ("roofline", "footprint", "timeline")
    try:
        return run_authored_analysis(args.source, analyses)
    except (AnalysisError, VerifyError, OSError, TypeError, ValueError) as error:
        print(f"tilefoundry: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
