"""One fixture per case: load once, run each stage once, report problems together."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dump import DumpFlags, dump
from tilefoundry.ir.core.module import Module


@dataclass(frozen=True)
class Case:
    path: Path
    name: str | None = None
    hooks: tuple[Callable[[object, "Problems"], None], ...] = ()

    @property
    def id(self) -> str:
        shown = f"{self.path.parent.name}/{self.path.stem}"
        return f"{shown}.{self.name}" if self.name else shown

    def load(self) -> Module:
        """Load a fresh program through the CLI; lowering mutates its module."""
        return load_authored_ir(f"{self.path}:{self.name}" if self.name else str(self.path))


def reload(printed: str, name: str) -> Module:
    """Reload runtime printer output for roundtrip checks only."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.py"
        path.write_text(printed)
        return load_authored_ir(f"{path}:{name}")


class Problems:
    def __init__(self, case: Case) -> None:
        self.case = case
        self.problems: list[tuple[str, str, str]] = []

    def run(self, name: str, fn, *args, **kwargs):
        """Record exceptions and dependent skips, leaving pytest outcomes alone."""
        if any(arg is None for arg in args):
            self.problems.append((f"skipped: {name}", "dependency is None", ""))
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as error:
            self.problems.append((name, f"{type(error).__name__}: {error}", traceback.format_exc()))
            return None

    def expect(self, what: str, predicate: Callable[..., bool], *args) -> None:
        """Check a predicate unless one of its dependencies failed."""
        result = self.run(what, predicate, *args)
        if result is not None and not result:
            self.problems.append((what, "expectation failed", ""))

    def dump(self) -> None:
        """Write every problem in the active dump scope and raise one summary."""
        if not self.problems:
            return
        dump(
            "problems.txt",
            lambda: "\n\n".join(
                f"{name}: {message}\n{trace}" for name, message, trace in self.problems
            ),
            DumpFlags.ALL,
        )
        summary = "\n".join(
            f"{name}: {message.splitlines()[0]}" for name, message, _trace in self.problems
        )
        raise AssertionError(f"{self.case.id}\n{summary}")


def cases(
    *roots: Path,
    case_type: type[Case] = Case,
    pattern: str = "*.py",
    expected: Mapping[str, Mapping[str, object]] = MappingProxyType({}),
) -> tuple[Case, ...]:
    """One case per root file, unexecuted; apply expectations by id and refuse stale ids."""
    found: dict[str, Case] = {}
    for root in roots:
        for path in sorted(root.glob(pattern)):
            if path.name == "__init__.py" or not path.is_file():
                continue
            case = case_type(path=path)
            if case.id in found:
                raise ValueError(f"duplicate case id {case.id!r}: {found[case.id].path}, {path}")
            found[case.id] = case
    missing = expected.keys() - found.keys()
    if missing:
        raise ValueError(f"expectations name missing cases: {', '.join(sorted(missing))}")
    return tuple(replace(found[key], **expected.get(key, {})) for key in sorted(found))
