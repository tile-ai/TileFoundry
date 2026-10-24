"""One fixture per case: load once, run each stage once, report problems together."""

from __future__ import annotations

import importlib.util
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from tilefoundry.dump import DumpFlags, dump
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction


@dataclass(frozen=True)
class Case:
    path: Path
    corpus: Path
    hooks: tuple[Callable[[object, "Problems"], None], ...] = ()

    @property
    def id(self) -> str:
        return f"{self.corpus.name}/{self.path.stem}"

    @classmethod
    def expand(cls, path: Path, corpus: Path) -> tuple["Case", ...]:
        return (cls(path=path, corpus=corpus),)

    def load(self) -> Module | PrimFunction:
        """Import a fresh program; lowering mutates its module."""
        return load_program(self.path)


def import_program(path: Path):
    """Execute a fixture file and return its Python namespace module."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def load_program(path: Path) -> Module | PrimFunction:
    """Import the first IR binding without retaining a cross-test module cache."""
    loaded = import_program(path)
    return next(
        value for value in vars(loaded).values() if isinstance(value, (Module, PrimFunction))
    )


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
    """Expand each root's files, apply expectations by id, and refuse stale ids."""
    found: dict[str, Case] = {}
    for root in roots:
        for path in sorted(root.glob(pattern)):
            if path.name == "__init__.py" or not path.is_file():
                continue
            for case in case_type.expand(path, root):
                if case.id in found:
                    raise ValueError(
                        f"duplicate case id {case.id!r}: {found[case.id].path}, {path}"
                    )
                found[case.id] = case
    missing = expected.keys() - found.keys()
    if missing:
        raise ValueError(f"expectations name missing cases: {', '.join(sorted(missing))}")
    return tuple(replace(found[key], **expected.get(key, {})) for key in sorted(found))
