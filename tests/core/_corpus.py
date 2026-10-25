"""One fixture per case: load once, run each stage once, report problems together."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import TypeVar

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dump import DumpFlags, dump
from tilefoundry.ir.core.module import Module

C = TypeVar("C", bound="Case")


@dataclass(frozen=True)
class Case:
    path: Path
    name: str | None = None

    @property
    def id(self) -> str:
        shown = f"{self.path.parent.name}/{self.path.stem}"
        return f"{shown}.{self.name}" if self.name else shown

    def load(self) -> Module:
        """Load a fresh program through the CLI; lowering mutates its module."""
        return load_authored_ir(f"{self.path}:{self.name}" if self.name else str(self.path))

    def guard(self, state, problems: Problems) -> None:
        """A case's own checks; the base has none."""


def case(cls: type[C]) -> type[C]:
    """Make *cls* a frozen dataclass case whose fields' defaults are its expectations.

    Its ``path`` default names the fixture it owns; ``cases()`` uses it in place of the
    plain case for that file.
    """
    return dataclass(frozen=True)(cls)


def _owners(case_type: type[C]) -> tuple[type[C], ...]:
    """Subclasses of *case_type*, at any depth, whose ``path`` has a default."""
    owners = []
    for subclass in case_type.__subclasses__():
        if subclass.__dataclass_fields__["path"].default is not MISSING:
            owners.append(subclass)
        owners.extend(_owners(subclass))
    return tuple(owners)


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
    case_type: type[C] = Case,
    pattern: str = "*.py",
    expected: Mapping[str, Mapping[str, object]] = MappingProxyType({}),
) -> tuple[C, ...]:
    """One case per file; the ``@case`` subclass that owns a file replaces its plain case.

    A subclass of *case_type* owns the file its ``path`` default names. An ``expected`` row
    for a file that a subclass owns, a subclass whose file does not exist, and two subclasses
    owning one file are errors.
    """
    owners: dict[Path, type[C]] = {}
    for owner in _owners(case_type):
        path = owner.__dataclass_fields__["path"].default
        if not any(path.is_relative_to(root) for root in roots):
            continue
        if path in owners:
            raise ValueError(f"{owners[path].__name__} and {owner.__name__} both own {path}")
        owners[path] = owner
    found: dict[str, C] = {}
    for root in roots:
        for path in sorted(root.glob(pattern)):
            if path.name == "__init__.py" or not path.is_file():
                continue
            case = case_type(path=path)
            if case.id in found:
                raise ValueError(f"duplicate case id {case.id!r}: {found[case.id].path}, {path}")
            if path in owners and case.id in expected:
                raise ValueError(f"{owners[path].__name__} owns {case.id!r}; drop its expected row")
            found[case.id] = owners.pop(path)() if path in owners else case
    if owners:
        stale = ", ".join(f"{owner.__name__} ({path})" for path, owner in owners.items())
        raise ValueError(f"case classes own missing files: {stale}")
    missing = expected.keys() - found.keys()
    if missing:
        raise ValueError(f"expectations name missing cases: {', '.join(sorted(missing))}")
    return tuple(replace(found[key], **expected.get(key, {})) for key in sorted(found))


def special(case_type: type[C]) -> tuple[C, ...]:
    """Instances of every ``@case`` subclass of *case_type*, sorted by id.

    They are the table of a file whose tests each load one program and assert.
    """
    found: dict[str, C] = {}
    for owner in _owners(case_type):
        case = owner()
        if case.id in found:
            raise ValueError(
                f"{type(found[case.id]).__name__} and {owner.__name__} own {case.id!r}"
            )
        found[case.id] = case
    return tuple(found[key] for key in sorted(found))
