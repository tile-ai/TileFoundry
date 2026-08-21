"""Turning a `SOURCE` argument into the IR a command runs over."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from typing import Mapping, Sequence

from tilefoundry.ir.core.module import Module, select
from tilefoundry.ir.hir.function import Function


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


def select_ir(namespace: dict[str, object], selector: str | None) -> Module:
    if selector is not None:


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


def _statement_codes(path: Path) -> tuple[list[tuple[ast.stmt, object]], bool]:
    """Each top-level statement of *path*, compiled to run on its own.

    The whole file is compiled first and thrown away, so every rule Python
    applies to a module still applies: a misplaced `__future__` import is refused
    there. Executing less of a file is what a selector asks for; accepting more of
    the language is not. Those imports govern the statements after them, so the
    leading run of them prefixes every statement -- alone, a statement would not
    have them -- and the compile does not inherit this caller's flags, because
    what a file postpones is the file's own decision.
    """
    text = path.read_text(encoding="utf-8")
    compile(text, str(path), "exec", dont_inherit=True)
    tree = ast.parse(text, filename=str(path))
    future: list[ast.stmt] = []
    for item in tree.body:
        if isinstance(item, ast.ImportFrom) and item.module == "__future__":
            future.append(item)
            continue
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            continue
        break
    documented = (
        tree.body[0]
        if tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
        else None
    )
    codes = []
    for statement in tree.body:
        if statement in future:
            continue
        prefix = () if statement is documented else future
        unit = ast.Module(body=[*prefix, statement], type_ignores=[])
        codes.append((statement, compile(unit, str(path), "exec", dont_inherit=True)))
    postponed = any(
        alias.name == "annotations" for item in future for alias in item.names
    )
    return codes, postponed


def _at_module_level(statement: ast.stmt, *, postponed: bool = False) -> set[str]:
    """The names *statement* reads while the file runs.

    Reads, not spellings: a name being written is not a reading of it. A
    comprehension's variables are its own -- only its first iterable is evaluated
    where it is written, every later one inside, where all the targets belong to
    it. A function body runs when something calls it, so its names are not read
    here, while what decorates it and defaults its arguments is. Whether its
    annotations are is the file's own decision, which *postponed* carries.
    """
    found: set[str] = set()
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
    comprehensions = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def evaluated_around(node: ast.AST) -> list[ast.AST]:
        """The parts of a function definition the file evaluates as it passes it."""
        parts: list[ast.AST] = list(getattr(node, "decorator_list", ()))
        parts.extend(node.args.defaults or ())
        parts.extend(item for item in (node.args.kw_defaults or ()) if item is not None)
        if postponed:
            return parts
        if getattr(node, "returns", None) is not None:
            parts.append(node.returns)
        parts.extend(
            argument.annotation
            for argument in ast.walk(node.args)
            if isinstance(argument, ast.arg) and argument.annotation is not None
        )
        return parts

    def walk(node: ast.AST, bound: frozenset[str]) -> None:
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in bound:
                found.add(node.id)
            return
        if isinstance(node, scopes):
            for part in evaluated_around(node):
                walk(part, bound)
            return
        if isinstance(node, comprehensions):
            inner = bound | {
                item.id
                for generator in node.generators
                for item in ast.walk(generator.target)
                if isinstance(item, ast.Name)
            }
            for index, generator in enumerate(node.generators):
                walk(generator.iter, bound if index == 0 else inner)
                for condition in generator.ifs:
                    walk(condition, inner)
            for part in (
                getattr(node, "elt", None),
                getattr(node, "key", None),
                getattr(node, "value", None),
            ):
                if part is not None:
                    walk(part, inner)
            return
        if postponed and isinstance(node, ast.AnnAssign):
            for part in (node.target, node.value):
                if part is not None:
                    walk(part, bound)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, bound)

    walk(statement, frozenset())
    return found


def _binds(statement: ast.stmt, name: str) -> bool:
    """Whether *statement* is what gives *name* its value."""
    if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return statement.name == name
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return any(
            (alias.asname or alias.name.partition(".")[0]) == name
            for alias in statement.names
        )
    targets: tuple = ()
    if isinstance(statement, ast.Assign):
        targets = tuple(statement.targets)
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = (statement.target,)
    return any(
        isinstance(item, ast.Name) and item.id == name
        for target in targets
        for item in ast.walk(target)
    )


def _follows_from(error: BaseException, aside: list[ast.stmt]) -> bool:
    """Whether *error* is a consequence of a statement already set aside.

    A statement that was set aside bound nothing, so one reaching for what it
    would have bound fails for that reason rather than its own. Reporting the
    second failure would name the symptom: the file that could not import its
    sibling would be described as a file with an undefined name in it.
    """
    return isinstance(error, NameError) and any(
        _binds(statement, error.name or "") for statement in aside
    )


def _run_for_selection(path: Path, module: object, root: str) -> None:
    """Execute *path* for the one root *root* names, statement by statement.

    A class that refuses to be built refuses while the file runs, so naming one
    root makes the rest a different question: a statement that is not the
    selection's is set aside when it fails. One that is -- binding the root, or
    reading it while the file runs -- is building or reconfiguring what was asked
    for, so its failure is raised even when something failed earlier and even when
    the root is already bound. Nothing is dropped unexecuted; a missing root at the
    end means something it needed failed, and the first of those is the reason.
    """
    first: BaseException | None = None
    aside: list[ast.stmt] = []
    codes, postponed = _statement_codes(path)
    for statement, code in codes:
        try:
            exec(code, module.__dict__)  # noqa: S102 — the authored file
        except Exception as error:  # noqa: BLE001 — one unfinished statement
            mine = _binds(statement, root) or root in _at_module_level(
                statement, postponed=postponed
            )
            if mine and not _follows_from(error, aside):
                raise
            aside.append(statement)
            if first is None:
                first = error
    if root not in module.__dict__ and first is not None:
        raise first


def load_namespace(source: str) -> tuple[dict[str, object], str | None]:
    """Load one authored file, returning what it defined and its selector.

    Loading it is how an authored file produces anything at all, so its own
    output is captured: what the command prints is its answer, not the file's.
    """
    path, selector = _split_source(source)
    directory = str(path.parent)
    sibling_names = {
        child.stem for child in path.parent.glob("*.py")
    } | {
        child.name
        for child in path.parent.iterdir()
        if child.is_dir() and (child / "__init__.py").is_file()
    }
    previous_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name.partition(".")[0] in sibling_names
    }
    sys.path.insert(0, directory)
    try:
        for name in previous_modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load source module {path}")
        module = importlib.util.module_from_spec(spec)


        sys.modules[spec.name] = module
        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            try:
                if selector is None:
                    spec.loader.exec_module(module)
                else:
                    _run_for_selection(path, module, selector.split(".")[0])
            except ModuleNotFoundError as error:
                raise ValueError(
                    f"{path.name} imports {error.name!r}, which is not importable.\n"
                    f"  sys.path got: {directory} (the file's own directory)\n"
                    "  a sibling module must sit in that directory"
                ) from None
        return vars(module), selector
    finally:
        for name in tuple(sys.modules):
            if name.partition(".")[0] in sibling_names:
                del sys.modules[name]
        sys.modules.update(previous_modules)
        sys.path.remove(directory)


def load_authored_ir(source: str) -> Module:
    """Execute one authored file and resolve its optional IR selector.

    The result is always a Module. A bare Function is rejected rather than
    resolved: it declares neither the Target its numbers would be measured
    against nor the topology hierarchy they divide over.
    """
    namespace, selector = load_namespace(source)
    return select_ir(namespace, selector)


def entry_function(ir: Module | Function) -> Function:
    """Resolve the HIR Function a command runs its pipeline over.

    The same Module -> entry_function() convention as `selected_target`.
    """
    function = ir.entry_function() if isinstance(ir, Module) else ir
    if not isinstance(function, Function):
        raise TypeError(f"schedule requires a HIR Function entry, got {type(function).__name__}")
    return function


def selected_target(ir: Module):
    """The Target the selection declares.

    Schedule and Analyze read hardware facts off it, so an undeclared Target is
    an authoring error rather than a cue to pick one: the selection must name
    the device it was written for.
    """
    if not isinstance(ir, Module):
        raise TypeError(
            f"expected a Module selection, got {type(ir).__name__}. A Function "
            "carries no Target; select the Module that declares it."
        )
    entry = ir.entry_function()
    if not isinstance(entry, Function):
        raise TypeError("capabilities requires a HIR Function entry")
    return ir.resolve_target()


def parse_dims(stated: Sequence[str] | None) -> dict[str, tuple[int, ...]] | None:
    """``NAME=V[,V...]`` arguments as every extent each dimension was given.

    ``None`` when nothing was stated, which is not the same as an empty mapping:
    a caller who stated no size is asking about the program as authored, while an
    empty mapping is a caller who meant to choose sizes and named none.
    """
    if stated is None:
        return None
    dims: dict[str, tuple[int, ...]] = {}
    for entry in stated:
        name, _, values = entry.partition("=")
        if not name or not values:
            raise ValueError(f"--dim takes NAME=V[,V...], got {entry!r}")
        if name in dims:
            raise ValueError(f"--dim {name} was given twice; list its values as NAME=V,V")
        try:
            dims[name] = tuple(int(value) for value in values.split(","))
        except ValueError:
            raise ValueError(
                f"--dim {name}: every value must be an integer, got {values!r}"
            ) from None
    return dims


def one_extent_per_dim(
    dims: Mapping[str, tuple[int, ...]] | None,
) -> dict[str, int] | None:
    """The one-extent mapping Analyze and Schedule give their operations."""
    if dims is None:
        return None
    chosen: dict[str, int] = {}
    for name, extents in dims.items():
        if len(extents) != 1:
            raise ValueError(
                f"--dim {name} takes one EXTENT at a time; asking several EXTENTs "
                "together is for check"
            )
        chosen[name] = extents[0]
    return chosen


def suggested_extents(lo: int, hi: int) -> tuple[int, ...]:
    """A few extents inside a declared range, for the suggestion that follows it."""
    candidates = {lo, lo + 1, (lo + hi) // 2, hi - 1}
    return tuple(sorted(value for value in candidates if lo <= value < hi))


__all__ = [
    "entry_function",
    "load_authored_ir",
    "load_namespace",
    "one_extent_per_dim",
    "parse_dims",
    "select_ir",
    "selected_target",
    "suggested_extents",
]
