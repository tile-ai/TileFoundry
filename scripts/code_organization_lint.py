#!/usr/bin/env python3
"""Check the Python source rules of docs/spec/code-organization.md §0.

Rules checked here: S1, S2, S4, S5, S7, S8, S9, S11, C3, C4, C5, C6, C10, C11.
Every rule applies to every Python file of the repository. The configuration in
``[tool.code_organization]`` is ``src`` (first-party import roots, as in ruff)
and ``per-file-ignores`` (path glob to rule codes, ``ALL`` for every rule).
Violations print as ``path:line: RULE message``; ``--package`` keeps those in or
into the named packages, and ``--report`` prints counts per package and rule.
"""

from __future__ import annotations

import argparse
import ast
import collections
import fnmatch
import subprocess
import symtable
import sys
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUTABLE_FACTORIES = frozenset(
    {"list", "dict", "set", "defaultdict", "OrderedDict", "deque", "Counter", "bytearray"}
)
DYNAMIC_IMPORTS = frozenset({"importlib.import_module", "import_module", "__import__"})


@dataclass(frozen=True)
class Config:
    """The ``[tool.code_organization]`` table."""

    src: tuple[Path, ...]
    ignores: tuple[tuple[str, frozenset[str]], ...]

    def ignored(self, path: Path, rule: str) -> bool:
        """Whether a per-file ignore covers *rule* in *path*."""
        relative = path.relative_to(ROOT).as_posix()
        return any(
            fnmatch.fnmatch(relative, pattern) and (rule in rules or "ALL" in rules)
            for pattern, rules in self.ignores
        )


@dataclass(frozen=True)
class ImportEdge:
    """One imported name with the module it resolves to."""

    line: int
    source: str
    name: str
    alias: str
    level: int
    target: str


@dataclass(frozen=True)
class ModuleInfo:
    """What one parse of a module tells every check."""

    name: str
    path: Path
    is_package: bool
    tree: ast.Module
    text: str
    exports: frozenset[str] | None
    imports: tuple[ImportEdge, ...]
    definitions: frozenset[str]
    first_party: bool

    @property
    def directory(self) -> str:
        """The package whose directory holds this file."""
        return self.name if self.is_package else parent_of(self.name)


@dataclass(frozen=True)
class Violation:
    """One rule violation."""

    path: Path
    line: int
    rule: str
    message: str
    module: str
    target: str = ""


def parent_of(name: str) -> str:
    """The dotted parent of *name*, or the empty string."""
    return name.rpartition(".")[0]


def within(name: str, package: str) -> bool:
    """Whether module *name* is *package* or lies below it."""
    return name == package or name.startswith(package + ".")


def is_private(name: str) -> bool:
    """Whether *name* has a leading underscore that C4 forbids."""
    return name.startswith("_") and name != "_" and not (name.startswith("__") and name.endswith("__"))


def load_config(pyproject: Path) -> Config:
    """Read ``[tool.code_organization]``."""
    table = tomllib.loads(pyproject.read_text())["tool"]["code_organization"]
    ignores = table.get("per-file-ignores", {})
    return Config(
        src=tuple(ROOT / root for root in table["src"]),
        ignores=tuple((pattern, frozenset(rules)) for pattern, rules in ignores.items()),
    )


def literal_exports(tree: ast.Module) -> frozenset[str] | None:
    """The literal ``__all__`` of *tree*, or ``None``."""
    for statement in tree.body:
        targets = statement.targets if isinstance(statement, ast.Assign) else []
        if isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            try:
                value = ast.literal_eval(statement.value)
            except ValueError:
                return None
            if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                return frozenset(value)
            return None
    return None


def imported_names(tree: ast.Module) -> frozenset[str]:
    """Names an ``__init__.py`` binds by its ``from`` imports, ``__future__`` excluded."""
    return frozenset(
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
        for alias in node.names
    )


def top_definitions(tree: ast.Module) -> frozenset[str]:
    """Names a module binds at top level, imports excluded."""
    names = set()
    for node in module_level(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return frozenset(names)


def module_level(body: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements executed at module level, entering compound statements."""
    for statement in body:
        yield statement
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody", "handlers"):
            inner = getattr(statement, field, None)
            if isinstance(inner, list):
                yield from module_level([s for s in inner if isinstance(s, ast.stmt)])
            for handler in inner if field == "handlers" and inner else ():
                yield from module_level(handler.body)


def resolve_from(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    """The absolute module an ``ImportFrom`` names."""
    if node.level == 0:
        return node.module or ""
    package = module if is_package else parent_of(module)
    parts = package.split(".")
    base = parts[: len(parts) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def python_files() -> list[Path]:
    """Every Python file of the repository, tracked or new, that git does not ignore."""
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted(ROOT / name for name in listed if (ROOT / name).is_file())


def module_name(path: Path, config: Config) -> tuple[str, bool]:
    """The dotted module name of *path*, and whether it lies in a first-party package."""
    for root in config.src:
        if root in path.parents:
            parts = path.relative_to(root).with_suffix("").parts
            if len(parts) > 1 and (root / parts[0] / "__init__.py").is_file():
                return ".".join(parts[:-1] if parts[-1] == "__init__" else parts), True
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts), False


def collect(config: Config) -> dict[str, ModuleInfo]:
    """Parse every Python file once and index it by module name."""
    raw = {}
    for path in python_files():
        if config.ignored(path, "ALL") and not any(root in path.parents for root in config.src):
            continue
        text = path.read_text()
        try:
            tree = ast.parse(text, str(path))
        except SyntaxError:
            continue
        name, first_party = module_name(path, config)
        raw[name] = (path, path.name == "__init__.py", tree, text, first_party)
    modules = {}
    for name, (path, is_package, tree, text, first_party) in raw.items():
        edges = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            source = resolve_from(name, is_package, node)
            for alias in node.names:
                submodule = f"{source}.{alias.name}"
                target = submodule if submodule in raw else source
                edges.append(
                    ImportEdge(node.lineno, source, alias.name, alias.asname or alias.name, node.level, target)
                )
        exports = literal_exports(tree)
        if is_package and exports is None:
            exports = imported_names(tree)
        modules[name] = ModuleInfo(
            name, path, is_package, tree, text, exports, tuple(edges), top_definitions(tree), first_party
        )
    return modules


class Resolver:
    """Resolve names written in a module to the module and name that define them."""

    def __init__(self, modules: Mapping[str, ModuleInfo], internal: tuple[str, ...]) -> None:
        self.modules = modules
        self.internal = internal
        self.bindings = {
            name: {edge.alias: edge for edge in info.imports if edge.name != "*"}
            for name, info in modules.items()
        }

    def definition(self, module: str, name: str, depth: int = 0) -> tuple[str, str] | None:
        """Follow re-exports of *name* in *module* to its defining module."""
        if depth > 20 or module not in self.modules:
            return None
        if name in self.modules[module].definitions:
            return module, name
        edge = self.bindings[module].get(name)
        if edge is None:
            return None
        if edge.target != edge.source:
            return edge.target, ""
        return self.definition(edge.source, edge.name, depth + 1)

    def expression(self, module: str, node: ast.expr) -> tuple[str, str] | None:
        """Resolve a ``Name`` or dotted ``Attribute`` to ``(module, name)``."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                node = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return None
        if isinstance(node, ast.Name):
            return self.definition(module, node.id)
        if isinstance(node, ast.Attribute):
            owner = self.expression(module, node.value)
            if owner is not None and owner[1] == "":
                return self.definition(owner[0], node.attr)
        return None

    def qualified(self, module: str, node: ast.expr) -> str | None:
        """Dotted defining name of *node*, or ``None``."""
        found = self.expression(module, node)
        if found is None:
            return None
        return f"{found[0]}.{found[1]}" if found[1] else found[0]

    def canonical(self, dotted: str) -> str:
        """Canonical defining name for a configured dotted class name."""
        module, _, name = dotted.rpartition(".")
        found = self.definition(module, name)
        return f"{found[0]}.{found[1]}" if found and found[1] else dotted


class Classes:
    """Class definitions of all modules and their resolved bases."""

    def __init__(self, modules: Mapping[str, ModuleInfo], resolver: Resolver) -> None:
        self.nodes = {}
        self.bases = {}
        for name, info in modules.items():
            for node in info.tree.body:
                if isinstance(node, ast.ClassDef):
                    key = f"{name}.{node.name}"
                    self.nodes[key] = (info, node)
                    self.bases[key] = [resolver.qualified(name, base) for base in node.bases]

    def ancestors(self, key: str) -> set[str]:
        """Every resolved ancestor of class *key*."""
        seen, stack = set(), list(self.bases.get(key, ()))
        while stack:
            base = stack.pop()
            if base and base not in seen:
                seen.add(base)
                stack.extend(self.bases.get(base, ()))
        return seen

    def fields(self, key: str) -> set[str]:
        """Annotated fields of class *key* and its resolved ancestors."""
        names = set()
        for owner in (key, *self.ancestors(key)):
            if owner in self.nodes:
                names.update(
                    s.target.id
                    for s in self.nodes[owner][1].body
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                )
        return names


def decorator_name(node: ast.expr) -> str:
    """The dotted name of a decorator, call arguments dropped."""
    if isinstance(node, ast.Call):
        node = node.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def call_name(node: ast.Call) -> str:
    """The dotted name of a call's callee."""
    return decorator_name(node.func)


def first_parameter(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The instance or class parameter of a method."""
    arguments = function.args.posonlyargs + function.args.args
    return arguments[0].arg if arguments else None


def parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Child-to-parent map of *tree*."""
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def owner_parameter(node: ast.AST, up: Mapping[ast.AST, ast.AST]) -> str | None:
    """``self`` / ``cls`` of the method enclosing *node*, through closures."""
    shadowed = set()
    current = up.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = current.args
            names = {a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
            if isinstance(up.get(current), ast.ClassDef) and not isinstance(current, ast.Lambda):
                decorators = {decorator_name(d) for d in current.decorator_list}
                if "staticmethod" in decorators:
                    return None
                first = first_parameter(current)
                return None if first in shadowed else first
            shadowed |= names
        current = up.get(current)
    return None


NAMESPACE_MUTATORS = frozenset({"update", "setdefault", "pop", "popitem", "clear", "__setitem__", "__delitem__"})
COPY_CALLS = frozenset({"copy", "copy.copy"})


def namespace_mutated(call: ast.Call) -> ast.expr | None:
    """The object whose ``vars(x)`` / ``x.__dict__`` *call* mutates, or ``None``."""
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr in NAMESPACE_MUTATORS):
        return None
    holder = func.value
    if isinstance(holder, ast.Call) and call_name(holder) == "vars" and len(holder.args) == 1:
        return holder.args[0]
    if isinstance(holder, ast.Attribute) and holder.attr == "__dict__":
        return holder.value
    return None


def owned(subject: ast.expr, node: ast.AST, up: Mapping[ast.AST, ast.AST]) -> bool:
    """Whether *subject* is the method's own self/cls, or a copy of it made in the same method."""
    owner = owner_parameter(node, up)
    if owner is None or not isinstance(subject, ast.Name):
        return False
    if subject.id == owner:
        return True
    function = up.get(node)
    while function is not None and not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        function = up.get(function)
    return function is not None and any(
        isinstance(statement, ast.Assign)
        and [ast.unparse(t) for t in statement.targets] == [subject.id]
        and isinstance(statement.value, ast.Call)
        and call_name(statement.value) in COPY_CALLS
        and [ast.unparse(a) for a in statement.value.args] == [owner]
        for statement in ast.walk(function)
    )


def check_module(info: ModuleInfo) -> Iterator[Violation]:
    """Single-module rules S1 S2 S4 S5 S7 C4 C5 C6, in one walk of the AST.

    S5: a custom ``metaclass=``, a class deriving from ``type``, ``__del__`` and any
    ``importlib.import_module`` / ``__import__`` call. C4: any binding, parameter,
    attribute, import alias or file name that starts with ``_``. C5: ``check_writes``,
    and ``setattr`` / ``object.__setattr__`` / ``vars(x)`` / ``x.__dict__`` changes on
    anything but the method's self/cls or a ``copy.copy(self)`` made in it. C6:
    ``global`` and ``check_container``. S1, S2, S7: ``check_function``. S4: ``check_nesting``.
    """
    up = parents(info.tree)

    def hit(node: ast.AST | int, rule: str, message: str) -> Violation:
        line = node if isinstance(node, int) else getattr(node, "lineno", 1)
        return Violation(info.path, line, rule, message, info.name)

    if is_private(info.path.stem):
        yield hit(1, "C4", f"file name {info.path.name!r} starts with '_'")
    yield from check_nesting(info, up, hit)
    for node in ast.walk(info.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from check_function(node, up, hit)
        elif isinstance(node, ast.ClassDef):
            if is_private(node.name):
                yield hit(node, "C4", f"class {node.name!r} starts with '_'")
            for keyword in node.keywords:
                if keyword.arg == "metaclass" and decorator_name(keyword.value) not in {"ABCMeta", "abc.ABCMeta"}:
                    yield hit(node, "S5", f"class {node.name!r} uses a custom metaclass")
            for base in node.bases:
                if decorator_name(base) == "type" or (isinstance(base, ast.Call) and call_name(base) == "type"):
                    yield hit(node, "S5", f"class {node.name!r} defines a metaclass")
        elif isinstance(node, ast.Call):
            name = call_name(node)
            if name in DYNAMIC_IMPORTS:
                yield hit(node, "S5", f"dynamic import {name}(); only a plugin loading point may import by name")
            if name in {"setattr", "delattr", "object.__setattr__", "object.__delattr__"} and node.args:
                owner = owner_parameter(node, up)
                subject = node.args[0]
                if not (isinstance(subject, ast.Name) and owner is not None and subject.id == owner):
                    yield hit(node, "C5", f"{name}() on an object other than the method's own self/cls")
            subject = namespace_mutated(node)
            if subject is not None and not owned(subject, node, up):
                yield hit(node, "C5", f"{ast.unparse(node.func)}() changes the attributes of an object other than the method's own self/cls")
        elif isinstance(node, ast.Global):
            yield hit(node, "C6", f"global statement for {', '.join(node.names)}")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            yield from check_writes(node, up, hit)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and is_private(node.id):
            yield hit(node, "C4", f"name {node.id!r} starts with '_'")
        elif isinstance(node, ast.arg) and is_private(node.arg):
            yield hit(node, "C4", f"parameter {node.arg!r} starts with '_'")
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and is_private(node.attr):
            yield hit(node, "C4", f"attribute {node.attr!r} starts with '_'")
        elif isinstance(node, ast.alias) and is_private(node.asname or node.name.rpartition(".")[2]):
            yield hit(up[node], "C4", f"import binds {(node.asname or node.name)!r}, which starts with '_'")
        elif isinstance(node, ast.ExceptHandler) and node.name and is_private(node.name):
            yield hit(node, "C4", f"name {node.name!r} starts with '_'")
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name and is_private(node.name):
            yield hit(node, "C4", f"name {node.name!r} starts with '_'")
    for statement in module_level(info.tree.body):
        yield from check_container(statement, hit)


def check_function(node: ast.FunctionDef | ast.AsyncFunctionDef, up, hit) -> Iterator[Violation]:
    """S1 S2 S5 S7 and C4 on one function definition.

    How: S1 flags ``@staticmethod``; S2 flags a ``@classmethod`` not named
    ``from_*`` or ``create``; S5 flags a method ``__del__``; C4 flags a name
    starting with ``_``; S7 is delegated to ``check_property``.
    """
    decorators = [decorator_name(d) for d in node.decorator_list]
    in_class = isinstance(up.get(node), ast.ClassDef)
    if is_private(node.name):
        kind = "method" if in_class else "function"
        yield hit(node, "C4", f"{kind} {node.name!r} starts with '_'")
    if "staticmethod" in decorators:
        yield hit(node, "S1", f"@staticmethod {node.name!r}; write a module function")
    if "classmethod" in decorators and not (node.name.startswith("from_") or node.name == "create"):
        yield hit(node, "S2", f"@classmethod {node.name!r} is not a named constructor (from_* or create)")
    if in_class and node.name == "__del__":
        yield hit(node, "S5", "__del__ finalizer")
    if in_class:
        yield from check_property(node, decorators, up, hit)


def check_property(node, decorators: list[str], up, hit) -> Iterator[Violation]:
    """S7 — a property does more than return or assign one field.

    How: a ``@property`` whose only statement is ``return self.<field>`` (a
    ``ClassVar`` is exempt), or a ``.setter`` whose only statement is
    ``self.<field> = <argument>``, is reported.
    """
    owner_class = up.get(node)
    class_variables = set()
    if isinstance(owner_class, ast.ClassDef):
        class_variables = {
            s.target.id
            for s in owner_class.body
            if isinstance(s, ast.AnnAssign)
            and isinstance(s.target, ast.Name)
            and "ClassVar" in ast.unparse(s.annotation)
        }
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    if len(body) != 1:
        return
    statement = body[0]
    owner = first_parameter(node)
    if "property" in decorators and isinstance(statement, ast.Return):
        value = statement.value
        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == owner
            and value.attr not in class_variables
        ):
            yield hit(node, "S7", f"property {node.name!r} only returns self.{value.attr}; use a field")
    if any(d.endswith(".setter") for d in decorators) and isinstance(statement, ast.Assign):
        arguments = node.args.args
        target, value = statement.targets[0], statement.value
        if (
            len(statement.targets) == 1
            and isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == owner
            and isinstance(value, ast.Name)
            and len(arguments) == 2
            and value.id == arguments[1].arg
        ):
            yield hit(node, "S7", f"setter {node.name!r} only assigns self.{target.attr}; use a field")


def written_object(target: ast.expr) -> ast.expr | None:
    """The object whose attribute *target* writes, or ``None`` for a plain name."""
    while isinstance(target, ast.Subscript):
        target = target.value
    if isinstance(target, ast.Attribute):
        return target.value
    return None


def check_writes(node, up, hit) -> Iterator[Violation]:
    """C5 — an attribute, or an element of one, is written only through the method's own self/cls.

    How: for every assignment, augmented assignment, annotated assignment and
    ``del``, the object left of the written attribute (subscripts stripped) must
    be the first parameter of the enclosing method, seen through closures.
    """
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.Delete):
        targets = node.targets
    else:
        targets = [node.target]
    expanded = []
    for target in targets:
        expanded.extend(t for t in ast.walk(target) if isinstance(t, (ast.Attribute, ast.Subscript)))
    owner = owner_parameter(node, up)
    reported = set()
    for target in expanded:
        if isinstance(getattr(target, "ctx", None), ast.Load):
            continue
        subject = written_object(target)
        if subject is None or id(target) in reported:
            continue
        if isinstance(subject, ast.Name) and owner is not None and subject.id == owner:
            continue
        reported.add(id(target))
        yield hit(node, "C5", f"writes {ast.unparse(target)} on an object other than the method's own self/cls")


def check_container(statement: ast.stmt, hit) -> Iterator[Violation]:
    """C6 — a module-level value is not a mutable container.

    How: a module-level assignment (``__all__`` exempt) whose value is a list,
    dict or set display or comprehension, or a call of ``list``, ``dict``,
    ``set``, ``defaultdict``, ``OrderedDict``, ``deque``, ``Counter`` or
    ``bytearray``, is reported.
    """
    if isinstance(statement, ast.Assign):
        targets, value = statement.targets, statement.value
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        targets, value = [statement.target], statement.value
    else:
        return
    if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
        return
    mutable = isinstance(value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp))
    if isinstance(value, ast.Call):
        mutable = call_name(value).rpartition(".")[2] in MUTABLE_FACTORIES
    if mutable:
        names = ", ".join(ast.unparse(t) for t in targets)
        yield hit(statement, "C6", f"module-level mutable container {names}")


def check_nesting(info: ModuleInfo, up, hit) -> Iterator[Violation]:
    """S4 — a nested function or class is a closure.

    How: ``symtable`` gives the free variables of each nested scope; a function
    or class defined inside a function that reads none of the enclosing
    functions' parameters or locals (the method's self/cls excluded) is reported.
    """
    try:
        table = symtable.symtable(info.text, str(info.path), "exec")
    except SyntaxError:
        return
    scopes = {}
    stack = [table]
    while stack:
        scope = stack.pop()
        scopes.setdefault((scope.get_name(), scope.get_lineno()), scope)
        stack.extend(scope.get_children())
    for node in ast.walk(info.tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        enclosing = enclosing_functions(node, up)
        if not enclosing or nearest_scope(node, up) is not enclosing[0]:
            continue
        scope = scopes.get((node.name, node.lineno))
        if scope is None and node.decorator_list:
            scope = scopes.get((node.name, node.decorator_list[0].lineno))
        if scope is None:
            continue
        outer = set()
        for function in enclosing:
            arguments = function.args
            outer |= {a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
            outer |= {a.arg for a in (arguments.vararg, arguments.kwarg) if a is not None}
            outer |= {
                n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
        method = enclosing[0]
        if isinstance(up.get(method), ast.ClassDef):
            outer.discard(first_parameter(method))
        if not (free_names(scope) & outer):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            yield hit(node, "S4", f"nested {kind} {node.name!r} is not a closure; move it to module level")


def enclosing_functions(node: ast.AST, up) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Functions enclosing *node*, outermost last."""
    found, current = [], up.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(current)
        current = up.get(current)
    return found


def nearest_scope(node: ast.AST, up) -> ast.AST | None:
    """The nearest function or class that encloses *node*."""
    current = up.get(node)
    while current is not None and not isinstance(
        current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        current = up.get(current)
    return current


def free_names(scope: symtable.SymbolTable) -> set[str]:
    """Free variables of *scope* and every scope nested in it."""
    names = {s.get_name() for s in scope.get_symbols() if s.is_free()}
    for child in scope.get_children():
        names |= free_names(child)
    return names


def facades(modules: Mapping[str, ModuleInfo], internal: tuple[str, ...]) -> frozenset[str]:
    """Packages whose ``__init__`` re-exports names from beyond its own directory.

    Such a package is a user entry point (``tilefoundry``, ``tilefoundry.dsl``);
    its own C3 and C11 findings carry a per-file ignore, and C11 lets any module
    import names from it.
    """
    found = set()
    for info in modules.values():
        if info.is_package and info.first_party:
            for node in info.tree.body:
                if isinstance(node, ast.ImportFrom) and node.module != "__future__":
                    source = resolve_from(info.name, True, node) if node.level else (node.module or "")
                    if source.split(".")[0] in internal and any(
                        import_problem(info.directory, True, node, source, alias.name, modules, frozenset())
                        for alias in node.names
                    ):
                        found.add(info.name)
    return frozenset(found)


def check_exports(info: ModuleInfo) -> Iterator[Violation]:
    """C3 — a module states its exports; an ``__init__.py`` only re-exports what it imports.

    How: a module other than ``__init__.py`` assigns a literal list or tuple of
    strings to ``__all__``. An ``__init__.py`` holds only its docstring and ``from``
    imports, so it writes no ``__all__``; a user entry point that needs one for
    ``import *`` carries a per-file ignore.
    """
    if not info.is_package:
        if literal_exports(info.tree) is None:
            yield Violation(info.path, 1, "C3", "module has no literal __all__", info.name)
        return
    for index, statement in enumerate(info.tree.body):
        if index == 0 and isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, ast.ImportFrom):
            continue
        yield Violation(info.path, statement.lineno, "C3", "__init__.py holds a statement other than its docstring and re-exports", info.name)


def check_import_forms(info: ModuleInfo, modules, internal, entry_points) -> Iterator[Violation]:
    """C11 import forms and C10 namespace rebinding, for one file.

    C11: no ``import a.b`` of a first-party module; each ``from source import name``
    passes ``import_problem``; each ``namespace.member`` read names a member of that
    module's ``__all__`` (dunders and modules with a module-level ``__getattr__``
    exempt). C10: no parameter, local, loop, ``with``/``except`` target, function or
    class definition, or module-level assignment takes an imported namespace's name.
    """
    namespaces = {}
    for node in ast.walk(info.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "." in alias.name and alias.name.split(".")[0] in internal:
                    yield Violation(info.path, node.lineno, "C11", f"'import {alias.name}'; use 'from ... import ...'", info.name, alias.name)
        elif isinstance(node, ast.ImportFrom):
            source = resolve_from(info.name, info.is_package, node) if node.level else (node.module or "")
            if source.split(".")[0] not in internal:
                continue
            for alias in node.names:
                message = import_problem(info.directory, info.is_package, node, source, alias.name, modules, entry_points)
                child = f"{source}.{alias.name}"
                target = child if child in modules else source
                if message:
                    yield Violation(info.path, node.lineno, "C11", message, info.name, target)
                    continue
                if child in modules:
                    namespaces[alias.asname or alias.name] = child
    up = parents(info.tree)
    for node in ast.walk(info.tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in namespaces
            and isinstance(node.ctx, ast.Load)
        ):
            package = modules[namespaces[node.value.id]]
            dunder = node.attr.startswith("__") and node.attr.endswith("__")
            dynamic = "__getattr__" in package.definitions or any(e.alias == "__getattr__" for e in package.imports)
            if package.exports is not None and node.attr not in package.exports and not dunder and not dynamic:
                yield Violation(info.path, node.lineno, "C11", f"{node.value.id}.{node.attr} is not in {package.name}.__all__", info.name, package.name)
        rebound = rebinding(node, up)
        if rebound in namespaces:
            yield Violation(info.path, getattr(node, "lineno", 1), "C10", f"{rebound!r} rebinds the namespace imported from {namespaces[rebound]}", info.name)


def rebinding(node: ast.AST, up) -> str | None:
    """The name *node* binds in a module or function scope, outside an import, if any."""
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return node.name
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None if isinstance(up.get(node), ast.ClassDef) else node.name
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        scope = up.get(node)
        while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Module)):
            scope = up.get(scope)
        return None if isinstance(scope, ast.ClassDef) else node.id
    return None


def import_problem(directory: str, is_package: bool, node: ast.ImportFrom, source: str, name: str, modules, entry_points) -> str | None:
    """Why ``from source import name`` breaks C11, or ``None``."""
    if source in entry_points:
        return None
    if name == "*":
        return f"star import from {source}"
    if source not in modules:
        return None
    child = modules.get(f"{source}.{name}")
    exports = modules[source].exports
    namespace = child is not None and (child.is_package or name in (exports or ()))
    leaf = source.rpartition(".")[2]
    if source == directory:
        if node.level == 1 and child is not None:
            return None
        return f"imports its own package {source}"
    if within(directory, source):
        if namespace:
            return None
        return f"imports {name} from its ancestor package {source}; import only subpackages or exported modules of an ancestor"
    if parent_of(source) == directory:
        if node.level != 1:
            return f"same-directory import from {source} must be 'from .{leaf} import'"
        if child is not None and not namespace:
            return f"imports module {child.name} of another directory; export it from {source} first"
        if child is None and exports is not None and name not in exports:
            return f"{name} is not in {source}.__all__"
        return None
    if node.level != 0:
        return f"relative import from another directory ({source}); use an absolute import"
    if child is None:
        return f"imports {name} from another directory; write 'from {parent_of(source)} import {leaf}' and use {leaf}.{name}"
    if not namespace:
        return f"imports module {child.name} of another directory; export it from {source} first"
    return None


def frozen_dataclass(node: ast.ClassDef) -> bool:
    """Whether *node* is decorated ``@dataclass(frozen=True)``."""
    return any(
        isinstance(d, ast.Call)
        and decorator_name(d).rpartition(".")[2] == "dataclass"
        and any(k.arg == "frozen" and isinstance(k.value, ast.Constant) and k.value.value is True for k in d.keywords)
        for d in node.decorator_list
    )


def check_values(classes: Classes) -> Iterator[Violation]:
    """S8 — a value class is a frozen dataclass whose ``__post_init__`` writes its own fields.

    A class with a frozen-dataclass ancestor that declares fields of its own (non-
    ``ClassVar`` annotations) must be ``@dataclass(frozen=True)``, or those are not
    fields. In a frozen dataclass's ``__post_init__``, each ``object.__setattr__(self,
    "<name>", ...)`` names an annotated field of the class or of an ancestor.
    """
    frozen = {key for key, (_, node) in classes.nodes.items() if frozen_dataclass(node)}
    for key, (info, node) in classes.nodes.items():
        own_fields = any(
            isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and "ClassVar" not in ast.unparse(s.annotation)
            for s in node.body
        )
        if own_fields and not frozen_dataclass(node) and frozen & classes.ancestors(key):
            yield Violation(info.path, node.lineno, "S8", f"value class {node.name} is not @dataclass(frozen=True)", info.name)
        if key not in frozen:
            continue
        fields = classes.fields(key)
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef) and statement.name == "__post_init__":
                for call in ast.walk(statement):
                    if isinstance(call, ast.Call) and call_name(call) == "object.__setattr__" and len(call.args) >= 2:
                        field = call.args[1]
                        if not (isinstance(field, ast.Constant) and field.value in fields):
                            yield Violation(info.path, call.lineno, "S8", f"__post_init__ of {node.name} writes {ast.unparse(field)}, not one of its fields", info.name)


def check_constructors(info: ModuleInfo) -> Iterator[Violation]:
    """S9 — a ``make_*`` function returns a class defined in another module.

    How: a module-level ``make_*`` function needs a return annotation, and the
    annotation may not name a class defined in the same module (that is a
    ``@classmethod from_*``).
    """
    local_classes = {n.name for n in info.tree.body if isinstance(n, ast.ClassDef)}
    for node in info.tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("make_"):
            continue
        if node.returns is None:
            yield Violation(info.path, node.lineno, "S9", f"{node.name} has no return annotation", info.name)
            continue
        returns = node.returns
        if isinstance(returns, ast.Constant) and isinstance(returns.value, str):
            returns = ast.parse(returns.value, mode="eval").body
        named = {n.id for n in ast.walk(returns) if isinstance(n, ast.Name)}
        if named & local_classes:
            yield Violation(info.path, node.lineno, "S9", f"{node.name} returns a class of its own module; write a @classmethod from_*", info.name)


def words(name: str) -> list[str]:
    """Lower-case words of a snake_case or CapWords name."""
    out, current = [], ""
    for index, char in enumerate(name):
        boundary = char.isupper() and current and (
            not current[-1].isupper() or (index + 1 < len(name) and name[index + 1].islower())
        )
        if char == "_" or boundary:
            if current:
                out.append(current.lower())
            current = "" if char == "_" else char
        else:
            current += char
    if current:
        out.append(current.lower())
    return out


def stutters(name: str, package: str) -> bool:
    """Whether *name* repeats *package* without being exactly it."""
    if name.lower() == package.lower():
        return False
    have, want = words(name), package.lower().split("_")
    return any(have[i : i + len(want)] == want for i in range(len(have) - len(want) + 1))


def check_names(info: ModuleInfo, modules) -> Iterator[Violation]:
    """S11 — a public name or package export does not repeat its package's name.

    ``words`` splits a CapWords or snake_case name into lower-case words; ``stutters``
    reports a name whose words contain the package name's words in order, unless the
    name is exactly the package name (``target.Target``). File names are not checked.
    """
    package = info.directory.rpartition(".")[2]
    for node in info.tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exported = info.exports is None or node.name in info.exports
            if exported and not is_private(node.name) and stutters(node.name, package):
                yield Violation(info.path, node.lineno, "S11", f"{node.name} repeats its package name {package!r}", info.name)
    if info.is_package and info.exports:
        own = info.name.rpartition(".")[2]
        for name in sorted(info.exports):
            if stutters(name, own) and f"{info.name}.{name}" not in modules:
                yield Violation(info.path, 1, "S11", f"{info.name} exports {name}, which repeats the package name {own!r}", info.name)


def run(config: Config) -> list[Violation]:
    """All violations of the repository, after the per-file ignores."""
    modules = collect(config)
    internal = tuple(sorted({info.name.split(".")[0] for info in modules.values() if info.first_party}))
    resolver = Resolver(modules, internal)
    classes = Classes(modules, resolver)
    entry_points = facades(modules, internal)
    found = list(check_values(classes))
    for info in modules.values():
        if config.ignored(info.path, "ALL"):
            continue
        found.extend(check_module(info))
        found.extend(check_exports(info))
        found.extend(check_import_forms(info, modules, internal, entry_points))
        found.extend(check_constructors(info))
        found.extend(check_names(info, modules))
    kept = {v for v in found if not config.ignored(v.path, v.rule)}
    return sorted(kept, key=lambda v: (str(v.path), v.line, v.rule, v.message))


def report(violations: Sequence[Violation]) -> str:
    """Counts per package and rule."""
    rules = sorted({v.rule for v in violations}, key=lambda r: (r[0], int(r[1:])))
    table = collections.defaultdict(collections.Counter)
    for violation in violations:
        package = violation.module if violation.path.name == "__init__.py" else parent_of(violation.module)
        table[package][violation.rule] += 1
    lines = ["| package | " + " | ".join(rules) + " | total |", "|---" * (len(rules) + 2) + "|"]
    totals = collections.Counter(v.rule for v in violations)
    for package in sorted(table):
        row = table[package]
        lines.append(f"| {package} | " + " | ".join(str(row[r]) for r in rules) + f" | {sum(row.values())} |")
    lines.append("| total | " + " | ".join(str(totals[r]) for r in rules) + f" | {len(violations)} |")
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    """Run the checks; exit 1 on any violation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", nargs="*", default=[], help="keep violations in or into these packages")
    parser.add_argument("--report", action="store_true", help="print counts per package and rule")
    options = parser.parse_args(argv)
    violations = run(load_config(ROOT / "pyproject.toml"))
    if options.package:
        violations = [
            v for v in violations
            if any(within(v.module, p) or (v.target and within(v.target, p)) for p in options.package)
        ]
    if options.report:
        print(report(violations))
    else:
        for violation in violations:
            print(f"{violation.path.relative_to(ROOT)}:{violation.line}: {violation.rule} {violation.message}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
