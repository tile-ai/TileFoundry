#!/usr/bin/env python3
"""Check the Python source rules of docs/spec/code-organization.md §0.

Rules checked here: S1, S2, S4, S5, S7, S8, S9, S11 and C3 to C7. The rule
configuration lives in ``[tool.code_organization]`` of ``pyproject.toml``.
Every module is parsed once; violations print as ``path:line: RULE message``.
``--package`` keeps violations located in, or importing into, the named
packages; ``--report`` prints counts per package and rule.
"""

from __future__ import annotations

import argparse
import ast
import collections
import fnmatch
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

    roots: tuple[Path, ...]
    exclude: tuple[str, ...]
    dynamic_import_modules: frozenset[str]
    value_bases: frozenset[str]
    composition_root: str
    registry_classes: frozenset[str]
    target_base: str
    core_node_packages: tuple[str, ...]


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
    return Config(
        roots=tuple(ROOT / root for root in table["roots"]),
        exclude=tuple(table.get("exclude", ())),
        dynamic_import_modules=frozenset(table.get("dynamic_import_modules", ())),
        value_bases=frozenset(table.get("value_bases", ())),
        composition_root=table["composition_root"],
        registry_classes=frozenset(table.get("registry_classes", ())),
        target_base=table["target_base"],
        core_node_packages=tuple(table.get("core_node_packages", ())),
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


def collect(config: Config) -> dict[str, ModuleInfo]:
    """Parse every module under the configured roots once."""
    raw = {}
    for root in config.roots:
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if any(fnmatch.fnmatch(relative, pattern) for pattern in config.exclude):
                continue
            parts = path.relative_to(root.parent).with_suffix("").parts
            is_package = parts[-1] == "__init__"
            name = ".".join(parts[:-1] if is_package else parts)
            text = path.read_text()
            raw[name] = (path, is_package, ast.parse(text, str(path)), text)
    modules = {}
    for name, (path, is_package, tree, text) in raw.items():
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
        modules[name] = ModuleInfo(
            name, path, is_package, tree, text, literal_exports(tree), tuple(edges), top_definitions(tree)
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


def check_module(info: ModuleInfo, config: Config) -> Iterator[Violation]:
    """Single-module rules: S1 S2 S4 S5 S7 C4 C5 C6."""
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
            if name in DYNAMIC_IMPORTS and info.name not in config.dynamic_import_modules:
                yield hit(node, "S5", f"dynamic import {name}() outside the configured plugin loading points")
            if name in {"setattr", "delattr", "object.__setattr__", "object.__delattr__"} and node.args:
                owner = owner_parameter(node, up)
                subject = node.args[0]
                if not (isinstance(subject, ast.Name) and owner is not None and subject.id == owner):
                    yield hit(node, "C5", f"{name}() on an object other than the method's own self/cls")
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
    """S1 S2 S5 S7 and C4 on one function definition."""
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
        yield from check_property(node, decorators, hit)


def check_property(node, decorators: list[str], hit) -> Iterator[Violation]:
    """S7: a property or setter that only forwards one field."""
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    if len(body) != 1:
        return
    statement = body[0]
    owner = first_parameter(node)
    if "property" in decorators and isinstance(statement, ast.Return):
        value = statement.value
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == owner:
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
    """C5: attribute writes go only through the method's own self/cls."""
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
    """C6: module-level containers are immutable."""
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
    """S4: a nested function or class must be a closure."""
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


def check_imports(modules: Mapping[str, ModuleInfo], internal: tuple[str, ...]) -> Iterator[Violation]:
    """C3: exports and imports of docs/spec/code-organization.md §0.4."""
    for info in modules.values():
        if info.exports is None:
            yield Violation(info.path, 1, "C3", "module has no literal __all__", info.name)
        if info.is_package:
            yield from check_init(info)
        namespaces = {}
        for node in ast.walk(info.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in internal:
                        yield Violation(info.path, node.lineno, "C3", f"'import {alias.name}'; use 'from ... import ...'", info.name, alias.name)
            elif isinstance(node, ast.ImportFrom):
                source = resolve_from(info.name, info.is_package, node)
                if source.split(".")[0] not in internal:
                    continue
                for alias in node.names:
                    message = import_problem(info, node, source, alias.name, modules)
                    target = f"{source}.{alias.name}" if f"{source}.{alias.name}" in modules else source
                    if message:
                        yield Violation(info.path, node.lineno, "C3", message, info.name, target)
                    elif target != source and modules[target].is_package:
                        namespaces[alias.asname or alias.name] = target
        for node in ast.walk(info.tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in namespaces
            ):
                package = modules[namespaces[node.value.id]]
                if package.exports is not None and node.attr not in package.exports:
                    yield Violation(
                        info.path, node.lineno, "C3",
                        f"{node.value.id}.{node.attr} is not in {package.name}.__all__", info.name, package.name,
                    )


def import_problem(info: ModuleInfo, node: ast.ImportFrom, source: str, name: str, modules) -> str | None:
    """Why ``from source import name`` breaks C3, or ``None``."""
    if name == "*":
        return f"star import from {source}"
    directory = info.directory
    submodule = f"{source}.{name}"
    if source not in modules:
        return None
    if source == directory or within(directory, source):
        if source == directory and info.is_package and node.level == 1 and submodule in modules:
            return None
        return f"imports its own or an ancestor package {source}"
    if parent_of(source) == directory:
        if node.level != 1:
            return f"same-directory import from {source} must be 'from .{source.rpartition('.')[2]} import'"
        exports = modules[source].exports
        if submodule in modules:
            if not modules[submodule].is_package:
                return f"imports module {submodule} of another directory; import from its package"
            if exports is not None and name not in exports:
                return f"{name} is not in {source}.__all__"
            return None
        if exports is not None and name not in exports:
            return f"{name} is not in {source}.__all__"
        return None
    if node.level != 0:
        return f"relative import from another directory ({source}); use an absolute import from the package"
    if not modules[source].is_package:
        return f"imports module {source} of another directory; import from its package"
    if submodule in modules and not modules[submodule].is_package:
        return f"imports module {submodule} of another directory; import from its package"
    exports = modules[source].exports
    if exports is not None and name not in exports:
        return f"{name} is not in {source}.__all__"
    return None


def check_init(info: ModuleInfo) -> Iterator[Violation]:
    """C3: an ``__init__.py`` only re-exports."""
    for index, statement in enumerate(info.tree.body):
        if index == 0 and isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, ast.ImportFrom) and (statement.module == "__future__" or statement.level == 1):
            continue
        if isinstance(statement, ast.Assign) and [ast.unparse(t) for t in statement.targets] == ["__all__"]:
            continue
        yield Violation(info.path, statement.lineno, "C3", "__init__.py holds a statement other than re-exports and __all__", info.name)


def registration_calls(info: ModuleInfo) -> Iterator[tuple[ast.Call, ast.expr]]:
    """Calls to ``register`` / ``register_*`` with their key argument."""
    for node in ast.walk(info.tree):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node).rpartition(".")[2]
        if name != "register" and not name.startswith("register_"):
            continue
        if node.args:
            yield node, node.args[0]


def check_registrations(modules, resolver: Resolver, classes: Classes, config: Config) -> Iterator[Violation]:
    """C7: the composition root checks every Registry; a key's owner holds its registration."""
    registry_classes = {resolver.canonical(name) for name in config.registry_classes}
    target_base = resolver.canonical(config.target_base)
    declared = {}
    for info in modules.values():
        for statement in info.tree.body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)) and isinstance(statement.value, ast.Call):
                if resolver.qualified(info.name, statement.value.func) in registry_classes:
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            declared[f"{info.name}.{target.id}"] = (info, statement.lineno)
    checked = set()
    root = modules.get(config.composition_root)
    if root is not None:
        for node in ast.walk(root.tree):
            if isinstance(node, ast.Call) and call_name(node).rpartition(".")[2] == "check_registries":
                for argument in node.args:
                    if isinstance(argument, ast.Dict):
                        checked |= {resolver.qualified(root.name, key) for key in argument.keys if key is not None}
    for name, (info, line) in declared.items():
        if name not in checked and within(info.name, config.composition_root.split(".")[0]):
            yield Violation(info.path, line, "C7", f"registry {name} is not checked by check_registries in {config.composition_root}", info.name)
    for info in modules.values():
        for call, key in registration_calls(info):
            owner = registration_owner(info, call, key, resolver, classes, target_base, config)
            if owner is not None:
                yield Violation(info.path, call.lineno, "C7", owner, info.name)


def registration_owner(info, call, key, resolver, classes, target_base, config) -> str | None:
    """Why a registration is not written where its key's owner lives, or ``None``."""
    elements = list(key.elts) if isinstance(key, ast.Tuple) else [key]
    elements += [k.value for k in call.keywords if k.arg == "target"]
    resolved = [resolver.qualified(info.name, element) for element in elements]
    keys = [name for name in resolved if name in classes.nodes]
    if not keys:
        return None
    for name in keys:
        if target_base in classes.ancestors(name):
            package = name.split(".")[0]
            if not within(info.name, package):
                return f"registration keyed by Target {name} belongs in package {package}"
            return None
    name = keys[0]
    module = parent_of(name)
    if any(within(module, package) for package in config.core_node_packages):
        callee = resolver.expression(info.name, call.func)
        registry_package = parent_of(callee[0]) if callee and callee[1] else (callee[0] if callee else "")
        if callee and not within(info.name, registry_package):
            return f"registration for core node {name} belongs in the registry's package {registry_package}"
        return None
    if module != info.name:
        return f"registration for {name} belongs in its defining module {module}"
    return None


def check_values(resolver: Resolver, classes: Classes, config: Config) -> Iterator[Violation]:
    """S8: subclasses of configured value bases are frozen dataclasses."""
    bases = {resolver.canonical(name) for name in config.value_bases}
    for key, (info, node) in classes.nodes.items():
        if not (bases & classes.ancestors(key)):
            continue
        frozen = any(
            isinstance(d, ast.Call)
            and decorator_name(d).rpartition(".")[2] == "dataclass"
            and any(k.arg == "frozen" and isinstance(k.value, ast.Constant) and k.value.value is True for k in d.keywords)
            for d in node.decorator_list
        )
        if not frozen:
            yield Violation(info.path, node.lineno, "S8", f"value class {node.name} is not @dataclass(frozen=True)", info.name)
        fields = classes.fields(key)
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef) and statement.name == "__post_init__":
                for call in ast.walk(statement):
                    if isinstance(call, ast.Call) and call_name(call) == "object.__setattr__" and len(call.args) >= 2:
                        field = call.args[1]
                        if not (isinstance(field, ast.Constant) and field.value in fields):
                            yield Violation(info.path, call.lineno, "S8", f"__post_init__ of {node.name} writes {ast.unparse(field)}, not one of its fields", info.name)


def check_constructors(modules) -> Iterator[Violation]:
    """S9: ``make_*`` returns a class defined in another module."""
    for info in modules.values():
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


def check_names(modules) -> Iterator[Violation]:
    """S11: names do not repeat their package's name."""
    for info in modules.values():
        package = info.directory.rpartition(".")[2]
        if not info.is_package and stutters(info.path.stem, package):
            yield Violation(info.path, 1, "S11", f"file {info.path.name} repeats its package name {package!r}", info.name)
        for node in info.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exported = info.exports is None or node.name in info.exports
                if exported and not is_private(node.name) and stutters(node.name, package):
                    yield Violation(info.path, node.lineno, "S11", f"{node.name} repeats its package name {package!r}", info.name)


def run(config: Config) -> list[Violation]:
    """All violations of the configured tree."""
    modules = collect(config)
    internal = tuple(sorted({root.name for root in config.roots}))
    resolver = Resolver(modules, internal)
    classes = Classes(modules, resolver)
    found = []
    for info in modules.values():
        found.extend(check_module(info, config))
    found.extend(check_imports(modules, internal))
    found.extend(check_registrations(modules, resolver, classes, config))
    found.extend(check_values(resolver, classes, config))
    found.extend(check_constructors(modules))
    found.extend(check_names(modules))
    return sorted(set(found), key=lambda v: (str(v.path), v.line, v.rule, v.message))


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
