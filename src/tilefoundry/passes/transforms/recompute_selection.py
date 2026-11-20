"""What ``differentiate`` recomputes: parse the selectors, copy each element they name.

A selector names a definition (``Layer:h``: each element of it) or one element
of a child slot of the root (``Stack.layers.1:h``). An element's own selectors
replace what the selectors of definitions choose in its subtree, so the element
is differentiated from a copy of its definition.

See [passes §7.4](docs/spec/passes.md#74-differentiate).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import tensor

__all__ = [
    "WHOLE",
    "Selection",
    "expand",
    "is_projection",
    "parse_selectors",
    "resolve",
    "select_value",
    "selectable",
    "where",
]

WHOLE = object()
"""The choice of a selector without ``:``: each value a selector may name, in the subtree."""

NOTHING: frozenset[tuple[str | None, str]] = frozenset()
"""The choice of ``:None``: the element saves what it computes."""


@dataclass(frozen=True)
class Selection:
    """What the selectors choose: by Module name, and by root slot element.

    A choice is :data:`WHOLE` or a set of ``(function name or None for the
    entry, variable)``. The key of an element is ``(slot key, index)``, the
    index ``None`` for a slot that is not a ModuleList.
    """

    by_definition: dict[str, object]
    by_instance: dict[tuple[str, int | None], object]


def parse_selectors(module: core.Module, recompute: tuple[str, ...]) -> Selection:
    """Check each selector against *module* and gather what it chooses."""
    by_definition: dict[str, object] = {}
    by_instance: dict[tuple[str, int | None], object] = {}
    for selector in recompute:
        key, choice = parse(module, selector)
        if isinstance(key, str):
            by_definition[key] = merged(by_definition.get(key), choice)
            continue
        if key in by_instance and NOTHING in (by_instance[key], choice):
            refuse(selector, "':None' and another selector name the same element")
        by_instance[key] = merged(by_instance.get(key), choice)
    return Selection(by_definition, by_instance)


def expand(module: core.Module, selection: Selection) -> core.Module:
    """*module* with each element an instance selector names replaced by a copy of its subtree.

    Elements of one definition with the same choice share one copy. The root
    keeps its name and its functions; every other definition is the same object.
    """
    if not selection.by_instance:
        return module
    taken = {each.name for each in core.module_definition_order(module)}
    copies: dict[tuple[int, object], core.Module] = {}
    children = {}
    for key, slot in module.children.items():
        elements = []
        for index, definition in enumerate(core.instances(slot)):
            choice = selection.by_instance.get((key, index if isinstance(slot, tuple) else None))
            if choice is None:
                elements.append(definition)
                continue
            kind = (id(definition), choice)
            if kind not in copies:
                copies[kind] = copied(definition, taken)
            elements.append(copies[kind])
        children[key] = tuple(elements) if isinstance(slot, tuple) else elements[0]
    return replace(module, children=children)


def resolve(tree: core.Module, selection: Selection) -> tuple[expr.Expr, ...]:
    """The values *selection* chooses in *tree*, which :func:`expand` returned."""
    replaced = {
        id(each)
        for key, index in selection.by_instance
        for each in core.module_definition_order(element(tree, key, index))
    }
    values: list[expr.Expr] = []
    for name, choice in selection.by_definition.items():
        for owner in core.module_definition_order(tree):
            if owner.name == name and id(owner) not in replaced:
                values.extend(chosen(owner, choice, replaced))
    for (key, index), choice in selection.by_instance.items():
        values.extend(chosen(element(tree, key, index), choice, frozenset()))
    return tuple(values)


def selectable(function: core.Function) -> tuple[expr.Expr, ...]:
    """Each value of *function* a selector may name: an Op Call outside every loop body."""
    looped = in_loops(function)
    return tuple(
        value
        for value in core.collect_exprs(function.body)
        if isinstance(value, expr.Call) and id(value) not in looped and not is_call_result(value)
    )


def select_value(module: core.Module, selector: str) -> expr.Expr:
    """Resolve ``Module[.function]:variable`` among the Modules reachable from *module*.

    The part before ``:`` is the selector of ``tilefoundry`` commands: a Module
    named by ``Module.name`` and, optionally, one of its functions; the entry
    when the function is left out. The variable is the name the author bound.
    """
    owner, colon, variable = selector.rpartition(":")
    if not colon or not owner or not variable:
        refuse(selector, "expected Module[.function]:variable")
    name, _, function_name = owner.partition(".")
    if "." in function_name:
        refuse(selector, "name a Module by its name, not by a path through child slots")
    return value_of(selector, named(module, selector, name), function_name or None, variable)


def parse(root: core.Module, selector: str) -> tuple[str | tuple[str, int | None], object]:
    """The key and the choice of one selector, checked against the tree of *root*."""
    owner, colon, variable = selector.rpartition(":")
    if not colon:
        owner, variable = selector, ""
    if not owner or (colon and not variable):
        refuse(selector, "expected Module[.function][:variable]")
    segments = owner.split(".")
    functions = {function.name for function in root.functions}
    if segments[0] == root.name and len(segments) > 1 and segments[1] not in functions:
        if segments[1] not in root.children:
            refuse(selector, f"{root.name!r} has no function or child slot named {segments[1]!r}")
        return instance(root, selector, segments[1], segments[2:], variable)
    if len(segments) > 2:
        if segments[0] == root.name:
            refuse(selector, "expected Module[.function][:variable]")
        refuse(selector, f"an instance path starts at the root Module {root.name!r}")
    if variable == "None":
        refuse(selector, "':None' names one element of a child slot of the root")
    definition = named(root, selector, segments[0])
    function_name = segments[1] if len(segments) > 1 else None
    if not variable:
        if function_name is not None:
            definition.lookup(function_name)
            refuse(selector, "name a variable after the function")
        return segments[0], WHOLE
    value_of(selector, definition, function_name, variable)
    return segments[0], frozenset({(function_name, variable)})


def instance(
    root: core.Module, selector: str, key: str, rest: list[str], variable: str
) -> tuple[tuple[str, int | None], object]:
    """The key and the choice of a selector that names an element of slot *key* of *root*."""
    slot = root.children[key]
    if isinstance(slot, tuple):
        if not rest or not rest[0].isdigit():
            refuse(selector, f"slot {key!r} is a ModuleList; name one element")
        index = int(rest[0])
        if index >= len(slot):
            refuse(selector, f"slot {key!r} has {len(slot)} elements")
        definition, rest = slot[index], rest[1:]
    else:
        if rest and rest[0].isdigit():
            refuse(selector, f"slot {key!r} is not a ModuleList")
        index, definition = None, slot
    if len(rest) > 1:
        refuse(selector, "an instance path has one child slot")
    function_name = rest[0] if rest else None
    if function_name is not None:
        definition.lookup(function_name)
    if variable == "None":
        if function_name is not None:
            refuse(selector, "':None' names one element of a child slot of the root")
        return (key, index), NOTHING
    if not variable:
        if function_name is not None:
            refuse(selector, "name a variable after the function")
        return (key, index), WHOLE
    value_of(selector, definition, function_name, variable)
    return (key, index), frozenset({(function_name, variable)})


def named(root: core.Module, selector: str, name: str) -> core.Module:
    """The one Module named *name* that *root* reaches."""
    found = {id(each): each for each in core.module_definition_order(root) if each.name == name}
    if not found:
        refuse(selector, f"no Module named {name!r} is reachable from {root.name!r}")
    if len(found) > 1:
        refuse(selector, f"{len(found)} different Modules are named {name!r}")
    (module,) = found.values()
    return module


def value_of(selector: str, module: core.Module, function_name: str | None, variable: str) -> expr.Expr:
    """The value *variable* of a function of *module*, checked to be one a selector may name."""
    function = module.lookup(function_name) if function_name else module.entry_function()
    if any(param.name == variable for param in function.params):
        refuse(selector, f"{variable!r} is a parameter; the backward reads a parameter, it never computes one")
    matches = {
        id(value): value for value in core.collect_exprs(function.body) if expr.binding_name(value) == variable
    }
    if not matches:
        refuse(selector, f"{function.name!r} binds no value named {variable!r}")
    if len(matches) > 1:
        places = ", ".join(where(value) or "?" for value in matches.values())
        refuse(selector, f"{len(matches)} values are named {variable!r}: {places}")
    (value,) = matches.values()
    if id(value) in in_loops(function):
        refuse(
            selector,
            f"{variable!r} is computed inside a loop; "
            "the first version saves every value of a loop body and recomputes none",
        )
    if is_call_result(value):
        produced = value.args[0] if is_projection(value) else value
        refuse(
            selector,
            f"{variable!r} is the result of calling {produced.target.qualified_name!r}; "
            "only a value an Op computes is recomputed",
        )
    if not isinstance(value, expr.Call):
        refuse(selector, f"{variable!r} is a {type(value).__name__}; only a value an Op computes is recomputed")
    return value


def chosen(module: core.Module, choice: object, skipped) -> list[expr.Expr]:
    """The values *choice* names in *module*; :data:`WHOLE` reaches its subtree but *skipped*."""
    if choice is not WHOLE:
        return [value_of(module.name, module, name, variable) for name, variable in sorted(choice, key=str)]
    return [
        value
        for owner in core.module_definition_order(module)
        if id(owner) not in skipped
        for function in owner.functions
        if isinstance(function, core.Function) and function.body is not None
        for value in selectable(function)
    ]


def copied(definition: core.Module, taken: set[str]) -> core.Module:
    """A copy of the subtree of *definition*: each Module once, named ``<name>_<k>``."""
    memo: dict[int, core.Module] = {}

    def copy(module: core.Module) -> core.Module:
        if id(module) not in memo:
            name = fresh(module.name, taken)
            children = {
                key: tuple(copy(child) for child in slot) if isinstance(slot, tuple) else copy(slot)
                for key, slot in module.children.items()
            }
            functions = tuple(cloned(function) for function in module.functions)
            memo[id(module)] = replace(module, name=name, functions=functions, children=children)
        return memo[id(module)]

    return copy(definition)


def cloned(function):
    """A copy of *function* with its own parameters and body; another kind is shared."""
    if not isinstance(function, core.Function) or function.body is None:
        return function
    params = tuple(replace(param) for param in function.params)
    bound = {id(old): new for old, new in zip(function.params, params)}
    return replace(function, params=params, body=core.BindingSubstitutionCloner().visit(function.body, bound))


def fresh(name: str, taken: set[str]) -> str:
    k = 1
    while f"{name}_{k}" in taken:
        k += 1
    taken.add(f"{name}_{k}")
    return f"{name}_{k}"


def element(tree: core.Module, key: str, index: int | None) -> core.Module:
    slot = tree.children[key]
    return slot if index is None else slot[index]


def merged(old: object | None, new: object) -> object:
    if old is None:
        return new
    if WHOLE in (old, new):
        return WHOLE
    return old | new


def in_loops(function: core.Function) -> set[int]:
    """The ids of the values inside a loop body or yield of *function*."""
    return {
        id(inner)
        for loop in core.collect_exprs(function.body)
        if isinstance(loop, core.LoopRegion)
        for root in (loop.body, *loop.yield_values)
        for inner in core.collect_exprs(root)
    }


def is_call_result(value: expr.Expr) -> bool:
    produced = value.args[0] if is_projection(value) else value
    return isinstance(produced, expr.Call) and isinstance(produced.target, core.FunctionRef)


def is_projection(node: expr.Expr) -> bool:
    return isinstance(node, expr.Call) and isinstance(node.target, tensor.TupleGetItem)


def where(node: expr.Expr) -> str:
    """Where the author wrote *node*, and the name they bound it to."""
    span = expr.span_text(node, column=False)
    name = expr.binding_name(node)
    return " ".join(part for part in (span, f"({name})" if name else None) if part)


def refuse(selector: str, reason: str):
    raise ValueError(f"recompute {selector!r}: {reason}")
