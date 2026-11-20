"""Give each element a recompute selector names its own definition, and cut the loops that call it.

An element whose choice differs from its definition's is differentiated from a
copy of its subtree. A ModuleList slot that a loop indexes, and whose elements
are no longer one definition, becomes one slot per run of equal elements, and
the loop one loop per run.

See [passes §7.4](docs/spec/passes.md#74-differentiate).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir import core, expr, types
from tilefoundry.ir.hir import tensor

__all__ = ["expand", "is_projection", "located", "projection", "refuse", "where"]


@dataclass(frozen=True)
class Run:
    """A run of consecutive elements of one slot, ``[start, stop)``, which become slot *key*."""

    start: int
    stop: int
    key: str


def expand(module: core.Module, by_element: dict, named_by: dict) -> core.Module:
    """*module* with each element an element selector names replaced by a copy of its subtree.

    Elements of one definition with the same choice share one copy. A slot
    that a loop indexes by its induction value, and whose elements differ, is
    cut into one slot per run of equal elements, and each such loop into one
    loop per run. The root keeps its name; every other definition is the same
    object.
    """
    if not by_element:
        return module
    taken = {each.name for each in core.module_definition_order(module)}
    copies: dict[tuple[int, object], core.Module] = {}
    children: dict[str, core.Module | tuple[core.Module, ...]] = {}
    functions = module.functions
    for key, slot in module.children.items():
        elements, classes = [], []
        for index, definition in enumerate(core.instances(slot)):
            choice = by_element.get((key, index if isinstance(slot, tuple) else None))
            kind = ("definition", id(definition)) if choice is None else (id(definition), choice)
            if choice is not None and kind not in copies:
                copies[kind] = copied(definition, taken)
            elements.append(definition if choice is None else copies[kind])
            classes.append(kind)
        if not isinstance(slot, tuple):
            children[key] = elements[0]
        elif len(set(classes)) == 1 or not indexed_at_run_time(module, key):
            children[key] = tuple(elements)
        else:
            selector = next(text for (slot_key, _), text in named_by.items() if slot_key == key)
            parts = runs(module, selector, key, classes)
            holding_const(selector, slot)
            children.update((part.key, tuple(elements[part.start : part.stop])) for part in parts)
            functions = tuple(cut_loops(function, selector, key, parts) for function in functions)
    return replace(module, functions=functions, children=children)


def runs(module: core.Module, selector: str, key: str, classes: list) -> tuple[Run, ...]:
    """Maximal runs of consecutive elements of one class, named ``<key>_<n>``."""
    names = {function.name for function in module.functions} | set(module.children) | set(module.methods)
    parts: list[Run] = []
    start = 0
    for stop in range(1, len(classes) + 1):
        if stop == len(classes) or classes[stop] != classes[start]:
            part = Run(start, stop, f"{key}_{len(parts)}")
            if part.key in names:
                refuse(selector, f"{part.key!r} is already a name in {module.name!r}")
            parts.append(part)
            start = stop
    return tuple(parts)


def cut_loops(function, selector: str, key: str, parts: tuple[Run, ...]):
    """*function* with each loop that indexes slot *key* by its induction value cut by *parts*."""
    if not isinstance(function, core.Function) or function.body is None:
        return function
    if not calls_slot(function, key):
        return function
    loops = slot_loops(function, selector, key, parts[-1].stop)
    kept = {id(param): param for param in function.params}
    return replace(function, body=LoopCutter(selector, key, parts, loops).visit(function.body, kept))


def slot_loops(function: core.Function, selector: str, key: str, size: int) -> dict[int, core.LoopRegion]:
    """The loops whose own body calls slot *key*, each checked against the *size* elements.

    A loop is indexed by its induction value, steps by 1, and runs a range that
    is not empty and inside the slot.
    ``collect_exprs`` lists what a loop holds before the loop, so the innermost
    loop around a value is the first to claim it.
    """
    owner: dict[int, core.LoopRegion] = {}
    for loop in core.collect_exprs(function.body):
        if isinstance(loop, core.LoopRegion):
            for root in (loop.body, *loop.yield_values):
                for inner in core.collect_exprs(root):
                    owner.setdefault(id(inner), loop)
    loops: dict[int, core.LoopRegion] = {}
    for call in core.collect_exprs(function.body):
        if not (isinstance(call, expr.Call) and isinstance(call.target, core.FunctionRef)):
            continue
        if call.target.path[:2] != (key, None):
            continue
        loop = owner.get(id(call))
        if loop is None or call.args[0] is not loop.induction_var:
            refuse(selector, f"slot {key!r} is indexed by a value that is not the loop index{located(call)}")
        if not all(isinstance(bound, int) for bound in (loop.start, loop.extent, loop.step)):
            refuse(selector, f"loop over {key!r}: trip count is not static{located(loop)}")
        if loop.step != 1:
            refuse(selector, f"loop over {key!r} has step {loop.step}{located(loop)}")
        if not 0 <= loop.start < loop.extent <= size:
            refuse(
                selector,
                f"loop over {key!r} runs [{loop.start}, {loop.extent}); the slot has {size} elements{located(loop)}",
            )
        loops[id(loop)] = loop
    return loops


class LoopCutter(core.BindingSubstitutionCloner):
    """Copy a body, cutting each loop of *loops* by *parts* and calling each part's slot."""

    def __init__(self, selector, key, parts, loops, part: Run | None = None, index=None) -> None:
        super().__init__()
        self.selector, self.key, self.parts, self.loops = selector, key, parts, loops
        self.part, self.index = part, index

    def visit_LoopRegion(self, region: core.LoopRegion, ctx) -> core.LoopRegion:
        if id(region) not in self.loops:
            return super().visit_LoopRegion(region, ctx)
        args = tuple(self.visit(arg, ctx) for arg in region.args)
        carried = len(region.yield_values)
        loop = None
        for part in self.parts:
            start, extent = max(region.start, part.start), min(region.extent, part.stop)
            if start >= extent:
                continue
            if loop is not None:
                results = (loop,) if carried == 1 else tuple(projection(loop, k, loop.type.fields[k]) for k in range(carried))
                args = (*results, *args[carried:])
            induction_var = replace(region.induction_var)
            params = tuple(replace(param) for param in region.params)
            inner = dict(ctx)
            inner[id(region.induction_var)] = induction_var
            inner.update((id(old), new) for old, new in zip(region.params, params))
            cloner = LoopCutter(self.selector, self.key, self.parts, self.loops, part, induction_var)
            loop = replace(
                region,
                start=start,
                extent=extent,
                induction_var=induction_var,
                params=params,
                args=args,
                body=cloner.visit(region.body, inner),
                yield_values=tuple(cloner.visit(value, inner) for value in region.yield_values),
            )
        return loop

    def visit_Call(self, node: expr.Call, ctx) -> expr.Expr:
        copied_call = super().visit_Call(node, ctx)
        target = node.target
        if not isinstance(target, core.FunctionRef) or target.path[:1] != (self.key,):
            return copied_call
        index = target.path[1]
        if index is None:
            offset = expr.Constant(type=self.index.type, value=self.part.start)
            relative = expr.Call(type=self.index.type, target=expr.DimSub(), args=(self.index, offset))
            path = (self.part.key, None, *target.path[2:])
            return replace(copied_call, target=replace(target, path=path), args=(relative, *copied_call.args[1:]))
        part = next(part for part in self.parts if part.start <= index < part.stop)
        path = (part.key, index - part.start, *target.path[2:])
        return replace(copied_call, target=replace(target, path=path))


def indexed_at_run_time(module: core.Module, key: str) -> bool:
    """Whether a function of *module* calls slot *key* by a run-time index."""
    return any(calls_slot(function, key, (key, None)) for function in module.functions)


def calls_slot(function, key: str, prefix: tuple | None = None) -> bool:
    """Whether *function* calls slot *key*, by a path that starts with *prefix* when given."""
    return isinstance(function, core.Function) and any(
        isinstance(call, expr.Call)
        and isinstance(call.target, core.FunctionRef)
        and call.target.path[: len(prefix or (key,))] == (prefix or (key,))
        for call in core.collect_exprs(function.body)
    )


def holding_const(selector: str, slot: tuple[core.Module, ...]) -> None:
    """Refuse a cut slot whose subtree has a ConstTensor parameter: its resource path would move."""
    for owner in core.module_definition_order(*slot):
        for function in owner.functions:
            if isinstance(function, core.Function) and any(param.is_const for param in function.params):
                refuse(selector, f"{owner.name!r} holds a ConstTensor; cutting its slot would move its resource path")


def projection(value: expr.Expr, index: int, type_: types.Type, name: str | None = None) -> expr.Call:
    """``value[index]``, bound to *name* when one is given; the final inference types it."""
    constant = expr.Constant(
        type=types.TensorType.from_scalar(expr.DType.i64, storage=expr.StorageKind.UMAT), value=index
    )
    metadata = () if name is None else (expr.BindingMetadata(name=name),)
    return expr.Call(tensor.TupleGetItem(), (value, constant), type=type_, metadata=metadata)


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


def is_projection(node: expr.Expr) -> bool:
    return isinstance(node, expr.Call) and isinstance(node.target, tensor.TupleGetItem)


def where(node: expr.Expr) -> str:
    """Where the author wrote *node*, and the name they bound it to."""
    span = expr.span_text(node, column=False)
    name = expr.binding_name(node)
    return " ".join(part for part in (span, f"({name})" if name else None) if part)


def located(node: expr.Expr) -> str:
    text = where(node)
    return f"\n  at {text}" if text else ""


def refuse(selector: str, reason: str):
    raise ValueError(f"recompute {selector!r}: {reason}")
