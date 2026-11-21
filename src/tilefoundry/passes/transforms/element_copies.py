"""Give each element a recompute selector names its own definition, and cut the loops that call it.

An element whose choice differs from its definition's is differentiated from a
copy of its subtree. A ModuleList slot that a loop indexes, and whose elements
are no longer one definition, becomes one slot per run of equal elements, and
the loop one loop per run.

See [passes §7.4](docs/spec/passes.md#74-differentiate).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NoReturn

from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import tensor

__all__ = ["expand", "refuse"]


@dataclass(frozen=True)
class Run:
    """A run of consecutive elements ``[start, stop)`` of one definition, which become *slot*."""

    start: int
    stop: int
    slot: str


def expand(module: core.Module, by_element: dict) -> tuple[core.Module, dict]:
    """*module* with each element *by_element* chooses for replaced by a copy of its subtree.

    Elements of one definition with the same choice share one copy. A slot
    that a loop indexes by its induction value, and whose elements differ, is
    cut into one slot per run of equal elements, and each such loop into one
    loop per run. The root keeps its name; every other definition is the same
    object. Returned beside the tree: the copy of each element, by its
    ``(slot key, index)``.
    """
    if not by_element:
        return module, {}
    taken = {each.name for each in core.module_definition_order(module)}
    copies: dict[tuple[int, object], core.Module] = {}
    placed: dict[tuple[str, int | None], core.Module] = {}
    children: dict[str, core.Module | tuple[core.Module, ...]] = {}
    functions = module.functions
    for key, slot in module.children.items():
        elements = []
        for index, definition in enumerate(core.instances(slot)):
            position = (key, index if isinstance(slot, tuple) else None)
            kind = (id(definition), by_element.get(position))
            if kind[1] is not None:
                if kind not in copies:
                    copies[kind] = copied(definition, taken)
                placed[position] = copies[kind]
            elements.append(placed.get(position, definition))
        if not isinstance(slot, tuple):
            children[key] = elements[0]
        elif len({id(element) for element in elements}) == 1 or not any(
            call.target.path[1:2] == (None,) for function in functions for call in slot_calls(function, key)
        ):
            children[key] = tuple(elements)
        else:
            label = f"{module.name}.{key}"
            cut = runs(module, label, key, elements)
            refuse_constants(label, slot)
            children.update((run.slot, tuple(elements[run.start : run.stop])) for run in cut)
            functions = tuple(cut_loops(function, label, key, cut) for function in functions)
    return replace(module, functions=functions, children=children), placed


def runs(module: core.Module, label: str, key: str, elements: list[core.Module]) -> tuple[Run, ...]:
    """Maximal runs of consecutive elements that are one definition, each named ``<key>_<n>``."""
    names = {function.name for function in module.functions} | set(module.children) | set(module.methods)
    found: list[Run] = []
    start = 0
    for stop in range(1, len(elements) + 1):
        if stop == len(elements) or elements[stop] is not elements[start]:
            run = Run(start, stop, f"{key}_{len(found)}")
            if run.slot in names:
                refuse(label, f"{run.slot!r} is already a name in {module.name!r}")
            found.append(run)
            start = stop
    return tuple(found)


def slot_calls(function, key: str) -> tuple[expr.Call, ...]:
    """Each call of *function* whose callee is in slot *key*."""
    if not isinstance(function, core.Function) or function.body is None:
        return ()
    return tuple(
        call
        for call in core.collect_exprs(function.body)
        if isinstance(call, expr.Call) and isinstance(call.target, core.FunctionRef) and call.target.path[:1] == (key,)
    )


def cut_loops(function, label: str, key: str, cut: tuple[Run, ...]):
    """*function* with each loop that indexes slot *key* by its induction value cut by *cut*."""
    calls = slot_calls(function, key)
    if not calls:
        return function
    loops = slot_loops(function, calls, label, key, cut[-1].stop)
    kept = {id(param): param for param in function.params}
    return replace(function, body=LoopCutter(key, cut, loops).visit(function.body, kept))


def slot_loops(function: core.Function, calls, label: str, key: str, size: int) -> dict[int, core.LoopRegion]:
    """The loops whose own body calls slot *key* by a run-time index, each checked against *size*.

    A loop is indexed by its induction value and is ``range(a, b)`` with
    constant ``0 <= a < b <= size``.
    """
    owner = core.innermost_loops(core.collect_exprs(function.body))
    loops: dict[int, core.LoopRegion] = {}
    for call in calls:
        if call.target.path[1:2] != (None,):
            continue
        loop = owner.get(id(call))
        if loop is None or call.args[0] is not loop.induction_var:
            refuse(label, f"slot {key!r} is indexed by a value that is not the loop index{expr.diagnostic_suffix(call)}")
        static = all(isinstance(bound, int) for bound in (loop.start, loop.extent, loop.step))
        if not (static and loop.step == 1 and 0 <= loop.start < loop.extent <= size):
            rule = f"must be range(a, b) with step 1 and 0 <= a < b <= {size}"
            refuse(label, f"loop over {key!r} {rule}{expr.diagnostic_suffix(loop)}")
        loops[id(loop)] = loop
    return loops


class LoopCutter(core.BindingSubstitutionCloner):
    """Copy a body, cutting each loop of *loops* by *cut* and calling each run's slot."""

    def __init__(self, key, cut, loops, run: Run | None = None, index=None) -> None:
        super().__init__()
        self.key, self.cut, self.loops = key, cut, loops
        self.run, self.index = run, index

    def visit_LoopRegion(self, region: core.LoopRegion, ctx) -> core.LoopRegion:
        if id(region) not in self.loops:
            return super().visit_LoopRegion(region, ctx)
        args = tuple(self.visit(arg, ctx) for arg in region.args)
        carried = len(region.yield_values)
        loop = None
        for run in self.cut:
            start, extent = max(region.start, run.start), min(region.extent, run.stop)
            if start >= extent:
                continue
            if loop is not None:
                results = (loop,) if carried == 1 else tensor.unpack(loop, loop.type.fields[:carried])
                args = (*results, *args[carried:])
            induction_var = replace(region.induction_var)
            params = tuple(replace(param) for param in region.params)
            inner = dict(ctx)
            inner[id(region.induction_var)] = induction_var
            inner.update((id(old), new) for old, new in zip(region.params, params))
            cutter = LoopCutter(self.key, self.cut, self.loops, run, induction_var)
            loop = replace(
                region,
                start=start,
                extent=extent,
                induction_var=induction_var,
                params=params,
                args=args,
                body=cutter.visit(region.body, inner),
                yield_values=tuple(cutter.visit(value, inner) for value in region.yield_values),
            )
        return loop

    def visit_Call(self, node: expr.Call, ctx) -> expr.Expr:
        copied_call = super().visit_Call(node, ctx)
        target = node.target
        if not isinstance(target, core.FunctionRef) or target.path[:1] != (self.key,):
            return copied_call
        index = target.path[1]
        if index is None:
            offset = expr.Constant(type=self.index.type, value=self.run.start)
            relative = expr.Call(type=self.index.type, target=expr.DimSub(), args=(self.index, offset))
            path = (self.run.slot, None, *target.path[2:])
            return replace(copied_call, target=replace(target, path=path), args=(relative, *copied_call.args[1:]))
        run = next(run for run in self.cut if run.start <= index < run.stop)
        path = (run.slot, index - run.start, *target.path[2:])
        return replace(copied_call, target=replace(target, path=path))


def refuse_constants(label: str, slot: tuple[core.Module, ...]) -> None:
    """Refuse to cut a slot whose subtree has a ConstTensor parameter: its resource path moves."""
    for owner in core.module_definition_order(*slot):
        for function in owner.functions:
            if isinstance(function, core.Function) and any(param.is_const for param in function.params):
                refuse(label, f"{owner.name!r} holds a ConstTensor; cutting its slot would move its resource path")


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


def refuse(selector: str, reason: str) -> NoReturn:
    raise ValueError(f"recompute {selector!r}: {reason}")
