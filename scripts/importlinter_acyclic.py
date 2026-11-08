"""import-linter contract: the whole module import graph is acyclic.

docs/spec/code-organization.md C2. The built-in ``acyclic_siblings`` contract
misses a cycle between a package and its own submodules, so this contract runs
Tarjan's algorithm over every module of the root packages. Modules matching an
``ignore`` pattern are left out of the graph. Each strongly connected component
of more than one module is reported with one cycle through it.
"""

from __future__ import annotations

import fnmatch

from importlinter.application import output
from importlinter.domain import fields
from importlinter.domain.contract import Contract, ContractCheck


class AcyclicContract(Contract):
    """No import cycle among the modules of the root packages."""

    ignore = fields.ListField(subfield=fields.StringField(), required=False)

    def check(self, graph, verbose: bool) -> ContractCheck:
        """Find every strongly connected component with more than one module."""
        patterns = list(self.ignore or [])
        modules = sorted(m for m in graph.modules if not any(fnmatch.fnmatch(m, p) for p in patterns))
        kept = set(modules)
        edges = {m: sorted(graph.find_modules_directly_imported_by(m) & kept) for m in modules}
        components = strongly_connected(modules, edges)
        cycles = [cycle_through(component, edges) for component in components]
        return ContractCheck(kept=not components, metadata={"cycles": cycles, "sizes": [len(c) for c in components]})

    def render_broken_contract(self, check: ContractCheck) -> None:
        """Print one cycle per strongly connected component."""
        for size, cycle in zip(check.metadata["sizes"], check.metadata["cycles"], strict=True):
            output.print_error(f"{size} modules in one cycle; one path: {' -> '.join(cycle)}")
            output.new_line()


def strongly_connected(modules: list[str], edges: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan's components of more than one node, iteratively."""
    index, low, on_stack, stack, found = {}, {}, set(), [], []
    counter = 0
    for start in modules:
        if start in index:
            continue
        work = [(start, 0)]
        while work:
            node, position = work.pop()
            if position == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            successors = edges[node]
            if position < len(successors):
                work.append((node, position + 1))
                successor = successors[position]
                if successor not in index:
                    work.append((successor, 0))
                elif successor in on_stack:
                    low[node] = min(low[node], index[successor])
                continue
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    found.append(sorted(component))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return found


def cycle_through(component: list[str], edges: dict[str, list[str]]) -> list[str]:
    """One import cycle inside *component*, starting at its first module."""
    members = set(component)
    start = component[0]
    previous = {start: None}
    queue = [start]
    while queue:
        node = queue.pop(0)
        for successor in edges[node]:
            if successor == start:
                path = [node]
                while previous[path[-1]] is not None:
                    path.append(previous[path[-1]])
                return [*reversed(path), start]
            if successor in members and successor not in previous:
                previous[successor] = node
                queue.append(successor)
    return component
