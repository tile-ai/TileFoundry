"""Enumerate the functions a Module tree declares and their effective execution contexts."""

from tilefoundry.ir import core

from .context import Context

__all__ = ["collect_functions"]


def collect_functions(root) -> dict[core.FunctionRef, Context]:
    """Every function declared in the Module tree, once, keyed by its root-relative reference."""
    found, modules, functions = {}, set(), set()

    def visit(node, path, target, topologies):
        if id(node) in modules:
            return
        modules.add(id(node))
        target = node.target if node.target is not None else target
        topologies = node.topologies if node.topologies is not None else topologies
        for function in node.functions:
            if id(function) not in functions:
                functions.add(id(function))
                found[core.FunctionRef(path, function.name)] = Context(node, function, target, topologies)
        for key, slot in node.children.items():
            for index, child in enumerate(slot.elements):
                visit(child, (*path, key, index) if slot.indexed else (*path, key), target, topologies)

    visit(root, (), root.target, root.effective_topologies())
    return found
