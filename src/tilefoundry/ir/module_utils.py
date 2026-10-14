"""Enumerate definition call paths and their effective execution contexts."""

from tilefoundry.visitor_registry.call_frame import CallFrame


def list_call_paths(root) -> dict[str, CallFrame]:
    """One frame per function, at its first DFS call path, without copying definitions."""
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
                found[f"{path}{function.name}"] = CallFrame(node, function, target, topologies)
        for key, slot in node.children.items():
            for index, child in enumerate(slot.elements):
                suffix = f"{key}.{index}." if slot.indexed else f"{key}."
                visit(child, f"{path}{suffix}", target, topologies)

    visit(root, "", root.target, root.effective_topologies())
    return found
