"""Names and order of the definitions a printed ``@module`` tree binds."""

from __future__ import annotations


def module_definition_order(root, *, before_children=False):
    """Definitions once, children before the slots that reference them."""
    seen = set()
    ordered = []

    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        if before_children:
            ordered.append(node)
        for slot in node.children.values():
            for child in slot.elements:
                visit(child)
        if not before_children:
            ordered.append(node)

    visit(root)
    return ordered


def definition_names(root, root_name):
    names = {id(root): root_name}
    taken = {root_name}
    for node in module_definition_order(root, before_children=True):
        if node is root:
            continue
        name = node.name
        suffix = 2
        while name in taken:
            name = f"{node.name}_{suffix}"
            suffix += 1
        taken.add(name)
        names[id(node)] = name
    return names


def slot_binding(key, slot, names):
    definitions = [names[id(element)] for element in slot.elements]
    if not slot.indexed:
        value = definitions[0]
    elif all(element is slot.elements[0] for element in slot.elements):
        value = f"ModuleList({definitions[0]}, {len(slot)})"
    else:
        value = f"ModuleList(({', '.join(definitions)},))"
    return f"{key} = {value}"


__all__ = ["definition_names", "module_definition_order", "slot_binding"]
