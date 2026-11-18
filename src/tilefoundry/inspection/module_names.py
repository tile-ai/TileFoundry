"""Names and order of the definitions a printed ``@module`` tree binds."""

from __future__ import annotations

from tilefoundry.ir import core


def definition_names(*named_roots):
    """A class name per definition: a ``(root, name)`` keeps its name, children take free ones."""
    names = {id(root): name for root, name in named_roots}
    taken = set(names.values())
    for node in core.module_definition_order(*(root for root, _ in named_roots), before_children=True):
        if id(node) in names:
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
    definitions = [names[id(element)] for element in core.instances(slot)]
    if not isinstance(slot, tuple):
        value = definitions[0]
    elif all(element is slot[0] for element in slot):
        value = f"ModuleList({definitions[0]}, {len(slot)})"
    else:
        value = f"ModuleList(({', '.join(definitions)},))"
    return f"{key} = {value}"


__all__ = ["definition_names", "slot_binding"]
