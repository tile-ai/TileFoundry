"""Draw a tensor's logical cells, participant ownership, and local projection."""

from __future__ import annotations

from math import prod
from textwrap import wrap
from xml.etree.ElementTree import Element, SubElement, indent, tostring

from tilefoundry.ir.types.stride import crd2idx

COLORS = ("#4c92cf", "#e4ad50", "#62ad86", "#ae88cb")


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, (list, tuple)) else value


def _holds(participant, index):
    return any(begin <= index < end for begin, end in participant["runs"])


def render_svg(report, node_id, *, element=None, participant=None, title=None):
    """Render exact index ownership; numbers label indices, never tensor values.

    Large tensors use a labeled window, not sampled or aggregated cells. The
    participant miniatures show that same window. Local order comes from the
    exported logical map, independently of the physical address map.
    """
    node = report["nodes"][node_id]
    view = report["views"][node["view"]]
    members = view["participants"]
    participant = min(1, max(0, len(members) - 1)) if participant is None else participant
    selected = members[min(participant, len(members) - 1)] if members else None
    if participant < 0 or members and participant >= len(members):
        raise ValueError("participant index is outside the logical mesh")
    index = int(element if element is not None else selected["runs"][0][0] if selected else 0)
    if not 0 <= index < view["elements"]:
        raise ValueError("element index is outside the global tensor")
    cols = view["shape"][-1] if view["shape"] else 1
    rows = view["elements"] // cols
    row, col = divmod(index, cols)
    cell_width = max(28, 8 * len(str(view["elements"] - 1)) + 2)
    column_limit = min(16, 448 // cell_width)
    r0, c0 = row // 16 * 16, col // column_limit * column_limit
    nr, nc = min(16, rows - r0), min(column_limit, cols - c0)
    pending_axes = [i for i, attr in enumerate(view["attrs"]) if attr.startswith("Partial")]
    holders = [p for p in members if _holds(p, index)]
    contributions = len({tuple(p["coordinate"][i] for i in pending_axes) for p in holders})
    group = next((g for g in node.get("collective", {}).get("groups", ())
                  if selected and any(tuple(c) == tuple(selected["coordinate"]) for c in g)), ())
    tile_count = min(4, report["participants"] // report["replicas"], rows)

    def tile(global_index):
        return min(tile_count - 1, (global_index // cols) * tile_count // rows)

    svg = Element("svg", xmlns="http://www.w3.org/2000/svg", width="1120", height="840",
                  viewBox="0 0 1120 840", role="img", **{"aria-labelledby": "title description"})
    SubElement(svg, "title", id="title").text = title or f"{node['name']}: tensor ownership"
    SubElement(svg, "desc", id="description").text = (
        f"Global element {index} has {len(holders)} holders. "
        f"Pending reduction axes: {', '.join(view['pending']) or 'none'}. "
        "Colored numbered global cells map to logical mesh participants and a local tensor."
    )
    SubElement(svg, "style").text = (
        "text{font-family:Arial,Helvetica,sans-serif;fill:#243040;font-size:14px}"
        ".heading{font-size:23px;font-weight:600}.label{font-size:16px;font-weight:600}"
        ".small{font-size:12px;fill:#4c5968}.number{font-size:12px;font-family:monospace}"
    )
    defs = SubElement(svg, "defs")
    hatch = SubElement(defs, "pattern", id="pending", width="7", height="7", patternUnits="userSpaceOnUse")
    SubElement(hatch, "path", d="M-1,1 L1,-1 M0,7 L7,0 M6,8 L8,6", stroke="#243040", **{"stroke-opacity": ".2", "stroke-width": "1"})

    def rect(x, y, width, height, **attrs):
        return SubElement(svg, "rect", x=str(x), y=str(y), width=str(width), height=str(height),
                          **{key.replace("_", "-"): str(value) for key, value in attrs.items()})

    def text(x, y, value, cls=None, **attrs):
        item = SubElement(svg, "text", x=str(x), y=str(y), **({"class": cls} if cls else {}),
                          **{key.replace("_", "-"): str(value) for key, value in attrs.items()})
        item.text = str(value)
        return item

    def lines(x, y, value, width=62, cls="small", line_height=17):
        for i, line in enumerate(wrap(value, width=width)):
            text(x, y + i * line_height, line, cls)

    def grid(x, y, height, width, start_row, start_col, nrows, ncols, mapper, owner=None):
        dx, dy = width / ncols, height / nrows
        for c in range(ncols):
            text(x + (c + .5) * dx, y - 9, start_col + c, "small", text_anchor="middle")
        for r in range(nrows):
            text(x - 9, y + (r + .5) * dy + 4, start_row + r, "small", text_anchor="end")
            for c in range(ncols):
                global_index = mapper(r + start_row, c + start_col)
                owned = owner is None or _holds(owner, global_index)
                box = (x + c * dx, y + r * dy, dx, dy)
                rect(*box, fill=COLORS[tile(global_index)], fill_opacity=".5" if owned else ".13", stroke="white", stroke_width="1")
                if pending_axes and owned:
                    rect(*box, fill="url(#pending)")
                text(box[0] + dx / 2, box[1] + dy / 2 + 4, global_index, "number", text_anchor="middle")
                if global_index == index:
                    rect(box[0] + 1, box[1] + 1, dx - 2, dy - 2, fill="none", stroke="#243040", stroke_width="2")

    rect(0, 0, 1120, 840, fill="white")
    text(30, 35, title or f"{node['name']} · {node['op']}", "heading")
    mode = "HSDP" if report["replicas"] > 1 else "FSDP"
    text(30, 61, f"{mode} {report['replicas']} × {report['participants'] // report['replicas']} · {report['policy']} weights · batch {report['batch']} · logical participants")
    text(30, 87, "   |   ".join(f"{axis}: {attr}" for axis, attr in zip(view["axes"], view["attrs"], strict=True)) or "Plain layout: no declared holder mesh")
    text(52, 122, f"Global tensor · {' × '.join(map(str, view['shape'])) or 'scalar'}", "label")
    text(52, 142, f"Rows {r0}–{r0 + nr - 1}, columns {c0}–{c0 + nc - 1}" + (" · leading axes flattened" if len(view["shape"]) > 2 else ""), "small")
    grid(70, 171, nr * 25, nc * cell_width, r0, c0, nr, nc, lambda r, c: r * cols + c, selected)
    for i in range(tile_count):
        left = 52 + i * 120
        rect(left, 594, 13, 13, fill=COLORS[i], fill_opacity=".65")
        begin, end = (i * rows + tile_count - 1) // tile_count, ((i + 1) * rows + tile_count - 1) // tile_count - 1
        text(left + 19, 605, f"Rows {begin}–{end}", "small")
    lines(52, 634, "Numbers label global elements, not numerical values. Stronger fill marks the selected participant's ownership.", width=63)
    text(52, 684, f"Selected element {index} · global ({row}, {col})", "label")
    if pending_axes:
        lines(52, 711, f"{len(holders)} holders; {contributions} distinct pending contributions. Sum required on {', '.join(view['pending'])}. Hatching marks incomplete values.", cls=None, width=57, line_height=20)
    elif holders:
        lines(52, 711, f"{len(holders)} declared {'replicas' if len(holders) > 1 else 'copy'} of a complete value. Replica equality is checked by numerical evaluation.", cls=None, width=57, line_height=20)
    else:
        lines(52, 711, "This plain layout does not declare a participant mesh.", cls=None)

    text(614, 122, "Participant mesh", "label")
    mesh_cols = view["mesh"][-1] if view["mesh"] else 1
    mesh_rows = len(members) // mesh_cols if members else 0
    text(614, 142, "Rows: replicate axis · columns: shard axis" if len(view["mesh"]) > 1 else "Columns: shard axis", "small")
    cell_w, cell_h = min(110, 456 / mesh_cols), 78
    for p_index, p in enumerate(members):
        mr, mc = divmod(p_index, mesh_cols)
        x, y = 614 + mc * cell_w, 169 + mr * cell_h
        chosen = p is selected
        in_group = any(tuple(c) == tuple(p["coordinate"]) for c in group)
        attrs = dict(fill="white", stroke="#243040" if chosen else "#a3acb5", stroke_width=2 if chosen else 1)
        if in_group and not chosen:
            attrs["stroke_dasharray"] = "4 3"
        rect(x, y, cell_w - 8, cell_h - 8, **attrs)
        SubElement(svg, "circle", cx=str(x + 10), cy=str(y + 12), r="3", fill="#243040" if _holds(p, index) else "white", stroke="#243040")
        text(x + cell_w / 2 - 2, y + 17, str(tuple(p["coordinate"])), "small", text_anchor="middle")
        # Coalesce adjacent same-colored rows to keep checked-in SVGs compact.
        boxes = {}
        for r in range(nr):
            c = 0
            while c < nc:
                global_index = (r0 + r) * cols + c0 + c
                if not _holds(p, global_index):
                    c += 1
                    continue
                begin, color = c, tile(global_index)
                while c < nc and _holds(p, (r0 + r) * cols + c0 + c) and tile((r0 + r) * cols + c0 + c) == color:
                    c += 1
                key = (begin, c, color)
                blocks = boxes.setdefault(key, [])
                if blocks and blocks[-1][0] + blocks[-1][1] == r:
                    blocks[-1][1] += 1
                else:
                    blocks.append([r, 1])
        w, h = cell_w - 20, 40
        rect(x + 6, y + 25, w, h, fill="#f0f2f4")
        for (a, b, color), blocks in boxes.items():
            for r, length in blocks:
                box = (x + 6 + a * w / nc, y + 25 + r * h / nr, (b - a) * w / nc, length * h / nr)
                rect(*box, fill=COLORS[color], fill_opacity=".75")
                if pending_axes:
                    rect(*box, fill="url(#pending)")
        if _holds(p, index):
            rect(x + 6 + (col - c0) * w / nc, y + 25 + (row - r0) * h / nr, max(2, w / nc), max(2, h / nr), fill="#243040")
    below = 169 + mesh_rows * cell_h
    text(614, below + 13, "Filled dot: holds selected element · dashed frame: collective group", "small")
    text(614, below + 31, "Miniatures show the same global window as the numbered grid.", "small")
    if selected:
        mapping = selected["index_map"]
        indices = [selected["logical_offset"] + crd2idx(i, _tuples(mapping["shape"]), _tuples(mapping["strides"]))
                   for i in range(prod(selected["shape"]))]
        local_index = indices.index(index) if index in indices else None
        local_cols = selected["shape"][-1] if selected["shape"] else 1
        lr, lc = divmod(local_index or 0, local_cols)
        lr0, lc0 = lr // 4 * 4, lc // column_limit * column_limit
        lnr, lnc = min(4, len(indices) // local_cols - lr0), min(column_limit, local_cols - lc0)
        text(614, 556, f"Local tensor · participant {tuple(selected['coordinate'])}", "label")
        text(614, 578, f"Shape {' × '.join(map(str, selected['shape'])) or 'scalar'} · rows {lr0}–{lr0 + lnr - 1}, columns {lc0}–{lc0 + lnc - 1}", "small")
        grid(636, 607, lnr * 25, lnc * cell_width, lr0, lc0, lnr, lnc, lambda r, c: indices[r * local_cols + c])
        text(614, 735, f"Element {index} → local ({lr}, {lc})" if local_index is not None else f"Element {index} is not held by this participant.")
        text(614, 757, f"Logical base: {selected['logical_offset']} · storage base: {selected['physical_offset']} elements", "small")
    if node.get("collective") and selected:
        before = report["views"][report["nodes"][node["parents"][0]]["view"]]
        incoming = sum(_holds(p, index) and any(tuple(c) == tuple(p["coordinate"]) for c in group) for p in before["participants"])
        outgoing = sum(_holds(p, index) and any(tuple(c) == tuple(p["coordinate"]) for c in group) for p in members)
        text(30, 798, f"{node['collective']['kind'].replace('_', '-')} on {node['collective']['axis']}: holders of selected element in highlighted group: {incoming} before → {outgoing} after.")
    text(30, 823, "Derived from checked HIR. Diagrams show logical ownership and declared value state; they do not model a physical network schedule.", "small")
    indent(svg, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(svg, encoding="unicode") + "\n"
