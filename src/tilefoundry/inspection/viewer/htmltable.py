"""Typed ``<TABLE>`` / ``<TR>`` / ``<TD>`` builder for DOT HTML labels.

The DOT format accepts an HTML-like label syntax (``label=<...>``) and
Graphviz Pango is strict about attribute order and nested ``<FONT>``
balance. Writing those tables as raw strings is what made the old
``viewer.py`` 2624 LOC of ``f'<TD ALIGN="LEFT" ...>'`` literals. This
module is the thin typed shell — exactly ``Table`` / ``Row`` / ``Cell``
+ a colored ``Span`` for inline ``<FONT COLOR>`` fragments (DimVar /
``@gmem``). Output goes straight into ``graphviz.Digraph.node(label=...)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@dataclass
class Span:
    """One inline run inside a cell. ``color`` (and ``face`` / ``size``)
    emit a ``<FONT>`` wrap so DimVar / storage tokens render colored
    without breaking the outer cell's font."""
    text: str
    color: str | None = None
    face: str | None = None
    size: int | None = None
    bold: bool = False

    def to_html(self) -> str:
        body = _html_escape(self.text)
        if self.bold:
            body = f"<B>{body}</B>"
        if self.color or self.face or self.size is not None:
            attrs = []
            if self.color:
                attrs.append(f'COLOR="{self.color}"')
            if self.face:
                attrs.append(f'FACE="{self.face}"')
            if self.size is not None:
                attrs.append(f'POINT-SIZE="{self.size}"')
            return f"<FONT {' '.join(attrs)}>{body}</FONT>"
        return body


@dataclass
class Cell:
    """A single ``<TD>``. ``spans`` (inline colored runs) take precedence
    over ``text``; supply spans for cells with mixed coloring (DimVar +
    plain shape numbers + ``@gmem`` storage etc.)."""
    text: str = ""
    spans: tuple[Span, ...] = ()
    port: str | None = None
    bgcolor: str | None = None
    color: str | None = None         # foreground (text) color, applied to plain ``text``
    font_face: str | None = None
    font_size: int | None = None
    bold: bool = False
    href: str | None = None
    title: str | None = None
    align: str = "LEFT"              # LEFT / CENTER / RIGHT
    colspan: int | None = None
    rowspan: int | None = None
    width: int | None = None
    cellpadding: int | None = None

    def to_html(self) -> str:
        attrs = [f'ALIGN="{self.align}"', 'BALIGN="LEFT"']
        if self.port is not None:
            attrs.append(f'PORT="{self.port}"')
        if self.bgcolor is not None:
            attrs.append(f'BGCOLOR="{self.bgcolor}"')
        if self.href is not None:
            attrs.append(f'HREF="{self.href}"')
        if self.title is not None:
            attrs.append(f'TITLE="{_html_escape(self.title)}"')
        if self.colspan is not None:
            attrs.append(f'COLSPAN="{self.colspan}"')
        if self.rowspan is not None:
            attrs.append(f'ROWSPAN="{self.rowspan}"')
        if self.width is not None:
            attrs.append(f'WIDTH="{self.width}"')
        if self.cellpadding is not None:
            attrs.append(f'CELLPADDING="{self.cellpadding}"')

        if self.spans:
            body = "".join(s.to_html() for s in self.spans)
        else:
            body = Span(
                text=self.text, color=self.color, face=self.font_face,
                size=self.font_size, bold=self.bold,
            ).to_html()

        return f"<TD {' '.join(attrs)}>{body}</TD>"


@dataclass
class Row:
    cells: list[Cell] = field(default_factory=list)

    def to_html(self) -> str:
        return "<TR>" + "".join(c.to_html() for c in self.cells) + "</TR>"


@dataclass
class Table:
    rows: list[Row] = field(default_factory=list)
    border: int = 0
    cellborder: int = 1
    cellspacing: int = 0
    cellpadding: int = 4
    bgcolor: str | None = None
    color: str | None = None

    def add_row(self, *cells: Cell) -> Row:
        row = Row(cells=list(cells))
        self.rows.append(row)
        return row

    def to_html(self) -> str:
        attrs = [
            f'BORDER="{self.border}"',
            f'CELLBORDER="{self.cellborder}"',
            f'CELLSPACING="{self.cellspacing}"',
            f'CELLPADDING="{self.cellpadding}"',
        ]
        if self.bgcolor is not None:
            attrs.append(f'BGCOLOR="{self.bgcolor}"')
        if self.color is not None:
            attrs.append(f'COLOR="{self.color}"')
        body = "".join(r.to_html() for r in self.rows)
        return f"<<TABLE {' '.join(attrs)}>{body}</TABLE>>"


__all__ = ["Cell", "Row", "Span", "Table"]
