"""Viewer colour palette.

A single source of truth for the builder, the ``/api/palette`` endpoint
and ``viewer.js::colorizeType``. The fixed *theme chrome* (paper / ink /
hairline) stays as named hex — it is the deliberate look. The *data*
palettes (DimVar / region-depth / storage) are generated from HSV via the
stdlib ``colorsys`` rather than hand-written hex tables, and storage is an
ordered pool bound by class order, not a name→hex map.
"""
from __future__ import annotations

import colorsys


def _hsv_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


# Fixed forest-paper theme chrome (the approved look — not a data palette).
INK = "#151a16"
MUTED = "#3a4640"
PAPER = "#fdfaf2"
BG = "#eef4ec"
HAIR = "#c8d2c6"

# DimVar is a single token-*class* colour: it marks "this is a dynamic
# dim", not which symbol (the detail panel carries the symbol identity).
DIMVAR_COLOR: str = _hsv_hex(0.77, 0.47, 0.43)  # plum

# Region nesting depth — an independent visual channel. Rotate the hue at
# low saturation for the cluster fill; the border reuses the same hue,
# darker / more saturated. Cycles for depth >= len(pool).
_DEPTH_HUES: tuple[float, ...] = (0.30, 0.555, 0.785, 0.125)  # green, blue, purple, amber
DEPTH_FILLS: tuple[str, ...] = tuple(_hsv_hex(h, 0.05, 0.95) for h in _DEPTH_HUES)
DEPTH_BORDERS: tuple[str, ...] = tuple(_hsv_hex(h, 0.18, 0.82) for h in _DEPTH_HUES)

# Storage classes — an ORDERED pool bound by class order (a list, not a
# name→hex map). The first 3 slots are pinned to the canonical classes
# (aliases fold onto them); the extra slots give an *unknown* storage
# level a stable, distinct colour (hashed into the pool) instead of muted.
STORAGE_CLASSES: tuple[str, ...] = ("gmem", "smem", "rmem")
_STORAGE_ALIASES: dict[str, str] = {"global": "gmem", "shared": "smem", "register": "rmem"}
_STORAGE_HSV: tuple[tuple[float, float, float], ...] = (
    (0.45, 0.66, 0.44),  # gmem — teal-green
    (0.39, 0.61, 0.31),  # smem — dark green
    (0.11, 0.95, 0.60),  # rmem — amber
    (0.58, 0.55, 0.50),  # spare — cyan
    (0.83, 0.45, 0.50),  # spare — magenta
    (0.20, 0.60, 0.45),  # spare — olive
)
STORAGE_POOL: tuple[str, ...] = tuple(_hsv_hex(*hsv) for hsv in _STORAGE_HSV)

# ExprKind / op-category title-cell fills (semantic, fixed).
EXPRKIND_COLORS: dict[str, str] = {
    "Function": "#2f6f63",
    "Call": "#3a5a6a",
    "Op": "#3a5a6a",
    "Var": "#a89a4e",
    "Constant": "#b27d2e",
    "Tuple": "#5a6f86",
}


def depth_fill(depth: int) -> str:
    """Low-saturation cluster fill for a region nesting depth."""
    return DEPTH_FILLS[depth % len(DEPTH_FILLS)]


def depth_border(depth: int) -> str:
    """Matching (darker) cluster border for a region nesting depth."""
    return DEPTH_BORDERS[depth % len(DEPTH_BORDERS)]


def stable_hash(s: str) -> int:
    """Deterministic 32-bit djb2 hash (NOT Python's salted ``hash``).
    Mirrored byte-for-byte in ``viewer.js`` so the detail panel colours
    unknown storage tokens identically to the graph."""
    h = 5381
    for ch in s:
        h = (h * 33 + ord(ch)) & 0xFFFFFFFF
    return h


def storage_color(name: str) -> str:
    """Colour for a storage-class token, from the ordered ``STORAGE_POOL``.
    Canonical classes (gmem / smem / rmem + aliases) pin to slots 0-2; any
    *other* memory level (e.g. ``hbm`` / ``pmem``) hashes stably into the
    spare slots so it is never left colourless and never collides with a
    known class's colour."""
    canon = _STORAGE_ALIASES.get(name, name)
    if canon in STORAGE_CLASSES:
        return STORAGE_POOL[STORAGE_CLASSES.index(canon)]
    spare = len(STORAGE_POOL) - len(STORAGE_CLASSES)
    if spare <= 0:
        return MUTED
    return STORAGE_POOL[len(STORAGE_CLASSES) + stable_hash(name) % spare]


def exprkind_color(kind: str) -> str:
    return EXPRKIND_COLORS.get(kind, "#3a5a6a")


def palette_pools() -> dict:
    """JSON-friendly dump for the ``/api/palette`` endpoint."""
    return {
        "ink": INK,
        "muted": MUTED,
        "paper": PAPER,
        "bg": BG,
        "hair": HAIR,
        "dimvar": DIMVAR_COLOR,
        "depth_fills": list(DEPTH_FILLS),
        "depth_borders": list(DEPTH_BORDERS),
        # Structured storage palette so the client can colour ANY storage
        # token (known or unknown) with the same rule as the graph.
        "storage_classes": list(STORAGE_CLASSES),
        "storage_aliases": dict(_STORAGE_ALIASES),
        "storage_pool": list(STORAGE_POOL),
        "exprkind": dict(EXPRKIND_COLORS),
    }


__all__ = [
    "INK", "MUTED", "PAPER", "BG", "HAIR",
    "DIMVAR_COLOR", "DEPTH_FILLS", "DEPTH_BORDERS",
    "STORAGE_CLASSES", "STORAGE_POOL", "EXPRKIND_COLORS",
    "storage_color", "exprkind_color",
    "depth_fill", "depth_border", "palette_pools",
]
