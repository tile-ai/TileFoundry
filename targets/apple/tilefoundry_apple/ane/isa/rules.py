"""H14 ANE field formulas and the cross-field checks a built descriptor must pass.

Two kinds of knowledge live here, kept apart on purpose:

* **formulas** -- how a field's value is derived from the tiling and op choices.
  They are the reverse-engineered rules, each tagged with its confidence
  (``hardware`` verified on the ANE, ``disasm`` read from the compiler,
  ``empirical`` fitted to the measured TD corpus).  The evidence itself lives in
  ``notes/CONTROL_LEDGER.md`` and ``notes/l2_allocator_RED_crack.md``, not here.
* **checks** -- cross-field relations that must hold in any built register map.
  :func:`check` runs them over a ``{byte address: word}`` image, reading only the
  descriptor's own words, so a malformed descriptor is rejected at build time.
  :meth:`TD.validate` calls it for every descriptor it builds.

There is deliberately no build context object: a formula takes the few inputs it
needs, and a check reads the words a descriptor already carries.
"""
from __future__ import annotations

from collections.abc import Mapping

from . import fields
from ._arith import align_up

NE_CHANNEL_GROUP = 16
SHIFT_MAX = 4


# --- formulas ---------------------------------------------------------------

def ceil_log2(x: int) -> int:
    return (max(x, 1) - 1).bit_length()


def channel_groups(channels: int) -> int:  # disasm + byte-matched: the ne_ocg_size rule
    groups = -(-channels // NE_CHANNEL_GROUP)
    return 0 if groups <= 1 else (groups - 1).bit_length()


def active_nodes(engine_nodes: int) -> int:  # hardware + corpus: active_nodes == nodes - 1
    return engine_nodes - 1


def shift(work_unit_dim: int) -> int:  # disasm: clamp(ceil_log2(dim), 0, SHIFT_MAX)
    return min(ceil_log2(work_unit_dim), SHIFT_MAX)


def guard(
    pw: int, ch: int, channels: int, task_type: int, tile_h: int, height: int, ne_op: int
) -> int:
    """L2 staging guard band between rows (``row_stride - depth_stride``), in bytes.

    empirical: 93.6% over the 2468-TD corpus.  The first pad-off clause (bypass
    with a staged depth >= 2 KB) is disasm-pointed; the rest of the switch is
    fitted.  ``pw`` is the patch width (0x0030[3:0]), ``ch`` the L2 channel stride.
    """
    depth = channels * ch
    pad_off = (
        (ne_op == 4 and depth >= 2048)
        or ch == 256
        or (task_type == 3 and ch == 128)
        or (ch == 192 and channels <= 32)
        or (ch == 96 and channels >= 33)
        or channels <= 8
        or (channels <= 17 and tile_h == height)
    )
    if pad_off:
        return 0
    k = 0 if (ch // 16) % 2 else 16  # (ch // 16) parity
    return align_up(2 * pw + k, 16)


def row_stride(pw: int, ch: int, channels: int, task_type: int, tile_h: int, height: int, ne_op: int) -> int:
    """disasm: the L2 source row stride is the channel depth plus the guard band."""
    return channels * ch + guard(pw, ch, channels, task_type, tile_h, height, ne_op)


# --- checks over a built register map ----------------------------------------

class CheckError(ValueError):
    """A built descriptor violates a cross-field rule."""


def _read(image: Mapping[int, int], field: fields.BitField) -> int | None:
    """The field's value when its word is present in the image, else ``None``."""
    word = image.get(field.addr)
    return None if word is None else field.get(word)


def check(image: Mapping[int, int]) -> None:
    """Raise :class:`CheckError` on the first cross-field rule the map breaks.

    Only relations between words the descriptor actually carries are checked; an
    absent word skips the rule that needs it.  Per-field range is already enforced
    by :meth:`BitField.put`, so it is not repeated here.
    """
    common, l2 = fields.Common, fields.L2

    mn = _read(image, common.shift_minimum)
    pf = _read(image, common.shift_preferred)
    mx = _read(image, common.shift_maximum)
    if mn is not None and pf is not None and mx is not None:
        if not mn <= pf <= mx:
            raise CheckError(f"shift slices out of order: min={mn} pref={pf} max={mx}")
        if mx > SHIFT_MAX:
            raise CheckError(f"shift_maximum {mx} exceeds {SHIFT_MAX}")

    row = _read(image, l2.l2_src_row_stride)
    depth = _read(image, l2.l2_src_depth_stride)
    if row is not None and depth is not None and row < depth:
        raise CheckError(f"l2_src_row_stride {row} is below l2_src_depth_stride {depth}")
