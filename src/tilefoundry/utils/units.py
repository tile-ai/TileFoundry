"""Human-readable units shared across package layers."""

from __future__ import annotations

__all__ = ["format_bytes"]


def format_bytes(value: int) -> str:
    """Format a nonnegative byte count in its largest reached binary unit."""
    if value < 0:
        raise ValueError(f"byte count must be nonnegative, got {value}")
    if value == 0:
        return "0"

    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB")
    unit = units[0]
    for candidate in units[1:]:
        if amount < 1024:
            break
        amount /= 1024
        unit = candidate
    return f"{value}B" if unit == "B" else f"{amount:.2f}{unit}"
