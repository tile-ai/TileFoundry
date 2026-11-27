"""Small integer-alignment helpers shared across the ANE backend."""


def align_up(value: int, alignment: int) -> int:
    """Round a byte size up to the next multiple of ``alignment``."""
    return (value + alignment - 1) // alignment * alignment


def align_pow2(value: int, exponent: int) -> int:
    """Round a value up to the next ``1 << exponent`` boundary."""
    mask = (1 << exponent) - 1
    return (value + mask) & ~mask


__all__ = ["align_pow2", "align_up"]
