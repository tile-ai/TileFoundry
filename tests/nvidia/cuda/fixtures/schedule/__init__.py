"""Authored schedule fixtures and their instruction-facing layouts.

Each module keeps layout constants beside the program that consumes them.
``_COMPUTE`` describes the executing warp or warpgroups; ``A_SMEM`` and
``B_SMEM`` describe instruction operand tiles; ``ACC`` describes the register
fragment held by those units. Grouped layout modes follow tensor axes,
while swizzled modes state the contiguous shared-memory run used by a transfer.
"""
