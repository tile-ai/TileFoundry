"""Warp matrix copy patterns compared with compiled CuTe."""

from tilefoundry.ir.tir.cuda.memory.stmatrix import StMatrix

from ._cute_reference import (
    cute_reference,
    reference_threads,
    reference_tv,
    to_thread_layout,
    to_tv_layout,
)


def test_stmatrix_atom():
    op = StMatrix()
    reference = cute_reference()["stmatrix"]
    for role in op.roles:
        assert to_tv_layout(op, role) == reference_tv(reference[role])
    assert to_thread_layout(op) == reference_threads(reference["ThrID"])
