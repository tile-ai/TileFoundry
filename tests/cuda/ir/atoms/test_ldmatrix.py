"""Warp matrix copy patterns compared with compiled CuTe."""

from tilefoundry_cuda.ir.tir.memory.ldmatrix import LdMatrix

from ._cute_reference import (
    cute_reference,
    reference_threads,
    reference_tv,
    to_thread_layout,
    to_tv_layout,
)


def test_ldmatrix_atom():
    op = LdMatrix()
    reference = cute_reference()["ldmatrix"]
    for role in op.roles:
        assert to_tv_layout(op, role) == reference_tv(reference[role])
    assert to_thread_layout(op) == reference_threads(reference["ThrID"])
