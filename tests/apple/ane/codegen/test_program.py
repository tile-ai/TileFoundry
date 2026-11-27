"""Descriptor construction and positional headers preserve immutable program values."""

from dataclasses import FrozenInstanceError

import pytest

from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.codegen import header, program


def test_header_replacement_preserves_dependency():
    words = {isa.fields.DEPENDENCY_ENABLE.addr: 1}
    td = isa.TD(tid=7, header=words, dependency=3)
    before = td.to_bytes()
    words[isa.fields.DEPENDENCY_ENABLE.addr] = 0
    updated = header.apply(isa.Program((td,))).tds[0]

    assert td.to_bytes() == before
    assert updated.tid == 7
    assert updated.dependency == 3
    assert isa.fields.DEPENDENCY_ENABLE.get(updated.header[isa.fields.DEPENDENCY_ENABLE.addr]) == 1
    assert updated.header[0x08] == 0x2A
    with pytest.raises(FrozenInstanceError):
        updated.tid = 8
    with pytest.raises(TypeError):
        updated.header[0x08] = 0


def test_stream_assigns_task_ids_and_boundary_masks():
    stream = program.to_program((({},), ({},), ({},)))
    assert [td.tid for td in stream.tds] == [0, 1, 2]
    assert [td.header[0x08] for td in stream.tds] == [0x08, 0, 0x22]
    assert stream.to_bytes().startswith(isa.PREAMBLE)
