"""Check physical buffer readiness and reuse through every consumer's completion."""

from dataclasses import replace

import pytest

from tilefoundry import analysis


def test_deferred_reader_prevents_alias_reuse_until_its_completion():
    hidden = analysis.BufferLifetime(
        "hidden",
        "gpu0.scratch",
        0,
        512,
        analysis.BufferAccess("forward", 0, 1),
        (analysis.BufferAccess("weight_gradient", 8, 10),),
    )
    grad = analysis.BufferLifetime(
        "gradient", "gpu0.scratch", 256, 1024, analysis.BufferAccess("derivative", 4, 5)
    )
    (hazard,) = analysis.check_buffer_lifetimes((hidden, grad))
    assert hazard.kind == "overwrite_live_value"
    assert hazard.operations == ("weight_gradient", "derivative")
    assert not analysis.check_buffer_lifetimes((hidden, replace(grad, offset=512)))
    assert not analysis.check_buffer_lifetimes((hidden, replace(grad, allocation="gpu1.scratch")))
    assert not analysis.check_buffer_lifetimes(
        (hidden, replace(grad, producer=analysis.BufferAccess("reuse", 10, 11)))
    )


def test_readiness_and_remote_completion_are_different_constraints():
    packet = analysis.BufferLifetime(
        "packet0",
        "source.slot",
        0,
        64,
        analysis.BufferAccess("produce", 0, 3),
        (analysis.BufferAccess("peer_read", 2, 6),),
    )
    reuse = analysis.BufferLifetime(
        "packet1", "source.slot", 0, 64, analysis.BufferAccess("produce_next", 4, 5)
    )
    assert [h.kind for h in analysis.check_buffer_lifetimes((packet, reuse))] == [
        "read_before_ready",
        "overwrite_live_value",
    ]
    packet = replace(packet, consumers=(analysis.BufferAccess("peer_read", 3, 6),))
    assert len(analysis.check_buffer_lifetimes((packet, reuse))) == 1
    assert not analysis.check_buffer_lifetimes(
        (packet, replace(reuse, producer=analysis.BufferAccess("next", 6, 7)))
    )


def test_multiple_readers_extend_liveness_and_input_order_does_not_matter():
    value = analysis.BufferLifetime(
        "input",
        "slot",
        0,
        16,
        analysis.BufferAccess("available", 0, 0),
        (
            analysis.BufferAccess("input_gradient", 1, 2),
            analysis.BufferAccess("weight_gradient", 5, 7),
        ),
    )
    reuse = analysis.BufferLifetime(
        "next", "slot", 0, 16, analysis.BufferAccess("next_forward", 3, 4)
    )
    assert value.end == 7
    assert analysis.check_buffer_lifetimes((value, reuse)) == analysis.check_buffer_lifetimes(
        (reuse, value)
    )
    assert analysis.check_buffer_lifetimes((value, reuse))[0].operations == (
        "weight_gradient",
        "next_forward",
    )


def test_malformed_trace_is_not_reported_as_safe():
    value = analysis.BufferLifetime(
        "input", "slot", 0, 16, analysis.BufferAccess("available", 0, 0)
    )
    for bad in (
        replace(value, nbytes=0),
        replace(value, offset=-1),
        replace(value, offset=False),
        replace(value, nbytes=True),
        replace(value, producer=analysis.BufferAccess("bad", 0, float("nan"))),
    ):
        with pytest.raises(ValueError):
            analysis.check_buffer_lifetimes((bad,))
    with pytest.raises(ValueError, match="uniquely"):
        analysis.check_buffer_lifetimes((value, value))
