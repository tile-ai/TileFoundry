"""Parser diagnostics for authored tile-window slices."""

from __future__ import annotations

import pytest

from tilefoundry.dsl import *
from tilefoundry.parser import ParseError

N = DimVar("N", 1, 256)


def test_a_tile_window_cannot_be_used_as_an_explicit_bound() -> None:
    with pytest.raises(
        ParseError,
        match=r"tile loop variable is already a window.*x\[:, t, :\].*base = t \+ 0",
    ):

        @func
        def stage(x: Tensor[(1, 256, 64), "f32"]):
            out = x[:, 0:128, :]
            for t in tf.tile(128, 128):
                out = x[:, t : t + 128, :]
            return out


def test_a_tile_window_rejects_true_division() -> None:
    with pytest.raises(ParseError, match=r"tile windows only support ± c and \* c"):

        @func
        def stage(x: Tensor[(1, 256, 64), "f32"]):
            out = x[:, 0:64, :]
            for t in tf.tile(128, 128):
                out = x[:, t / 2, :]
            return out


def test_a_literal_index_rejects_true_division() -> None:
    with pytest.raises(ParseError, match=r"index arithmetic uses //"):

        @func
        def stage(x: Tensor[(1, 256, 64), "f32"]):
            return x[:, 8 / 2, :]


def test_a_symbolic_index_rejects_true_division() -> None:
    with pytest.raises(ParseError, match=r"index arithmetic uses //"):

        @func
        def stage(x: Tensor[(1, 256, 64), "f32"]):
            return x[:, N / 2, :]
