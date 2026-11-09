"""The return statement."""

from __future__ import annotations

from dataclasses import dataclass

from .stmt import Stmt


@dataclass(unsafe_hash=True)
class Return(Stmt):
    """Empty return; tir functions have no value return."""

__all__ = ['Return']
