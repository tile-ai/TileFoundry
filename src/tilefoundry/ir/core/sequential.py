"""The statement sequence."""

from __future__ import annotations

from dataclasses import dataclass

from .stmt import Stmt


@dataclass(unsafe_hash=True)
class Sequential(Stmt):
    """Wrap a ``tuple[Stmt, ...]`` as one TIR statement."""

    body: tuple[Stmt, ...]

    def __iter__(self):
        return iter(self.body)

    def __len__(self) -> int:
        return len(self.body)

    def __getitem__(self, idx):
        return self.body[idx]

__all__ = ['Sequential']
