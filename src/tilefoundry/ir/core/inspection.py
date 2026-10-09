"""The interfaces that let IR values be printed without importing a printer."""

from __future__ import annotations

from abc import ABC, abstractmethod


class PrinterBase(ABC):
    """A printer: the one entry that writes a value as source text."""

    @abstractmethod
    def print(self, value, ctx=None, indent: str = "") -> str:
        """*value* as source text, recording what it needs in *ctx*."""


class Printable(ABC):
    """A value that writes itself, handing the values it holds to *printer*."""

    @abstractmethod
    def print(self, printer: PrinterBase, ctx=None) -> str:
        """This value as source text, written with *printer* in *ctx*."""


__all__ = ["Printable", "PrinterBase"]
