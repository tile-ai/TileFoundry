"""A deliberate negative: a runtime implementation names no authored Module."""

from tilefoundry import runtime


class Handwritten(runtime.Module):
    def __init__(self):
        super().__init__(name="handwritten")
