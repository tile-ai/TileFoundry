"""A deliberate negative: a runtime implementation names no authored Module."""

from tilefoundry.runtime import RuntimeModule


class Handwritten(RuntimeModule):
    def __init__(self):
        super().__init__(name="handwritten")
