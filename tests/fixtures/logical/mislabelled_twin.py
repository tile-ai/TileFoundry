"""A deliberate negative: the runtime implementation labels a non-Module value."""

from tilefoundry.runtime import RuntimeModule


class Mislabelled(RuntimeModule):
    module = "not a Module"

    def __init__(self):
        super().__init__(name="mislabelled")
