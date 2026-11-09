"""The runtime reference for the sibling CPU square program."""

from model import Mine

from tilefoundry import runtime


@runtime.module(Mine)
class MineTwin:
    @runtime.func
    def main(self, x):
        return x * x
