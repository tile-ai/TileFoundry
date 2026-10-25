from provider import ReloadTarget

from tilefoundry.dsl import *


@module(target=ReloadTarget())
class Model:
    def forward(self):
        return None
