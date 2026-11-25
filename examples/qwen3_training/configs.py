"""The five parallel configurations of PR #327 and the one model size.

A configuration places 8 GPUs on the mesh ``(pp, data, ep, cp)``. ``dense`` names the
axes that split each dense weight inside a stage (FSDP); an axis left out holds a
whole copy (HSDP). ``expert`` names the axes that split each expert matrix inside one
``ep`` coordinate; ``ep`` itself always splits the experts.
"""

from __future__ import annotations

from dataclasses import dataclass

GPUS = 8
AXES = ("pp", "data", "ep", "cp")


@dataclass(frozen=True)
class Config:
    name: str
    pp: int
    data: int
    ep: int
    cp: int
    dense: tuple[str, ...]
    expert: tuple[str, ...]

    def __post_init__(self):
        units = self.pp * self.data * self.ep * self.cp
        if units != GPUS:
            raise ValueError(
                f"config {self.name!r}: pp*data*ep*cp is {units}, the mesh has {GPUS} units"
            )
        if self.pp != 2:
            raise ValueError(
                f"config {self.name!r}: pp is {self.pp}; the model has a first and a last stage only"
            )

    def size_of(self, axis: str) -> int:
        return getattr(self, axis)


@dataclass(frozen=True)
class Size:
    layers: int = 8
    hidden: int = 2048
    q_heads: int = 32
    kv_heads: int = 4
    head_dim: int = 128
    experts: int = 128
    top_k: int = 8
    expert_hidden: int = 768
    vocab: int = 151936
    tokens: int = 32768
    microbatches: int = 4

    def check(self, config: Config) -> None:
        if self.layers % config.pp:
            raise ValueError(
                f"config {config.name!r}: {self.layers} layers do not split into {config.pp} stages"
            )
        if self.tokens % config.cp:
            raise ValueError(
                f"config {config.name!r}: {self.tokens} tokens do not split into {config.cp} cp parts"
            )
        if self.experts % config.ep or (self.tokens // config.cp * self.top_k) % self.experts:
            raise ValueError(
                f"config {config.name!r}: {self.experts} experts with top {self.top_k} give no whole capacity "
                f"for {self.tokens // config.cp} local tokens on {config.ep} ep parts"
            )


CONFIGS = {
    config.name: config
    for config in (
        Config("cp", pp=2, data=2, ep=1, cp=2, dense=("data", "cp"), expert=("data", "cp")),
        Config("cp_ep", pp=2, data=1, ep=2, cp=2, dense=("ep", "cp"), expert=("cp",)),
        Config("ep", pp=2, data=2, ep=2, cp=1, dense=("data", "ep"), expert=("data",)),
        Config("cp_hsdp", pp=2, data=2, ep=1, cp=2, dense=("cp",), expert=("cp",)),
        Config("ep_hsdp", pp=2, data=2, ep=2, cp=1, dense=("ep",), expert=()),
    )
}
FULL = Size()
