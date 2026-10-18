"""Reload exported scheduled/TIR sources, without compiling machine code."""

import argparse, importlib.util
from config import ROOT
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction


def verify(model):
    path = ROOT / "artifacts" / ("ir-" + model)
    for name in ("model", "scheduled", "finalized"):
        spec = importlib.util.spec_from_file_location(
            "exported_" + name, path / (name + ".py")
        )
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        root = loaded.QwenPrefill
        assert isinstance(root, Module)
        if name == "finalized":
            assert isinstance(root.entry_function(), PrimFunction)
        print(
            "PASS",
            model,
            name,
            "shared layers:",
            len({id(m) for m in root.layers}),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("1.7b", "4b"), required=True)
    verify(p.parse_args().model)
