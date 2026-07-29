"""Write `tests/models/catalog.json` from the live corpus.

Run as `python -m scripts.generate_model_catalog`; it imports the corpus, so it
only resolves as a module of this repository. Forest and counts come from the
models; validation levels come from `tests/models/verified.json`. What the catalog
is for is in [cli.md](../docs/spec/cli.md).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_MODELS_ROOT = Path(__file__).resolve().parents[1] / "tests" / "models"


def _annotation(param: Any) -> str:
    """The parameter's type as the DSL spells it."""
    from tilefoundry.ir.types import TensorType, TupleType  # noqa: PLC0415

    def extent_text(extent: object) -> str:
        # A ranged dimension is written by name in the DSL, and its repr carries
        # the bounds, which belong to the model's own declaration and not here.
        return getattr(extent, "name", None) or str(extent)

    def render(ty: object) -> str:
        if isinstance(ty, TensorType):
            shape = ", ".join(extent_text(extent) for extent in ty.shape)
            suffix = "," if len(ty.shape) == 1 else ""
            return f'[({shape}{suffix}), "{ty.dtype.name}"]'
        if isinstance(ty, TupleType):
            return "Tuple[" + ", ".join(render(field) for field in ty.fields) + "]"
        return repr(ty)

    kind = "ConstTensor" if param.is_const else "Tensor"
    rendered = render(param.type)
    return f"{kind}{rendered}" if rendered.startswith("[") else rendered


def _function(function: Any) -> str:
    """One function as a signature line, which is also how it is printed."""
    params = ", ".join(f"{param.name}: {_annotation(param)}" for param in function.params)
    return f"{function.name}({params})"


_NUMBERED = re.compile(r"^(.*?)(\d+)$")


def _shape(node: dict[str, Any]) -> str:
    """A node's functions and whole subtree, excluding only its own name."""
    return json.dumps(
        {"functions": node["functions"], "modules": node["modules"]}, sort_keys=True
    )


def _grouped(children: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse runs that are adjacent, one shape, one name stem, and consecutive.

    Every name is kept, so the result is the whole list written as ranges.
    """
    out: list[dict[str, Any]] = []
    for child in children:
        numbered = _NUMBERED.match(child["names"][0])
        if out and numbered and len(child["names"]) == 1:
            last = out[-1]
            previous = _NUMBERED.match(last["names"][-1])
            if (
                previous
                and previous.group(1) == numbered.group(1)
                and int(numbered.group(2)) == int(previous.group(2)) + 1
                and _shape(last) == _shape(child)
            ):
                last["names"].append(child["names"][0])
                continue
        out.append(child)
    return out


def _node(module: Any, counts: dict[str, int]) -> dict[str, Any]:
    """One Module and everything under it, counted by this traversal.

    Counted before any run collapses, so the totals are the real tree's.
    """
    children = [_node(child, counts) for child in module.modules]
    if not children:
        counts["leaf_modules"] += 1
    counts["functions"] += len(module.functions)
    return {
        "names": [module.name],
        "leaf": not children,
        "functions": [_function(function) for function in module.functions],
        "modules": _grouped(children),
    }


def _roots(model: Any) -> list[Any]:
    """The Modules a package defines that no other Module of it contains."""
    from tilefoundry.ir.core.module import Module  # noqa: PLC0415

    declared: list[Any] = []
    seen: set[int] = set()
    for value in vars(model).values():
        if isinstance(value, Module) and id(value) not in seen:
            seen.add(id(value))
            declared.append(value)

    def walk(module: Any):
        yield module
        for child in module.modules:
            yield from walk(child)

    contained = {
        id(child) for root in declared for node in walk(root) for child in node.modules
    }
    return [root for root in declared if id(root) not in contained]


def catalog() -> dict[str, Any]:
    """The catalog as the live corpus describes it right now."""
    import importlib  # noqa: PLC0415

    from tests.models.registry import MODELS  # noqa: PLC0415

    verified = json.loads((_MODELS_ROOT / "verified.json").read_text(encoding="utf-8"))
    models = []
    for name in sorted(MODELS):
        record = verified["models"][name]
        counts = {"leaf_modules": 0, "functions": 0}
        roots = _grouped([
            _node(root, counts)
            for root in _roots(importlib.import_module(f"tests.models.{name}.model"))
        ])
        models.append({
            "name": name,
            "level": record["level"],
            "evidence": record["evidence"],
            "counts": counts,
            "modules": roots,
        })
    return {
        "levels": verified["levels"],
        "oracle_level": verified["oracle_level"],
        "models": models,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="exit non-zero if the file would change"
    )
    arguments = parser.parse_args(argv)

    destination = _MODELS_ROOT / "catalog.json"
    rendered = json.dumps(catalog(), indent=2, sort_keys=True) + "\n"
    if arguments.check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != rendered:
            print(f"{destination} is stale; regenerate it with "
                  f"python -m scripts.generate_model_catalog")
            return 1
        return 0
    destination.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
