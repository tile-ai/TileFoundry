"""Which models the corpus contains.

One list of names, and each model's own package states what is selected from it.
A model appears here once; what its package does not select is untested, and the
report derives that from the model's own function inventory rather than from a
second list somebody has to remember to update.

The names are written out rather than discovered from the filesystem. A directory
appearing in the corpus because it exists would make "in the corpus" an accident
of layout, and a model half-written would join it silently. Naming it here is the
act of putting it in.

Every gate in those packages states a limit that was measured, not one that was
expected. A gate is a claim about today: if the limit it names is lifted and the
case starts passing, the case fails until the package is corrected, so the matrix
cannot quietly drift into describing a system nobody has.

Analyze selects every function a model defines. Schedule cannot: the device-wide
partition algorithm decides the launch, so it admits only the module entry
function, and selecting a leaf for it would be selecting something the algorithm
has no answer to rather than something it answers badly.

`sized` is a third question, asked separately because a model can answer the
others without answering it: whether it can be analysed at a context length of
the caller's choosing. A model authored as one fixed shape analyses and schedules
perfectly well and has no context length to state, and the two facts must not be
collapsed -- a working analysis recorded as broken, or a missing capability
recorded as nothing at all. It stays its own row once a model answers both, so
there is somewhere to record the next model that answers only one.

A case that names `dims` is asking about the model at those extents. A model that
leaves a dimension open cannot be asked about without them at all -- counting
elements needs a number and a range is not one -- so for those the extents are
part of the question rather than a refinement of it.
"""

from __future__ import annotations

import importlib

from tests.models.corpus import ModelCase

#: The models in the corpus, by package name under `tests/models/`.
MODELS: tuple[str, ...] = (
    "qwen3_1_7b",
    "qwen2_5_1_5b",
)


def _case(package: str) -> ModelCase:
    """The `CASE` the named package states.

    Imported by name, so a package that is listed and states none fails loudly
    here rather than being skipped -- being listed is what puts a model in the
    corpus, and a listed model contributing nothing would read as a model with
    nothing to select.
    """
    module = importlib.import_module(f"tests.models.{package}.case")
    case = getattr(module, "CASE", None)
    if not isinstance(case, ModelCase):
        raise TypeError(
            f"tests.models.{package}.case must state CASE as a ModelCase, "
            f"got {type(case).__name__}"
        )
    if case.id != package:
        raise ValueError(
            f"tests.models.{package}.case states id {case.id!r}; a model's case "
            f"id must be the package that holds it, so a report row names "
            f"something a reader can go and open"
        )
    return case


CORPUS: tuple[ModelCase, ...] = tuple(_case(package) for package in MODELS)


def case(model_id: str) -> ModelCase:
    """The one model case called *model_id*."""
    for model in CORPUS:
        if model.id == model_id:
            return model
    known = ", ".join(model.id for model in CORPUS)
    raise KeyError(f"no model case {model_id!r} in the corpus; it holds {known}")


__all__ = ["CORPUS", "MODELS", "case"]
