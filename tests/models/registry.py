"""Which models the corpus contains.

One list of names, and each model's own package states what is selected from it.
A model appears here once; what its package does not select is untested, and the
report derives that from the model's own function inventory rather than from a
second list somebody has to remember to update.

A package states one case per Module it selects from -- one for most models, three
for a hybrid stack whose token mixers are separate Modules. Every case names its
package as its model, so the report is one row per model regardless.

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

#: The models in the corpus, by package name under `tests/models/`.  The
#: `orchestrator/` package holds shared family code, not a model case.
MODELS: tuple[str, ...] = (
    "qwen3_1_7b",
    "qwen2_5_1_5b",
    "gemma2_2b",
    "minicpm3_4b",
    "deepseek_v4_flash",
    "qwen3_5_35b_a3b",
    "kimi_linear_48b_a3b",
)

def _cases(package: str) -> tuple[ModelCase, ...]:
    """The `CASES` the named package states, one per Module it selects from.

    Imported by name, so a package that is listed and states none fails loudly
    here rather than being skipped -- being listed is what puts a model in the
    corpus, and a listed model contributing nothing would read as a model with
    nothing to select.

    A model whose kernels live in more than one Module states more than one case,
    because a Module is the execution domain of the functions it owns and analysis
    selects functions of one Module. Every one of them names *this* package as its
    model, so the report stays one row per model however many Modules that took.
    """
    module = importlib.import_module(f"tests.models.{package}.case")
    cases = getattr(module, "CASES", None)
    if not isinstance(cases, tuple) or not cases:
        raise TypeError(
            f"tests.models.{package}.case must state CASES as a non-empty tuple "
            f"of ModelCase, got {type(cases).__name__}"
        )
    for case in cases:
        if not isinstance(case, ModelCase):
            raise TypeError(
                f"tests.models.{package}.case states a "
                f"{type(case).__name__} in CASES, not a ModelCase"
            )
        if case.model != package:
            raise ValueError(
                f"tests.models.{package}.case states a case whose model is "
                f"{case.model!r}; every case a package states must name that "
                f"package as its model, so a report row names something a reader "
                f"can go and open"
            )
    return cases


# The authored-loop kernels are corpus inputs, not models shipped in the catalog.
CORPUS: tuple[ModelCase, ...] = tuple(
    case for package in MODELS for case in _cases(package)
) + _cases("access_footprint")


def case(case_id: str) -> ModelCase:
    """The one case called *case_id*."""
    for model in CORPUS:
        if model.id == case_id:
            return model
    known = ", ".join(model.id for model in CORPUS)
    raise KeyError(f"no case {case_id!r} in the corpus; it holds {known}")


def cases_of(model_id: str) -> tuple[ModelCase, ...]:
    """Every case the model called *model_id* states, in the order it states them."""
    found = tuple(model for model in CORPUS if model.model == model_id)
    if not found:
        known = tuple(dict.fromkeys(case.model for case in CORPUS))
        raise KeyError(f"no model {model_id!r} in the corpus; it holds {known}")
    return found


__all__ = ["CORPUS", "MODELS", "case", "cases_of"]
