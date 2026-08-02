"""What `case.py` claims about this model, held to being true today.

The corpus harness asks these questions once the case is named in the registry;
until it is, a case is an unverified assertion, and an unverified case is exactly
what a capability matrix exists to prevent. So each claim is checked here: the
context length the case names is one the description can really be asked at, and
every question the case selects really does get an answer.

These questions were blocked when this package was migrated -- MLA splits each
head into its unrotated and rotated halves and DeepSeek's interleaved rotation
takes every other channel, so both kernels are built out of `tf.slice` and
`tf.concat`, and the cost registry could evaluate neither. It can now, so the
case states no gate and this file is what would notice if that regressed.
"""

from __future__ import annotations

import pytest

from tests.models.deepseek_v4_flash.case import ANALYZED_AT, CASE
from tests.models.deepseek_v4_flash.model import REAL
from tilefoundry.ir.hir.specialize import specialize_concretely
from tilefoundry.ir.types.substitute import DimSubstitutionError


def test_the_case_selects_every_function_the_description_defines():
    """Analyze has no reason to leave a function out, and schedule admits only
    the entry function -- so the other one is untested rather than blocked."""
    module = CASE.build()
    assert CASE.untested("analyze", module) == ()
    assert CASE.selected("schedule") == (module.entry_function().name,)
    assert CASE.untested("schedule", module) == ("mla_kv_update",)


def test_the_context_lengths_the_case_names_are_ones_the_model_has():
    """The window is what bounds the range, so the corpus's usual 1024 is not a
    context this layer type has rather than a long one -- and the length the case
    analyses at sits below the ceiling, which is asked about separately."""
    function = CASE.build().lookup("mla_attend")

    assert ANALYZED_AT["ctx_len"] < REAL.max_ctx < REAL.window
    sized = specialize_concretely(function, dict(ANALYZED_AT))
    cache = next(p for p in sized.params if p.name == "kv_cache")
    assert tuple(int(d) for d in cache.type.shape) == (
        1, ANALYZED_AT["ctx_len"], 1, REAL.head_dim,
    )

    with pytest.raises(
        DimSubstitutionError, match=r"declared over \[0, 128\) and cannot take 1024"
    ):
        specialize_concretely(function, {"ctx_len": 1024})
