"""What `case.py` claims about this model, held to being true today.

The corpus verifies every registry claim: its context length is supported and
each selected question gets an answer. MLA relies on ``tf.slice`` and
``tf.concat`` for split head rotation; the cost registry now evaluates both, so
the case has no gate. These tests detect any regression to an unverified case.
"""

from __future__ import annotations

import pytest

from tests.models.deepseek_v4_flash.case import ANALYZED_AT, CASE
from tests.models.deepseek_v4_flash.model import REAL
from tilefoundry.ir.hir.specialize import specialize_concretely
from tilefoundry.ir.types.substitute import DimSubstitutionError


def test_the_case_selects_every_function_the_description_defines():
    """Analyze has no reason to leave a function out.

    Every function this Module defines is selected, so the model's own inventory
    and the case's selection agree and nothing escapes the report.
    """
    module = CASE.build()
    assert CASE.untested("analyze", module) == ()


def test_the_context_lengths_the_case_names_are_ones_the_model_has():
    """The window is what bounds the range.

    The window is what bounds the range, so the corpus's usual 1024 is not a
    context this layer type has rather than a long one -- and the length the case
    analyses at sits below the ceiling, which is asked about separately.
    """
    function = CASE.build().lookup("mla_attend")

    assert ANALYZED_AT["ctx_len"] < REAL.max_ctx < REAL.window
    sized = specialize_concretely(function, dict(ANALYZED_AT))
    cache = next(p for p in sized.params if p.name == "kv_cache")
    assert tuple(int(d) for d in cache.type.shape) == (
        1,
        ANALYZED_AT["ctx_len"],
        1,
        REAL.head_dim,
    )

    with pytest.raises(
        DimSubstitutionError, match=r"declared over \[0, 128\) and cannot take 1024"
    ):
        specialize_concretely(function, {"ctx_len": 1024})
