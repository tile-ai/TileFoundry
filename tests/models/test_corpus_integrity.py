"""The corpus is held to being describable before it is held to working.

A capability matrix is read as a claim about a system, so the ways it can lie are
worth failing on directly rather than hoping a reader notices. Five of them are
checked here, each because it produces a report that looks complete:

- two cases with one id collapse into one row, and whichever ran second silently
  replaces the first;
- a reference with no oracle is a boundary compared against nothing;
- a block with no reason cannot be reviewed or retired, so it becomes permanent;
- a gate that has gone stale describes a limit nobody has any more;
- a model whose configuration is not pinned is a model whose dimensions can change
  under a green report.

Each is asserted against the corpus itself rather than against a list kept beside
it, so a model added tomorrow is checked by the same five rules.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import re
from pathlib import Path

import pytest

from tests.models.corpus import ModelCase
from tests.models.registry import CORPUS, MODELS

#: A pinned revision is a full commit sha, not a branch. A branch names whatever it
#: points at today, which is the thing being pinned against.
_REVISION = re.compile(r"\b[0-9a-f]{40}\b")

#: A pinned digest is a sha256 of the fetched file.
_DIGEST = re.compile(r"\b[0-9a-f]{64}\b")


def _all_cases() -> tuple[ModelCase, ...]:
    return CORPUS


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    """The plain names this statement binds, unpacking included."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [
        one.id
        for target in targets
        for one in (target.elts if isinstance(target, ast.Tuple | ast.List) else [target])
        if isinstance(one, ast.Name)
    ]


def _is_a_lookup(value: ast.expr) -> bool:
    """`<expr>.lookup(...)`, however the receiver is spelled, or a tuple of them."""
    if isinstance(value, ast.Tuple | ast.List):
        return any(_is_a_lookup(element) for element in value.elts)
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "lookup"
    )


def test_no_two_cases_share_an_id() -> None:
    """One id, one row. Two cases sharing one would report as one case, and which
    of the two the row describes would depend on collection order."""
    seen: dict[str, str] = {}
    for case in _all_cases():
        assert case.id not in seen, (
            f"case id {case.id!r} is stated by both {seen[case.id]} and "
            f"{case.model}; a report row would name one and describe the other"
        )
        seen[case.id] = case.model


def test_no_two_selected_cases_share_an_id() -> None:
    """The same rule for what is selected from a model, across every kind.

    Analyze, schedule, sized and reference ids all end up in one report, so an id
    reused between two of them is the same collapse one level down.
    """
    seen: dict[str, str] = {}
    for model in _all_cases():
        selected = [
            *((case.id, "analyze") for case in model.analyze),
            *((case.id, "schedule") for case in model.schedule),
            *((case.id, "sized") for case in model.sized),
        ]
        if model.reference is not None:
            selected.append((model.reference.id, "reference"))
        for case_id, kind in selected:
            where = f"{model.id}/{kind}"
            assert case_id not in seen, (
                f"selected case id {case_id!r} is stated by both {seen[case_id]} "
                f"and {where}"
            )
            seen[case_id] = where


def test_every_reference_states_an_oracle_and_a_boundary() -> None:
    """A reference without something to be judged against is not a reference.

    `inputs` and `oracle` are one pair by construction elsewhere; what this adds is
    that both are there and callable, and that the boundary is written down. A
    boundary left blank is how a reference quietly shrinks to a leaf op while still
    reporting PASS.
    """
    for model in _all_cases():
        reference = model.reference
        if reference is None:
            continue
        assert callable(reference.inputs), f"{reference.id} draws no inputs"
        assert callable(reference.oracle), f"{reference.id} states no oracle"
        assert len(reference.boundary.split()) >= 5, (
            f"{reference.id} states its boundary as {reference.boundary!r}, which "
            f"is too short to say what was run"
        )


def test_every_model_declares_at_least_one_reference() -> None:
    """A model in the corpus is held to an oracle somewhere."""
    with_reference = {
        case.model for case in _all_cases() if case.reference is not None
    }
    missing = sorted({case.model for case in _all_cases()} - with_reference)
    assert not missing, f"these models are held to no oracle at all: {missing}"


def test_every_block_states_a_reason() -> None:
    """`CapabilityGate` refuses an unreasoned block at construction; this is the
    same rule asked of the whole corpus at once, so a gate built some other way
    cannot slip past it."""
    for model in _all_cases():
        gates = [
            *((case.gate, case.id) for case in model.analyze),
            *((case.gate, case.id) for case in model.schedule),
            *((case.gate, case.id) for case in model.sized),
        ]
        if model.reference is not None:
            gates.append((model.reference.gate, model.reference.id))
        for gate, case_id in gates:
            if not gate.blocked:
                assert not gate.reason, (
                    f"{case_id} states a reason while passing: {gate.reason!r}"
                )
                continue
            assert len(gate.reason.split()) >= 5, (
                f"{case_id} is blocked on {gate.reason!r}, which does not say "
                f"enough to review or retire it"
            )


def test_a_blocked_reason_names_what_would_lift_it() -> None:
    """A reason is only actionable if it says what the limit is about.

    Checked as "names something concrete" rather than by matching prose: a reason
    that mentions no version, no class, no operation and no shape is a statement
    that something did not work, which is what was already known.
    """
    concrete = re.compile(
        r"\d|transformers|no [a-z]+ (implementation|evaluator|oracle)|"
        r"[A-Z][a-zA-Z0-9_]{3,}"
    )
    for model in _all_cases():
        gates = [
            *((case.gate, case.id) for case in model.analyze),
            *((case.gate, case.id) for case in model.schedule),
            *((case.gate, case.id) for case in model.sized),
        ]
        if model.reference is not None:
            gates.append((model.reference.gate, model.reference.id))
        for gate, case_id in gates:
            if not gate.blocked:
                continue
            assert concrete.search(gate.reason), (
                f"{case_id} is blocked on {gate.reason!r}, which names nothing a "
                f"reader could go and check"
            )


@pytest.mark.parametrize("package", MODELS)
def test_every_model_pins_the_configuration_it_was_built_from(package: str) -> None:
    """Dimensions come from something stated, and stated so it can be checked.

    Without a pin, a model's shape is whatever somebody typed, and a report about
    the model is a report about that. It is not a hypothetical: two packages here
    quoted `max_position_embeddings` wrong -- one of them quoted a library default
    while naming the model's published file -- and pinning is what surfaced it.

    Three forms count, because three are honest:

    - a published file, by URL, at a full-sha revision, with its digest;
    - a file checked in beside the module, by name, with its digest -- the artifact
      itself is the record, and anyone with the repository can verify it;
    - neither, with a stated reason. One repository here is gated, so an
      unauthenticated fetch returns prose instead of a configuration; a digest of
      that prose would look like a pin and be worth less than none.

    What is refused is silence.
    """
    module = importlib.import_module(f"tests.models.{package}.config")
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    reason = getattr(module, "SOURCE_UNPINNED_REASON", "")

    if reason:
        assert len(reason.split()) >= 10, (
            f"{package}/config.py says its configuration is unpinned but not "
            f"enough about why: {reason!r}"
        )
        assert getattr(module, "SOURCE_URL", ""), (
            f"{package}/config.py states no source at all, pinned or not"
        )
        return

    assert _DIGEST.search(source), (
        f"{package}/config.py states no sha256 of the configuration it read, and "
        f"no reason why it cannot"
    )
    if getattr(module, "SOURCE_FILE", ""):
        return
    assert "://" in source, f"{package}/config.py names no source for its dimensions"
    assert _REVISION.search(source), (
        f"{package}/config.py names a published source but pins no 40-character "
        f"revision; a branch names whatever it points at today"
    )


@pytest.mark.parametrize("package", MODELS)
def test_a_stated_digest_is_the_digest_of_what_is_checked_in(package: str) -> None:
    """Where a package checks a configuration in, some stated digest has to be of it.

    A digest of something not present cannot be wrong, which makes it worth nothing.
    Checked against every digest the module states rather than a named one, because
    a package may pin both a published file and the subtree it keeps.
    """
    module = importlib.import_module(f"tests.models.{package}.config")
    directory = Path(inspect.getfile(module)).parent
    checked_in = sorted(directory.glob("*config*.json"))
    if not checked_in:
        pytest.skip(f"{package} keeps no configuration file of its own")

    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    stated = set(_DIGEST.findall(source))
    for path in checked_in:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest in stated, (
            f"{package}/{path.name} hashes to {digest}, which {package}/config.py "
            f"does not state; the file and its pin have drifted apart"
        )


@pytest.mark.parametrize("package", MODELS)
def test_no_model_file_re_exports_a_looked_up_function(package: str) -> None:
    """A module-level `NAME = Root.lookup("...")` in a model file is a re-export
    shim: the node already has a name inside the tree. Matched on the assignment's
    shape, so calling `lookup` inline anywhere stays legitimate.
    """
    module = importlib.import_module(f"tests.models.{package}.model")
    tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))
    shims = [
        f"line {node.lineno}: {', '.join(_assigned_names(node))}"
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and node.value is not None
        and _assigned_names(node)
        and _is_a_lookup(node.value)
    ]
    assert not shims, (
        f"{package}/model.py binds a looked-up function to a module-level name "
        f"({'; '.join(shims)}); call it where it is used instead"
    )
