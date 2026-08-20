"""One file holding a sound root beside an unsound one, as DSL source for the CLI.

A selector names one root of a file, and whether the rest of that file is sound
is not a fact about the one that was named. These are text rather than importable
fixtures because importing them is the thing under test: the unsound class raises
while the file executes. `broken_after` and `broken_before` are the same two roots
in the two orders, and the order is the whole question -- a reading that survives
only when the unsound class comes last is one that got away with it.
"""

from __future__ import annotations

_HEAD = (
    "from tilefoundry import func, module\n"
    "from tilefoundry.dsl import Mesh, Tensor, Topology, tf\n"
    "from tilefoundry.target import CudaTarget\n"
    "N = 132 * 128\n"
    "_H200 = CudaTarget('nvidia.h200_sxm')\n"
)

_UNSOUND = (
    "@module(entry='nope', target=_H200, topologies=(Topology('cta', 132),))\n"
    "class Unsound:\n"
    "    @func\n"
    "    def kernel(x: Tensor[(N,), 'f32']) -> Tensor[(N,), 'f32']:\n"
    "        with Mesh(('cta',), layout=(132,), names=('block',)) as m:\n"
    "            placed = tf.reshard(x, (N @ m.block,), 'gmem')\n"
    "            return tf.reshard(tf.square(placed), (N @ m.block,), 'gmem')\n"
)

_SOUND = (
    "@module(entry='kernel', target=_H200, topologies=(Topology('cta', 132),))\n"
    "class Sound:\n"
    "    @func\n"
    "    def kernel(x: Tensor[(N,), 'f32']) -> Tensor[(N,), 'f32']:\n"
    "        with Mesh(('cta',), layout=(132,), names=('block',)) as m:\n"
    "            placed = tf.reshard(x, (N @ m.block,), 'gmem')\n"
    "            return tf.reshard(tf.square(placed), (N @ m.block,), 'gmem')\n"
)

_CHILD = (
    "@module(entry='step')\n"
    "class Child:\n"
    "    @func\n"
    "    def step(x: Tensor[(N,), 'f32']) -> Tensor[(N,), 'f32']:\n"
    "        return tf.square(x)\n"
)

_PARENT = (
    "@module(entry='kernel', target=_H200, topologies=(Topology('cta', 132),))\n"
    "class Parent:\n"
    "    inner = Child\n"
    "    @func\n"
    "    def kernel(x: Tensor[(N,), 'f32']) -> Tensor[(N,), 'f32']:\n"
    "        with Mesh(('cta',), layout=(132,), names=('block',)) as m:\n"
    "            placed = tf.reshard(x, (N @ m.block,), 'gmem')\n"
    "            return tf.reshard(inner(placed), (N @ m.block,), 'gmem')\n"
)


def broken_after() -> str:
    """`Sound` first, then `Unsound`."""
    return _HEAD + _SOUND + _UNSOUND


def broken_before() -> str:
    """`Unsound` first, then `Sound`."""
    return _HEAD + _UNSOUND + _SOUND


def broken_beside_a_parent() -> str:
    """`Unsound` between a child and the sound parent that reaches it."""
    return _HEAD + _CHILD + _UNSOUND + _PARENT


def broken_named_like_a_local() -> str:
    """The unsound root is called `x`, which is also the sound kernel's parameter.

    Nothing connects the two: one is a module-level class and the other is a name
    bound inside a function. A reading that collects every name a class mentions
    cannot tell them apart, and concludes the selection needs the unsound class.
    """
    return _HEAD + _UNSOUND.replace("class Unsound:", "class x:") + _SOUND


def broken_inside_a_compound_statement() -> str:
    """The unsound root is defined inside a top-level `if`.

    A statement that is not itself a class definition still defines one, so a
    reading that only recognises a bare `class` at the top level sees a statement
    that binds nothing and keeps it.
    """
    guarded = "if N:\n" + "".join(
        f"    {line}\n" for line in _UNSOUND.splitlines()
    )
    return _HEAD + guarded + _SOUND


def broken_beside_an_alias() -> str:
    """The selector names `Root`, an alias of the sound class rather than a class.

    A selector names a root of the file, and a root reached by a second name is
    still that root. A reading that looks for a class statement of the selected
    name finds none and gives up on isolating anything.
    """
    return _HEAD + _UNSOUND + _SOUND.replace("class Sound:", "class Built:") + "Root = Built\n"


def broken_beside_a_future_annotation() -> str:
    """A file whose annotations are postponed, holding one unsound root.

    `from __future__ import annotations` governs the statements after it, so a
    statement compiled without it in front reads a forward-referencing annotation
    as a name to resolve now. The dataclass here annotates a class defined below
    it, which is exactly what that import is for, and which fails without it.
    """
    return (
        "from __future__ import annotations\n"
        + _HEAD
        + "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Holder:\n"
        "    later: Described\n"
        "class Described:\n"
        "    pass\n"
        + _UNSOUND
        + _SOUND
    )


def broken_beside_an_eager_annotation() -> str:
    """The same file with no `__future__` import, so its annotations are eager.

    What a file postpones is the file's own decision. This one postpones nothing
    and annotates a class defined above it, so the annotation is that class rather
    than its name -- which is what a load that handed out postponed annotations to
    a file that never asked for them would get wrong.
    """
    return (
        _HEAD
        + "from dataclasses import dataclass\n"
        "class Described:\n"
        "    pass\n"
        "@dataclass\n"
        "class Holder:\n"
        "    later: Described\n"
        + _UNSOUND
        + _SOUND
    )


def both_roots_unsound() -> str:
    """Two roots that both refuse, so which one is reported is the question.

    The unrelated one refuses first. A load that reported the first refusal it saw
    would answer about a program nobody asked about, and the selection's own
    reason -- the only one the caller can act on -- would be the one thrown away.
    """
    return (
        _HEAD
        + _UNSOUND
        + _SOUND.replace("entry='kernel'", "entry='absent_from_sound'", 1)
    )


def configured_after_it_is_built() -> str:
    """A sound root, then a statement that reconfigures it and fails.

    The root is bound by the time that statement runs, so asking only whether it
    is bound says yes. What it is bound to is not what the file describes: the
    statement meant to replace it never finished, so the answer would be a program
    the file does not state.
    """
    return _HEAD + _SOUND + "Sound = Sound.no_such_attribute\n"


def future_import_out_of_place() -> str:
    """A `__future__` import after an ordinary statement.

    Python refuses this: the import governs how the file compiles, so it has to
    come before there is anything to govern. A load that gathered the file's
    `__future__` imports from anywhere and put them in front would accept a file
    no interpreter does.
    """
    return (
        "x = 1\n"
        "from __future__ import annotations\n" + _HEAD + _UNSOUND + _SOUND
    )


def unsound_with_a_same_named_attribute() -> str:
    """The unsound root sets a class attribute called `Sound`.

    Writing a name is not reading one. A class that only happens to have an
    attribute spelled like the selection has said nothing about it, so its own
    refusal is still not the selection's.
    """
    return (
        _HEAD
        + _UNSOUND.replace("class Unsound:\n", "class Unsound:\n    Sound = 1\n", 1)
        + _SOUND
    )


def unsound_with_a_same_named_comprehension() -> str:
    """The unsound root binds `Sound` as a comprehension variable.

    The name lives for the length of the comprehension and means nothing outside
    it, so a load that treated it as a reading of the selection would block on a
    coincidence of spelling.
    """
    return (
        _HEAD
        + _UNSOUND.replace(
            "class Unsound:\n",
            "class Unsound:\n    counted = [Sound for Sound in range(3)]\n",
            1,
        )
        + _SOUND
    )


def documented_and_postponed() -> str:
    """A module docstring, then a `__future__` import, then the roots.

    Both are things a file states about itself, and each statement executed on its
    own has to keep them. A docstring is a docstring because it comes first, so a
    load that put anything in front of it would leave the file with none.
    """
    return (
        '"""The file\'s own docstring."""\n'
        "from __future__ import annotations\n"
        + _HEAD
        + _UNSOUND
        + _SOUND
        + "SEEN = __doc__\n"
    )


def unsound_by_control_exception() -> str:
    """The unsound root raises something that is not an `Exception`.

    Setting a failure aside is for an unfinished program. A control exception is
    not that: it is the interpreter unwinding for its own reasons, and catching it
    to carry on with the load would be answering while something else is ending.
    """
    return (
        _HEAD
        + _UNSOUND.replace(
            "class Unsound:\n",
            "class Unsound:\n    raise GeneratorExit('unwinding')\n",
            1,
        )
        + _SOUND
    )


def unsound_annotating_the_selection(postponed: bool) -> str:
    """An unsound helper whose annotations name the selection.

    Whether those annotations are read while the file loads is the file's own
    decision: postponed, they are strings and name nothing yet, so the helper's
    own failure is not the selection's; evaluated, they really do reach for the
    selection and its failure is. The two files differ in one line.
    """
    head = "from __future__ import annotations\n" if postponed else ""
    return (
        head
        + _HEAD
        + "def absent(value):\n"
        "    return value\n"
        "@absent.no_such_attribute\n"
        "def helper(value: Sound) -> Sound:\n"
        "    return value\n"
        + _SOUND
    )
