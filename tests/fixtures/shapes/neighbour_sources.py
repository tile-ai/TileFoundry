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
