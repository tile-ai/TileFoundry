"""A tiled MLP against a naive one: the loop nest reassociates, and nothing else.

Test-only: no checkpoint, no Hugging Face oracle, no corpus or catalog entry.
The dimensions below are this file's own, sized so the blocking divides.
"""
from __future__ import annotations

import torch

from tests.models.decode_oracle import agrees_to_one_rounding
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.runtime.resource import DictResource

#: One token per step, as every decode kernel in the corpus is authored.
S = 1

#: Wide enough that the K walk actually walks -- 4 gate/up steps and 12 down
#: steps -- and small enough to run on a host in the time a unit test may take.
HIDDEN, INTERMEDIATE = 256, 768

#: The dtype the corpus's checkpoints publish, and so the one the rewrite has to
#: hold at: bf16 is where a K reduction that rounds per tile diverges from one
#: that rounds once, and f32 would hide exactly the defect this test exists for.
_DT = "bf16"
_TORCH_DT = torch.bfloat16

# ── block shape ──────────────────────────────────────────────────────────────
# The AMX f32 register files (Apple M2 Pro, target/hardware/*.toml): Z holds
# 4096 B = 32x32 f32, X and Y 512 B each. NT x KT are sized against those; the
# token axis is not blocked at all, because a decode step has one token to
# block. MT = 1 makes each block matmul the [1, KT] @ [KT, NT] row-times-panel
# the step actually performs.
MT, NT, KT = 1, 32, 64
MB = S // MT                    # token blocks
NB_INT = INTERMEDIATE // NT     # gate/up column blocks
NB_HID = HIDDEN // NT           # down-projection column blocks
NK_HID = HIDDEN // KT           # gate/up K steps
NK_INT = INTERMEDIATE // KT     # down-projection K steps


@module(entry="tiled_mlp")
class TiledMLP:
    """Both spellings of one gated MLP, so they can be run against each other."""

    @func
    def naive_mlp(
        x: Tensor[(1, S, HIDDEN), _DT],
        w_gate: ConstTensor[(1, HIDDEN, INTERMEDIATE), _DT],
        w_up: ConstTensor[(1, HIDDEN, INTERMEDIATE), _DT],
        w_down: ConstTensor[(1, INTERMEDIATE, HIDDEN), _DT],
    ) -> Tensor[(1, S, HIDDEN), _DT]:
        gate = tf.matmul(x, w_gate)
        up = tf.matmul(x, w_up)
        return tf.matmul(tf.silu(gate) * up, w_down)

    @func
    def tiled_mlp(
        x: Tensor[(1, S, HIDDEN), _DT],
        w_gate: ConstTensor[(1, HIDDEN, INTERMEDIATE), _DT],
        w_up: ConstTensor[(1, HIDDEN, INTERMEDIATE), _DT],
        w_down: ConstTensor[(1, INTERMEDIATE, HIDDEN), _DT],
    ) -> Tensor[(1, S, HIDDEN), _DT]:
        # The same value, written as the loop nest AMX wants: every matmul is
        # [MT, KT] @ [KT, NT] over a (token-block, column-block) batch pair, and
        # the K walk is an authored `for ... in tile(...)` whose carried arg IS
        # the accumulator buffer -- `zeros` declares it, the loop carry holds it,
        # and nothing allocates. The reshape/transpose pairs only re-index:
        # [1, S, K] -> [NK, MB, 1, MT, KT] blocks the M/K axes, [1, K, N] ->
        # [NK, 1, NB, KT, NT] the K/N axes, and `gather(_, k, axis=0)` picks
        # iteration k's K slice of both.
        x_blk = tf.reshape(
            tf.transpose(
                tf.reshape(x, new_shape=(MB, MT, NK_HID, KT)), perm=(2, 0, 1, 3)
            ),
            new_shape=(NK_HID, MB, 1, MT, KT),
        )
        wg_blk = tf.reshape(
            tf.transpose(
                tf.reshape(w_gate, new_shape=(NK_HID, KT, NB_INT, NT)), perm=(0, 2, 1, 3)
            ),
            new_shape=(NK_HID, 1, NB_INT, KT, NT),
        )
        wu_blk = tf.reshape(
            tf.transpose(
                tf.reshape(w_up, new_shape=(NK_HID, KT, NB_INT, NT)), perm=(0, 2, 1, 3)
            ),
            new_shape=(NK_HID, 1, NB_INT, KT, NT),
        )
        # f32 accumulator, as the matmul it rewrites already has: a bf16 matmul
        # sums K in f32 and rounds once out. A bf16 partial would round per tile.
        gate_z = tf.zeros(shape=(MB, NB_INT, MT, NT), dtype="f32")
        up_z = tf.zeros(shape=(MB, NB_INT, MT, NT), dtype="f32")
        for kh in tile(NK_HID):
            x_k = tf.cast(tf.gather(x_blk, kh, axis=0), dtype="f32")
            gate_z = gate_z + tf.matmul(
                x_k, tf.cast(tf.gather(wg_blk, kh, axis=0), dtype="f32")
            )
            up_z = up_z + tf.matmul(
                x_k, tf.cast(tf.gather(wu_blk, kh, axis=0), dtype="f32")
            )
        gate = tf.cast(
            tf.reshape(
                tf.transpose(gate_z, perm=(0, 2, 1, 3)),
                new_shape=(1, S, INTERMEDIATE),
            ),
            dtype=_DT,
        )
        up = tf.cast(
            tf.reshape(
                tf.transpose(up_z, perm=(0, 2, 1, 3)),
                new_shape=(1, S, INTERMEDIATE),
            ),
            dtype=_DT,
        )
        h = tf.silu(gate) * up
        h_blk = tf.reshape(
            tf.transpose(tf.reshape(h, new_shape=(MB, MT, NK_INT, KT)), perm=(2, 0, 1, 3)),
            new_shape=(NK_INT, MB, 1, MT, KT),
        )
        wd_blk = tf.reshape(
            tf.transpose(
                tf.reshape(w_down, new_shape=(NK_INT, KT, NB_HID, NT)), perm=(0, 2, 1, 3)
            ),
            new_shape=(NK_INT, 1, NB_HID, KT, NT),
        )
        out_z = tf.zeros(shape=(MB, NB_HID, MT, NT), dtype="f32")
        for ki in tile(NK_INT):
            out_z = out_z + tf.matmul(
                tf.cast(tf.gather(h_blk, ki, axis=0), dtype="f32"),
                tf.cast(tf.gather(wd_blk, ki, axis=0), dtype="f32"),
            )
        return tf.cast(
            tf.reshape(
                tf.transpose(out_z, perm=(0, 2, 1, 3)),
                new_shape=(1, S, HIDDEN),
            ),
            dtype=_DT,
        )


def _drawn(device="cpu", seed=0):
    """One input and one set of weights, shared by both spellings."""
    torch.manual_seed(seed)

    def draw(*shape):
        return (torch.randn(*shape, device=device) * 0.05).to(_TORCH_DT)

    x = draw(1, S, HIDDEN)
    loaded = TiledMLP.cloned().load(
        DictResource({
            "w_gate": draw(1, HIDDEN, INTERMEDIATE),
            "w_up": draw(1, HIDDEN, INTERMEDIATE),
            "w_down": draw(1, INTERMEDIATE, HIDDEN),
        })
    )
    return loaded, x


def test_the_tiled_mlp_computes_what_the_naive_one_does() -> None:
    """The blocked K walk reassociates the reduction and changes nothing else."""
    loaded, x = _drawn()

    agrees_to_one_rounding(loaded.tiled_mlp(x), loaded.naive_mlp(x))
