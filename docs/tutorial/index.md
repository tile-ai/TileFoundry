# TileFoundry in two steps

TileFoundry is source to source: the reference is source code, the fast
implementation is source code, and either can be pointed at any command.
`check` says whether two of them agree; `analyze` says what one costs.

```text
  step one — describe it, until it agrees

                                 ┌──── not yet ───── ----------─┐
                                 ▼                              │
      published model ─────► authored HIR ─────► check ok ? ────┘
                             the reference          │
                                                 agrees    

  step two — make it fast; both roads lead back to the same source

        ┌────── change the HIR ◄────── not yet ─────-------------─┐
        ▼                                                         │
   authored HIR ─────► analyze ─────► predicted performance ok? ──┘
        ▲                                     │
        │                                    yes
        │                                     ▼
        │                            write a runtime twin ─────► check ─────► measure
        │                                                                       │
        └────────────────── no ◄────────────────────────----------measured performance ok?
                                                                                |
                                                                                └─────────► ship
```

## Where the other answers are

- `tilefoundry tutorial migrate` — describe a published step as authored HIR, and
  make `check` agree with the implementation that shipped.
- `tilefoundry tutorial optimize` — price a placement decision with `analyze`, then
  hold a runtime twin to the authored program.
- `tilefoundry tutorial authoring` — one kernel through six analyze-driven stages.
- `tilefoundry tutorial orchestrator causal_lm` — the shipped autoregressive decode
  loop.
