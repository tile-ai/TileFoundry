# TileFoundry in two steps

TileFoundry is source to source: the reference is source code, the fast
implementation is source code, and either can be pointed at any command.
`check` says whether two of them agree; `analyze` says what one costs.

```text
  step one — describe it, until it agrees

        ┌────────── fix the HIR ◄────────── not yet ───────────┐
        ▼                                                      │
   authored HIR ─────► check ─────► agrees? ───────────────────┘
   the reference                       │
        ▲                             yes
        │                              ▼
   published model            the reference is finished

  step two — make it fast; both roads lead back to the same source

        ┌────── change the HIR ◄────── not yet ───────────────────┐
        ▼                                                         │
   authored HIR ─────► analyze ─────► predicted performance ok? ──┘
        ▲                                     │
        │                                    yes
        │                                     ▼
        │          write a runtime twin ─────► check ─────► measure
        │                                                     │
        │                                       measured performance ok?
        │                                             ┌───────┴───────┐
        └─────────────────── no ◄─────────────────────┘               └──► ship
```

## Where the other answers are

- `tilefoundry tutorial orchestrator causal_lm` — the shipped autoregressive decode
  loop.
