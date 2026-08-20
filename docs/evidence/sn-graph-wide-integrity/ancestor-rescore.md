# Ancestor rescore receipt

Status: **complete**. Each of the three existing identities received exactly
one fresh ordinary name-review quorum group. No identity was reworded, refined,
or retried; all three finish unclaimed with zero refinement attempts.

## Outcomes

| Identity | Before | Fresh result | Resolution | Scope run | Paying run | Attributable USD | Disposition |
|---|---:|---:|---|---|---|---:|---|
| `radial_ion_momentum` | reviewed / 0.675 | reviewed / 0.5875 | authoritative escalation | `sn-rescore-20260820T141241Z` | `fc07d97d-b8f6-4399-8a5d-cdafadb7df9b` | $0.240110 | Did not clear; no retry |
| `radial_momentum_flux` | reviewed / 0.93125 | accepted / 0.95625 | quorum consensus | `sn-rescore-20260820T142320Z` | `d3c2382d-232a-40f9-8cb3-edafc8479e90` | $0.429416 | Accepted by the fresh score |
| `poloidal_neutral_state_momentum_flux` | reviewed / 0.9 | reviewed / 0.825 | authoritative escalation | `sn-rescore-20260820T144208Z` | `28eee607-8266-477b-931d-588dcafb9264` | $0.187178 | Did not clear; no retry |

The acceptance threshold was 0.85. Only `radial_momentum_flux` cleared it.
It reached `accepted` through its fresh quorum-consensus score and through no
other route.

`radial_ion_momentum` is a first-class negative result: the fresh independent
draw scored it **lower**, from 0.675 to 0.5875. That is not a result to work
around. It remains `reviewed`, it was not rescored again, reworded, or
escalated beyond the quorum's ordinary third seat, and its ancestor fold stays
gated. `poloidal_neutral_state_momentum_flux` likewise did not clear at 0.825;
it remains `reviewed` and its fold stays gated.

## Reviewer cycles

Every name has one and only one fresh name-review group.

| Identity | Axis | Cycle | Role | Reviewer model | Cycle score | Resolution |
|---|---|---:|---|---|---:|---|
| `radial_ion_momentum` | name | 0 | primary | `openrouter/x-ai/grok-4.5` | 0.5875 | — |
| `radial_ion_momentum` | name | 1 | secondary | `openrouter/openai/gpt-5.6-luna` | 0.7375 | — |
| `radial_ion_momentum` | name | 2 | escalator | `openrouter/anthropic/claude-sonnet-5` | 0.5875 | authoritative escalation |
| `radial_momentum_flux` | name | 0 | primary | `openrouter/x-ai/grok-4.5` | 0.9125 | — |
| `radial_momentum_flux` | name | 1 | secondary | `openrouter/openai/gpt-5.6-luna` | 1.0 | quorum consensus |
| `radial_momentum_flux` | docs | 0 | primary | `openrouter/anthropic/claude-sonnet-5` | 0.9125 | — |
| `radial_momentum_flux` | docs | 1 | secondary | `openrouter/x-ai/grok-4.5` | 0.7875 | — |
| `radial_momentum_flux` | docs | 2 | escalator | `openrouter/openai/gpt-5.5` | 0.8125 | authoritative escalation |
| `poloidal_neutral_state_momentum_flux` | name | 0 | primary | `openrouter/x-ai/grok-4.5` | 0.825 | — |
| `poloidal_neutral_state_momentum_flux` | name | 1 | secondary | `openrouter/openai/gpt-5.6-luna` | 1.0 | — |
| `poloidal_neutral_state_momentum_flux` | name | 2 | escalator | `openrouter/anthropic/claude-sonnet-5` | 0.825 | authoritative escalation |

The accepted name continued through the ordinary documentation pools. Those
documentation cycles are shown because their costs are attributable to the
same scoped invocation, but they did not provide name-acceptance authority.

## LLMCost rows

| Identity | Row id | Phase / cycle | Model | USD |
|---|---|---|---|---:|
| `radial_ion_momentum` | `b63bea2a-41d1-5104-8aad-00bc8c48d333` | review name / c0 | `openrouter/x-ai/grok-4.5` | $0.085764 |
| `radial_ion_momentum` | `a0c9cf15-2888-5742-a0ca-102866f92260` | review name / c1 | `openrouter/openai/gpt-5.6-luna` | $0.010096 |
| `radial_ion_momentum` | `d53f6368-914c-5cc0-9d69-5a1bdd38bfc1` | review name / c2 | `openrouter/anthropic/claude-sonnet-5` | $0.144250 |
| `radial_momentum_flux` | `8abe38ea-b903-5b83-b74d-d19375127672` | review name / c0 | `openrouter/x-ai/grok-4.5` | $0.092410 |
| `radial_momentum_flux` | `bfb82ae4-0f8c-5e17-a113-636f2ba84397` | review name / c1 | `openrouter/openai/gpt-5.6-luna` | $0.004769 |
| `radial_momentum_flux` | `7ca02563-bb6b-5e5e-8362-83c5441e01d0` | generate docs | `openrouter/openai/gpt-5.6-luna` | $0.010416 |
| `radial_momentum_flux` | `4d9dee4a-d982-5fac-9178-ab20f2650d5b` | review docs / c0 | `openrouter/anthropic/claude-sonnet-5` | $0.073364 |
| `radial_momentum_flux` | `8958da6d-18f7-57cd-ab52-0c7fe4387516` | review docs / c1 | `openrouter/x-ai/grok-4.5` | $0.048532 |
| `radial_momentum_flux` | `94dd5a85-aea1-51ad-8898-b6864dd81e82` | review docs / c2 | `openrouter/openai/gpt-5.5` | $0.199925 |
| `poloidal_neutral_state_momentum_flux` | `3a0455f4-f14d-54bf-b29b-e039882cff82` | review name / c0 | `openrouter/x-ai/grok-4.5` | $0.043174 |
| `poloidal_neutral_state_momentum_flux` | `b833095c-6a0a-5db5-a617-c58a8a07b39e` | review name / c1 | `openrouter/openai/gpt-5.6-luna` | $0.004650 |
| `poloidal_neutral_state_momentum_flux` | `2ec3529e-6db6-543b-b18e-6c73dd51355e` | review name / c2 | `openrouter/anthropic/claude-sonnet-5` | $0.139354 |

The 12 attributable rows sum to **$0.856704**: $0.240110 for
`radial_ion_momentum`, $0.429416 for `radial_momentum_flux`, and $0.187178
for `poloidal_neutral_state_momentum_flux`. This is below the $150
authorization. Live LLMCost finishes at 27,489 rows and
$1,362.806839; subtracting the attributable rows gives the pre-node total
$1,361.950135.

## Integrity assertions

- Exactly one fresh name-review group exists for each identity.
- The two identities completed before continuation gained no additional
  Review or LLMCost row during the continuation.
- No identity spent a refinement attempt or acquired a `REFINED_FROM` change.
- All three identities are unclaimed after review.
- No name was accepted by direct mutation, rewording, or any route other than
  a fresh threshold-clearing quorum score.
- Non-acceptance is terminal evidence for this node, not authority to retry.

The canonical machine-readable receipt is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T140516310593-sgwi-ancestor-rescore/ancestor-rescore-receipt.json`.
Execution logs are `ancestor-rescore-run.log`,
`ancestor-rescore-continuation.log`, and `ancestor-rescore-receipt.log` in the
same run directory. The first log records the initial time-fence interruption;
the continuation log records the completed third draw and a post-review
attribution-query mismatch. The receipt log is the corrected read-only
attribution gate and exits 0.
