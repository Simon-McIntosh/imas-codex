# Corrected-grammar fresh draws

## Outcome

Exactly **7 identities received exactly 1 fresh draw each** against the sole
active graph grammar snapshot, **ISN 0.8.0rc66 with 956 tokens**. Five existing
identities received one new ordinary name-review quorum group apiece. The two
absent identities received one exact corrected-grammar compose opportunity
apiece in the same single batch; neither materialized, so their retained result
is absence and no review group could follow. No identity was submitted again
after its draw.

The acceptance threshold was **0.85**. Four identities cleared it, one remained
below it, and two remained absent. Attributable provider spend was
**USD 0.563595**, entirely from 12 name-review cost rows, against the authorized
USD 25 ceiling. The local compose attempt cost USD 0. No claim remains.

| Identity | Before | Fresh draw | Result against 0.85 | Attributable USD | Draw count |
|---|---:|---|---|---:|---:|
| `toroidal_coordinate_at_beam_tracing_point` | reviewed / 0.7625 | quorum group `4e75be3f-6ce7-42ef-973b-aaf0dcc781f2` | **accepted / 0.9125** | 0.071428 | 1 |
| `toroidal_coordinate_at_pellet_path_point` | reviewed / 0.8375 | quorum group `d94e3dbf-ae6e-4c4f-bbb1-26e0ace3428c` | **accepted / 0.96875** | 0.036515 | 1 |
| `toroidal_coordinate_at_shattering_position` | reviewed / 0.8250 | quorum group `306c2a98-50ae-432c-bc8b-8bffadfbcdb2` | **accepted / 0.89375** | 0.064954 | 1 |
| `toroidal_coordinate_of_reflectometer_antenna` | absent | one exact compose draw over three still-extracted reflectometer centre sources | **absent / no score** | 0.000000 | 1 |
| `toroidal_coordinate_of_shatter_cone` | absent | one exact compose draw over `spi/injector/shatter_cone/origin/phi` | **absent / no score** | 0.000000 | 1 |
| `radial_ion_momentum` | reviewed / 0.5875 | quorum group `b3a590bb-3b31-43c6-afcb-3ba4fc7c6479` | reviewed / **0.575**, below threshold | 0.223927 | 1 |
| `poloidal_neutral_state_momentum_flux` | reviewed / 0.8250 | quorum group `4d2be5ba-b67a-4a42-80df-bc5805eb4441` | **accepted / 0.8875** | 0.166771 | 1 |
| **Total** | **5 reviewed + 2 absent** | **5 quorum groups + 2 single compose draws** | **4 accepted + 1 below + 2 absent** | **0.563595** | **7** |

The two fold ancestors therefore split cleanly on the fresh evidence:
`poloidal_neutral_state_momentum_flux` now clears at 0.8875, while
`radial_ion_momentum` scores 0.575 and remains reviewed. The latter is a
first-class negative result; it was not reworded, refined, rescored again, or
accepted through another route.

## Absent-identity draw

Immediately before the absent-identity draw, three reflectometer centre
sources and the shatter-cone origin source were extracted, unclaimed, and held
open exact hints for the two requested identities. Each source's attempt count
advanced by exactly one in one invocation. The batch ran only after an in-call
gate confirmed active grammar `0.8.0rc66` and exactly 956 graph tokens.

The corrected vocabulary changed the observable result but did not produce a
persistable batch:

- the reflectometer side proposed
  `toroidal_coordinate_of_diagnostic_component_center`, which strict grammar
  rejected rather than substituting for the requested owner-specific identity;
- the shatter-cone side emitted
  `toroidal_coordinate_of_shatter_cone`, but the same batch also reported a
  token-miss classification and the rich writer rejected the exact-claim
  winner set atomically.

Consequently neither requested identity was written and no partial source
binding survived. This was the single authorized draw, so it was not rerun.
The four failed claims were released once through the token-verified
`release_generate_name_failed_claims` recovery operator; all four sources are
again extracted with open hints and null claim tokens. Recovery made no model
call and is not a retry.

## Integrity and accounting proof

- Baseline and postflight each returned exactly one active
  `ISNGrammarVersion`: `0.8.0rc66`; both counted exactly 956 matching
  `GrammarToken` nodes.
- Each of the five materialized identities gained exactly one new name-review
  group. The three accepted position identities used two-seat quorum consensus;
  both ancestors used the ordinary third-seat authoritative escalation.
- The two absent identities gained no `StandardName` or review node. Their four
  exact sources each advanced by one compose attempt, all in one batch.
- `LLMCost` moved from 27,619 to 27,631: exactly 12 rows, all phase
  `review_name`, totaling USD 0.563595. `StandardNameChange` remained 7,754.
- Final state has zero claims on every named identity and every source in the
  absent draw. There was no direct acceptance, raw graph text mutation,
  refinement, second rescore, or second compose invocation.
- The machine gate reports: 7 identities, 7 draws, 5 fresh quorum groups,
  2 single compose-draw identities, 4 accepted, 1 below threshold, 2 absent,
  0 claims remaining, and 0 retries after a draw.

## Durable runtime evidence

All runtime evidence is under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T143734253079-freshdraw/`:

- `baseline.json` and `baseline.log`: pre-draw identity, grammar, and counter
  census;
- `review-five.log`: the only review invocation, including all five quorum
  outcomes;
- `after-five.json` and `after-five.log`: review-group and pre-compose source
  state;
- `compose-absent-once.log`: the only absent-identity compose invocation and
  its atomic refusal;
- `failed-compose-release.json` and `failed-compose-release.log`: exact
  four-claim token-verified recovery;
- `final.json` and `final.log`: final identities, review cycles, grammar,
  source states, spend, and claims;
- `verification.json` and `verification.log`: quantitative done-when gate,
  exit 0.
