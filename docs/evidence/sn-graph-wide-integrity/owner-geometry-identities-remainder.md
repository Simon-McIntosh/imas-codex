# Owner-geometry identity remainder

## Outcome

The authorized remainder produced one newly accepted identity and three
grammar-valid, freshly reviewed identities that did not clear the acceptance
threshold. Two requested identities remain absent after their single sanctioned
compose batch and were not retried.

| Final spelling | `name_stage` | `reviewer_score_name` | Fresh quorum | Attributable USD | Result |
|---|---|---:|---|---:|---|
| `toroidal_coordinate_at_beam_tracing_point` | `reviewed` | 0.7625 | authoritative escalation, 3 review rows in 1 group | 0.177789 | below threshold; not retried |
| `toroidal_coordinate_at_pellet_path_point` | `reviewed` | 0.8375 | authoritative escalation, 3 review rows in 1 group | 0.176596 | below threshold; not retried |
| `toroidal_coordinate_at_shattering_position` | `reviewed` | 0.8250 | quorum consensus, 2 review rows in 1 group | 0.061169 | below threshold; not retried |
| `toroidal_coordinate_of_pellet_fragment` | `accepted` | 1.0000 | quorum consensus, 2 review rows in 1 group | 0.066298 | accepted through the fresh quorum |
| `toroidal_coordinate_of_reflectometer_antenna` | absent | null | no identity existed to claim | 0.000000 | one compose batch emitted `toroidal_coordinate_of_diagnostic_antenna`; requested identity not retried |
| `toroidal_coordinate_of_shatter_cone` | absent | null | no identity existed to claim | 0.000000 | one compose batch collided with the superseded shattering-position predecessor; requested identity not retried |
| **Total** | **1 accepted / 3 reviewed / 2 absent** | — | **4 fresh review groups** | **0.481852** | **one compose batch, no refinement or rescore** |

The four reviewed identities reached their final stages only through the
ordinary name-review pool. No direct acceptance, Cypher `SET`, direct graph text
edit, refine pool, rescore, or second compose-model call was used.

## Pinned-grammar gate

The installed `imas-standard-names` dependency is `0.8.0rc66`. A fresh call to
`canonical_locus_check` returned an empty issue list for all six final spellings:

- `toroidal_coordinate_at_beam_tracing_point`
- `toroidal_coordinate_at_pellet_path_point`
- `toroidal_coordinate_at_shattering_position`
- `toroidal_coordinate_of_pellet_fragment`
- `toroidal_coordinate_of_reflectometer_antenna`
- `toroidal_coordinate_of_shatter_cone`

The three former `_of_` nodes are now superseded and retain their prior
quarantine diagnostics; their `_at_` successors are `validation_status=valid`
with no validation issues. The review scores above are negative quorum results,
not grammar refusals.

The compose log separately warned that the graph has no `GrammarToken` nodes
for rc66 and fell back to its rc9 token snapshot while classifying one token
miss. That stale graph-side vocabulary snapshot did not alter the pinned-package
`canonical_locus_check` result and was not repaired in this evidence-only node.

## Claim-timeout proof

The interrupted source was
`dd:spi/injector/shatter_cone/origin/phi`. Immediately before recovery it was
still `status=extracted`, `attempt_count=1`, hint `open`, with the original
claim token and `claimed_at=2026-08-20T17:20:28.918Z`. At the recovery gate its
measured claim age was **1,894 seconds**, greater than the configured ordinary
orphan-sweep timeout of **1,800 seconds**.

One invocation of the repository's ordinary
`standard_names.orphan_sweep._orphan_sweep_tick`, using the configured timeout,
reported `stale_token_source=1` and every other sweep count zero. The immediate
read-back showed `claim_token=null` and `claimed_at=null` while status,
attempt count, and open hint were preserved. No direct property edit was used.

The one-shot compose processor later released shatter cone normally after its
lifecycle collision. Three unbound reflectometer batch members were also
released through the token-verified
`release_generate_name_failed_claims` error-recovery operator; no claim remains
on any of the nine source paths in this report.

## Exact invocations

All Python and CLI calls reused the repository's root environment and loaded the
repository environment file. Long output was captured in the named logs listed
below.

```text
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync --env-file /home/ITER/mcintos/Code/imas-codex/.env python /home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T173937650741-sgwi-owner-geometry-identities-remainder/stage.py

env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync --env-file /home/ITER/mcintos/Code/imas-codex/.env python /home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T173937650741-sgwi-owner-geometry-identities-remainder/release_timeout.py

env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync --env-file /home/ITER/mcintos/Code/imas-codex/.env python /home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T173937650741-sgwi-owner-geometry-identities-remainder/compose_once.py

env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync --env-file /home/ITER/mcintos/Code/imas-codex/.env imas-codex sn run --name toroidal_coordinate_at_beam_tracing_point --name toroidal_coordinate_at_pellet_path_point --name toroidal_coordinate_at_shattering_position --name toroidal_coordinate_of_pellet_fragment --only review_name --names-only --skip-global-maintenance --cost-limit 4.0 --time 8
```

`stage.py` dry-ran all three `sn edit --rename` equivalents before applying
them, then dry-ran and applied the five semantic detaches and five exact-source
hints through the canonical repository functions. `compose_once.py` used the
sanctioned explicit-path pool adapter to claim exactly six paths and called
`process_generate_name_batch` exactly once. Its first launch stopped before any
model call because the adapter result lacked the processor's `id` alias; the
token-verified failed-claim recovery released exactly those six claims, and the
corrected launch made the single compose-model call. This bookkeeping failure
added a claim sequence but no candidate, `LLMCost`, or model attempt.

A preliminary six-name review command was refused before pool startup because
the reflectometer-antenna and shatter-cone identities were absent. It wrote no
review or cost rows. The four-name invocation shown above is the only review run
that executed; it completed with `stop_reason=no_eligible_work`, four reviewed
names, and no refinement.

## Spend and postflight

The local `hosted_vllm/deepseek-v4-flash` compose call cost **$0.000000**. The
four fresh quorums wrote 10 `LLMCost` rows totaling **$0.481852**. Adding this to
the coordinator's prior measured session spend of **$1.387862** gives running
session spend of **$1.869714 / $150.000000**, leaving **$148.130286**.

Global counters moved as follows during this node:

- `LLMCost`: 27,528 to **27,538** (`+10`), exactly the review cost rows;
- `StandardNameChange`: 7,676 to **7,687** (`+11`), from the sanctioned renames
  and semantic detaches;
- `toroidal_angle_of_measurement_position`: still `accepted` at 0.95625, with
  **28** live producers after the four reflectometer sources were removed.

The pellet-fragment source now realizes the accepted owner-specific identity.
The four reflectometer sources did not realize the requested identity: one
realizes the model-chosen drafted `toroidal_coordinate_of_diagnostic_antenna`
and three remain extracted with their hints open. The shatter-cone source
remains extracted, unclaimed, and hint-open. Those outcomes are retained as the
one sanctioned attempt's result rather than repaired or retried here.

## Durable evidence

Runtime artifacts live under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T173937650741-sgwi-owner-geometry-identities-remainder/`:

- `baseline.json`, `postcleanup.json`, and their logs: fresh graph, grammar,
  source, counter, review, and spend censuses;
- `staging-receipt.json` and `stage.log`: rename, detach, and hint previews plus
  applies;
- `claim-timeout-receipt.json` and `claim-timeout.log`: configured-timeout proof;
- `compose-once-receipt.json`, `compose-once-retry.log`, and
  `compose-once.log`: the single model batch and the pre-model shape failure;
- `review-present.log`: the only executed fresh quorum run;
- `unbound-compose-claim-release.json` and its log: token-verified cleanup of
  the three unbound reflectometer claims;
- `grammar-version.log`: pinned installed grammar version.
