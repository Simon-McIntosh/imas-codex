# The flux-surface area derivative: the vendor-diverse seat fails differently

## Outcome

`equilibrium/time_slice/profiles_1d/darea_dpsi` — the radial derivative of the
cross-sectional area of a magnetic flux surface with respect to poloidal
magnetic flux — still does **not** have a standard name, and the WEST cut still
does **not** carry it. The governed escalation was exercised exactly once on
the vendor-diverse seat `openrouter/anthropic/claude-fable-5`; it emitted
grammar-invalid structured output, so no further seat was attempted.

The failure is narrower than the local composer's. Fable selected the correct
operator, `derivative_with_respect_to_poloidal_magnetic_flux_coordinate`, but
marked it `bare_prefix=True`. `StandardNameIR` refuses that combination because
this operator has no bare spelling. The focused run ended at exit 1 after
252.724276 seconds with four counted `generate_name` pool errors,
`names_composed=0`, `names_reviewed=0`, and `$4.356897` spent against the
node's `$10.00` command cap. The complete 141,540-byte terminal transcript is
stored at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260915T083636424567-n-swcr-the-area-derivative-is-named-on-a-diverse-seat/compose-fable.log`
(SHA-256 `b3ebf837f7458286f1a15e572891242e93f283eb6d4557376707bcbcd8ebfc9f`).

## Starting state (quoted)

The source was read from the live graph before any action:

```
status extracted with attempt_count 4 and produced_sn_id null
```

`compose_hint_status` was absent (no steering). The DD-grounded unit in the
graph is `Wb^-1.m^2` (dimensionally `m^2.Wb^-1`), and the source's enriched
description reads:

> Derivative of the cross-sectional area of a magnetic flux surface with
> respect to poloidal flux (ψ), dA/dψ. A geometric metric used in
> flux-surface-averaged transport calculations.

## Parse confirmation — the grammar is not the barrier

The target spelling and its toroidal sibling both parse cleanly under the
installed imas-standard-names 0.9.3:

```
TARGET  derivative_of_area_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate  parses: True
SIBLING derivative_of_area_of_flux_surface_with_respect_to_toroidal_flux_coordinate           parses: True
version: 0.9.3
```

So the barrier is not the vocabulary or grammar of the intended name; it is
that the composer never lands on this parseable form.

## Diagnosis of the failed attempts (quoted)

Every logged composition for this source ends in a grammar failure and a
`1/1` re-composition with expanded DD context. The compose log records the
three 0.9.3-era attempts before this node:

```
2026-09-15 00:11:19 … Pool generate_name: composition retry 1/1: 1 grammar failures (equilibrium/time_slice/profiles_1d/darea_dpsi), 0 token-reuse hits, editorial_retry=False — re-composing with expanded DD context
2026-09-15 00:22:20 … same
2026-09-15 00:27:23 … same
```

The composer (`local/deepseek-v4-flash`) reaches for an invented
`physical_base` token that does not exist in the ISN vocabulary, instead of
composing the derivative-operator form. The graph records the attempt as
genuinely absent since July:

```
VocabGap: token=rate_of_change_of_area_with_respect_to_poloidal_magnetic_flux
          category=absent  triage=genuine  first_seen=2026-07-28
```

The sibling `darea_drho_tor` demonstrates the exact-path outcome this source
needs: after five attempts under the same cap, plain composition produced
`derivative_of_area_of_flux_surface_with_respect_to_toroidal_flux_coordinate`
— read back from the graph with `name_stage=accepted` and `docs_stage=accepted`
— without any hint.

## The fifth and final attempt (guided, still failed)

Per the diagnosed-then-fix-then-compose sequence, the fifth attempt was not
spent blind. A compose hint carrying the exact parseable target string was set
on the source (`imas-codex sn source-hint`) and verified bound before the run:

```
compose_hint_status=open
compose_hint=derivative_of_area_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate
compose_hint_reason=<… radial derivative, unit m2.Wb-1, sibling toroidal cell accepted …>
```

The steering is injected verbatim into the generate prompt whenever
`compose_hint_status=='open'` (template `generate_name_dd.md` renders the
operator-steering block; mechanically confirmed by rendering the fragment and
by the claim query selecting the scalar fields onto the item). The focused run
then composed under the hint:

```
imas-codex sn run --focus equilibrium/time_slice/profiles_1d/darea_dpsi --skip-global-maintenance
```

The attempt — the fifth and final under `_MAX_COMPOSE_CLAIM_ATTEMPTS = 5` —
still failed the grammar round trip (09:21:40 compose log), produced no name,
and the run exited with `names_composed=0  names_reviewed=0`. The composer did
not adopt the steering even when handed the exact spelling.

## Final graph read-back

```
StandardNameSource dd:equilibrium/time_slice/profiles_1d/darea_dpsi
  attempt_count        : 5            (cap 5 — exhausted)
  status               : failed
  produced_sn_id       : null
  name_stage           : null
  docs_stage           : null
  compose_hint_status  : open         (never consumed — no product to steer)
  PRODUCED_NAME edges  : 0
```

No standard name exists for this quantity, so there are no reviewer scores,
no accepted-aggregate, and no documentation to report. The WEST cut's name,
docs, and validation gates all see an unnamed source and drop it.

## Open question for the review

The DD documentation for `darea_dpsi` describes a **radial** derivative of the
cross-sectional area with respect to psi, but the three accepted sibling
family members carry no radial qualifier:

* `derivative_of_volume_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate` — accepted, accepted
* `derivative_of_volume_of_flux_surface_with_respect_to_toroidal_flux_coordinate` — accepted, accepted
* `derivative_of_area_of_flux_surface_with_respect_to_toroidal_flux_coordinate` — accepted, accepted

The steering hint used the unqualified
`derivative_of_area_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate`
spelling, matching the family and the sibling the DD pairs it with. Whether
"radial" belongs in the name is staged for the review to judge against that
family convention; it is moot for this cut because no composed name survived to
be exported.

## Vendor-diverse escalation (2026-09-15)

The previous failure still reproduced before recovery:

```
status=failed  attempt_count=5  produced_sn_id=null
PRODUCED_NAME edges=0  compose_hint_status=open
```

The sibling control remained present and useful:
`derivative_of_volume_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate`
was `name_stage=accepted`, `docs_stage=accepted`, with one producing source.
This ruled out an empty graph read before the absence was treated as evidence.

The source was then released only through the governed CLI:

```
imas-codex sn retry --failed equilibrium/time_slice/profiles_1d/darea_dpsi \
  --reason "release the exhausted local-composer failure for one vendor-diverse composition attempt using the exact parseable sibling-family tail"

retried: 1 of 1 requested source(s)
  dd:equilibrium/time_slice/profiles_1d/darea_dpsi
```

The durable retry event is
`source-retry:8dbf7131-a22b-4301-83c1-d28104db544e`; it records
`previous_status=failed`, `previous_attempt_count=5`, and
`previous_error="compose claim-attempt cap reached"`. Read-back immediately
afterward showed `status=extracted` and `attempt_count=0`, with the exact source
hint still open.

The sole escalation run was:

```
imas-codex sn run \
  --focus equilibrium/time_slice/profiles_1d/darea_dpsi \
  --skip-global-maintenance \
  --compose-model openrouter/anthropic/claude-fable-5 \
  --time 20 --cost-limit 10
```

The routing receipt named the required model, direct OpenRouter route, and the
project-specific key source. The first failed composition then reported:

```
Value error, operator
'derivative_with_respect_to_poloidal_magnetic_flux_coordinate' has no bare
spelling; ... input_value={'kind': 'unary_prefix', ...,
'bare_prefix': True}
```

The same run logged five `composition retry 1/1` grammar-failure lines for the
exact source and ultimately refused successful completion after four counted
pool errors. The fifth claim had already reached the attempt cap; its late model
response lost the claim race and was ignored. Final graph read-back is again:

```
status=failed  attempt_count=5  produced_sn_id=null
PRODUCED_NAME edges=0  compose_hint_status=open
```

The intended identity is absent from the graph, while the accepted volume
sibling remains present. The CLI corroborates the absence:

```
imas-codex sn status --family derivative_of_area_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate
No family found for
'derivative_of_area_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate'
```

## Why a further retry is not a plan

The local seat invented an absent compound base. The vendor-diverse seat instead
found the correct operator and encoded it in a form the model schema explicitly
forbids. That is the measured outcome required by the escalation branch: the
model family changed, the failure mode changed, and no identity was minted.
Another seat would be an unplanned experiment after the fence explicitly says
to stop. The source remains terminal with its exact spelling hint intact for a
future, separately governed recovery.
