# The flux-surface area derivative: a model-side barrier, not the grammar

## Outcome

`equilibrium/time_slice/profiles_1d/darea_dpsi` — the radial derivative of the
cross-sectional area of a magnetic flux surface with respect to poloidal
magnetic flux — does **not** gain an accepted standard name, and the WEST cut
does **not** carry it. There is no composed name to export: the final,
exact-spelling-guided composition attempt under the five-attempt cap failed the
grammar round trip like every attempt before it, the source is terminal
(`status=failed`, `attempt_count=5`, `produced_sn_id=null`), and the focused
run exited with `names_composed=0` and `names_reviewed=0`.

This is the goal's second branch, reported in full below: **the grammar is not
the barrier, and the cause is visible, diagnosed, and resistant to the
in-scope fix** — the local composer seat repeatedly emits grammar-invalid
structured output for this source even when the prompt carries the exact,
parseable target spelling as operator steering.

Node LLM spend was `$0.00` — every composition call was the local
`deepseek-v4-flash` seat (cache `MISS`, `cost=$0.0000`); with no produced name
there were no paid review or documentation stages, so the campaign spend figure
is unchanged (`107.18` USD before, `107.18` after, against the `250.00`
ceiling).

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

## Why a further retry is not a plan

The in-scope fix (operator steering to the exact parseable spelling) is
verified injected and did not change the outcome; a sixth attempt would re-run
the identical experiment with no new information. The remaining lever — making
the composer produce grammar-valid structured output for this source — lives in
the composer seat/prompt rather than in the data path this node is scoped to,
and is visible to the fleet as a follow-on.
