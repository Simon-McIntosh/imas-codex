# The two excluded WEST names: rotation evidence

## Outcome

The WEST cut contained 220 candidates, of which 218 were published. Exactly
two were excluded, so the cut arithmetic remains `220 = 218 + 2` with residue
zero.

This scoped recovery reached the required terminal outcomes:

* `hot_neutral_temperature` now has an accepted successor,
  `hot_neutral_temperature_at_plasma_boundary`, at score `0.9875`.
* `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`
  now has a grammar-valid successor, `inner_hard_xray_peak_width`, but its
  complete review quorum remained below threshold. Its third rotation was
  spent and the successor is now `name_stage=exhausted` at score `0.625`.

No quarantine was cleared, no review row was deleted, no candidate was cut,
and no request or signed manifest operation was attempted. The compose seat
was not invoked. The review-only and refine-only commands used the remote
review/refine path.

## Instrument and bounded scope

The live graph was reached on the login node through the project resolver. The
observed profile was `bolt://98dci4-gpu-0002:7687`; the compute-node service
was reached by its hostname, not by localhost or a tunnel. The client was
`GraphClient.from_profile()`.

Before asserting a missing property, this bounded read was run against the
hot identity:

```cypher
MATCH (sn:StandardName {id: $id})
RETURN keys(sn) AS property_keys
LIMIT 1
```

The returned key set includes `id`, `name_stage`, `docs_stage`,
`validation_status`, `reviewer_score_name`, `review_quorum_shortfall`,
`validation_issues`, `edit_status`, and the source/review bookkeeping fields.
It does not include a `name` key. It also does not include a durable
`refine_attempts` key on the original hot node, so the effective attempt value
is the code's fallback:
`coalesce(sn.refine_attempts, coalesce(sn.chain_length, 0))`.
The second identity's key set explicitly includes `edit_status`,
`edit_mode`, `name_hint`, and `validation_issues`, but its original
`refine_attempts` value was also null. All reads below were bounded to the
named identities and completed within the ten-second per-query ceiling.

The configured values in force were `DEFAULT_MIN_SCORE=0.85` from
`imas_codex/standard_names/defaults.py` and `refine-rotations=3` from the
project configuration. The refine predicate in `graph_ops.py` requires:

```text
name_stage='reviewed'
reviewer_score_name IS NOT NULL
reviewer_score_name < min_score
coalesce(refine_attempts, coalesce(chain_length, 0)) < rotation_cap
name_stage NOT IN ['superseded', 'exhausted', 'contested']
not a capped pinned rename resubmission
origin <> 'derived'
review_quorum_shortfall IS NULL
```

The name-review predicate requires `name_stage='drafted'`,
`validation_status='valid'`, a substantive description, and non-derived
origin. These predicates explain the initial refusal rather than treating the
zero processed count as a successful guard by itself.

## Starting state and exact claim refusal

### `hot_neutral_temperature`

Before this node's changes, the identity had these values:

| Field | Starting value |
| --- | --- |
| Producing DD paths | `spectrometer_visible/channel/isotope_ratios/isotope/hot_neutrals_temperature`; `spectrometer_visible/channel/polarization_spectroscopy/temperature_hot_neutrals` |
| Producer statuses | `attached`; `composed` |
| `name_stage` | `reviewed` |
| `docs_stage` | `accepted` |
| `status` | `draft` |
| `validation_status` | `valid` |
| `validation_issues` | `[]` |
| `reviewer_score_name` | `0.30` |
| `refine_attempts` | `null`; effective value `0` |
| Rotation budget | `0/3` |
| `edit_status` | `null` |
| Name-axis reviews | 5, all score `0.30` after the earlier zero-cost rescore |
| Newest name review | `2026-09-14T09:12:20.463703Z` |
| `review_quorum_shortfall` | `fewer reviewer seats scored than the chain defines (method=semantic_similarity_gate)` |
| Resolution method | `semantic_similarity_gate` |

The exact-name refine-only run used scope
`f3b3ec1a-3765-4fe2-9cc3-8cb73b1115b3` and returned exit 0 after the normal
empty-claim shutdown. Its measured pending count was `refine_name=0` and it
processed `0` names. Hot would otherwise satisfy the score and budget tests:
`0.30 < 0.85` and `0 < 3`. It was excluded solely because
`review_quorum_shortfall IS NOT NULL`. That is why its untouched budget could
not buy the rewrite that the below-threshold score ordinarily requests.

The source description is:

> Translational kinetic temperature, expressed as energy per particle, of the
> energetic neutral-atom component in the plasma edge or scrape-off layer.

The first complete-name steering attempt proposed
`hot_neutral_internal_state_temperature`. The attachment guard refused and
rolled it back because both producing paths are species-level while that
spelling is state-resolved. It therefore produced no scored candidate and no
durable source transition. The next candidate,
`hot_neutral_temperature_at_plasma_edge`, passed the lightweight edit dry run
but failed the full admission validator with:

```text
parse_error: grammar round-trip failed for hot_neutral_temperature_at_plasma_edge
```

It was preserved as a quarantined intermediate and then superseded by the
grammar-valid candidate below; it received no review score because the
name-review predicate excludes quarantined names.

### `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`

Before steering, the identity had these values:

| Field | Starting value |
| --- | --- |
| Producing DD path | `hard_x_rays/emissivity_profile_1d/half_width_internal` |
| Producer status | `attached` |
| Description | Inward half-width of the hard X-ray emissivity peak in normalized toroidal flux coordinate. |
| `name_stage` | `drafted` |
| `docs_stage` | `pending` |
| `status` | `draft` |
| `validation_status` | `quarantined` |
| Quarantine reason | `parse_error: grammar round-trip failed for inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` |
| `reviewer_score_name` | `null` |
| `refine_attempts` | `null`; effective value `0` |
| Rotation budget | `0/3` before the replacement chain |
| `edit_status` | `open` |
| Name-axis reviews | `0` |
| `review_quorum_shortfall` | `null`; no review had run |
| Resolution method | `null` |

The source is one scalar, `half_width_internal`; it is not a fold of two
independent quantities. The original spelling was therefore correctly left
quarantined. A grammar-valid, physics-preserving steered replacement was
`inner_hard_xray_peak_width`, which keeps the inner-side hard-X-ray peak width
quantity without placing the normalized coordinate and the measured quantity
in one invalid construction.

## Rotation trajectory

The ordinary scoped run was attempted first and made no claims, as reported
above. Because the two initial predicates structurally withheld refine, the
sanctioned complete-name steered-edit route was used without the unavailable
compose seat. The edits were staged with `--stage-only`; the review-only runs
then claimed the drafted, valid candidates.

### Hot identity

| Candidate or identity | Outcome and score | Effective attempts | Evidence |
| --- | --- | --- | --- |
| `hot_neutral_temperature` | Starting score `0.30`; semantic-gate shortfall; no new refine score | `0/3` before steering | Below threshold but withheld by non-null `review_quorum_shortfall` |
| `hot_neutral_internal_state_temperature` | Rejected before persistence; score `null` | No charge persisted | Attachment guard: both source paths are species-level, candidate is state-resolved |
| `hot_neutral_temperature_at_plasma_edge` | Quarantined; score `null` | `1/3` | Full admission `parse_error` on grammar round-trip; no review claim |
| `hot_neutral_temperature_at_plasma_boundary` | Accepted; aggregate score `0.9875` | `1/3` | Two name reviews at `0.975` and `1.000`; `resolution_method=quorum_consensus` |

The accepted successor currently reads `name_stage=accepted`,
`validation_status=valid`, `validation_issues=[]`, `edit_status=applied`,
and `review_quorum_shortfall=null`. Its two producing paths are still the two
species-level paths listed above, with statuses `attached` and `composed`.
The review receipt was:

```text
persist_reviewed_name: hot_neutral_temperature_at_plasma_boundary
  → name_stage=accepted (score=0.988, rotations=1/3, chain=2)
review_name: hot_neutral_temperature_at_plasma_boundary
  → accepted (score=0.988, cycles=2, method=quorum_consensus)
```

The graph stores the precise aggregate as `0.9875`; the receipt rounds it to
`0.988`.

### Inner identity

| Candidate or identity | Outcome and score | Effective attempts | Evidence |
| --- | --- | --- | --- |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | Starting quarantine; score `null` | `0/3` before steering | Original grammar round-trip `parse_error`; no name reviews |
| `inner_hard_xray_peak_width`, first review | Reviewed at `0.7500` | `2/3` | Three scores `0.7125`, `0.875`, `0.750`; complete `authoritative_escalation` quorum |
| `inner_hard_xray_peak_width`, ordinary refine pass | Same spelling resubmitted; no new candidate and no LLM spend | `2/3` | Refine claimed the pinned rename and logged `pinned rename ... not rewritten — resubmitted` |
| `inner_hard_xray_peak_width`, second review | Exhausted at `0.6250` | `3/3` | Three scores `0.6875`, `0.750`, `0.625`; complete `authoritative_escalation` quorum |

The final inner successor currently reads `name_stage=exhausted`,
`validation_status=valid`, `validation_issues=[]`, `edit_status=exhausted`,
`review_resubmit_count=1`, `docs_stage=pending`, and
`review_quorum_shortfall=null`. Its one producing path remains
`hard_x_rays/emissivity_profile_1d/half_width_internal` with status
`attached`. The final receipt was:

```text
persist_reviewed_name: inner_hard_xray_peak_width
  → name_stage=exhausted (score=0.625, rotations=3/3, chain=2)
review_name: inner_hard_xray_peak_width
  → exhausted (score=0.625, cycles=3, method=authoritative_escalation)
```

Every review in both inner cycles reached a complete quorum. The final
exhaustion is therefore not a quorum-shortfall artifact: the exact refusing
predicate is the score `0.625 < 0.85` combined with the spent rotation budget
`3/3`. The candidate spelling is grammar-valid, so the remaining refusal is
semantic review quality rather than the original grammar quarantine.

## WEST cut and spend consequences

Neither final successor would be carried by a fresh WEST cut at this instant:

* The hot successor is name-accepted and valid, but its rename reset the docs
  axis to `docs_stage=pending`; the cut requires acceptance on both axes.
* The inner successor is `name_stage=exhausted`, with `docs_stage=pending`, so
  it fails the name-acceptance requirement as well as the docs requirement.

The hot name is now a viable follow-on once its documentation axis is
reviewed. The inner name is not a candidate for this cut: its exact recorded
terminal reason is exhaustion at score `0.625` after `3/3` rotations. The
original quarantine remains in the superseded history and was not cleared.

The campaign ledger was measured before and after the node's paid work:

| Measurement | LLMCost rows | Total |
| --- | ---: | ---: |
| Before this node | 1,692 | `$101.638907` |
| Before the final inner refine/review sequence | 1,709 | `$102.610250` |
| After this node | 1,712 | `$102.872694` |
| Measured node delta | 20 rows | `$1.233787` |

The delta is below the authorized `$25.00` node ceiling. It includes the
remote review work and no compose-seat spend. The final campaign total is
`$102.872694` against the authorized `$250.00` campaign ceiling.

## Receipts and follow-ons

The ordinary commands and receipts are recorded in:

```text
/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log
```

The log contains the initial exact-name refusal (`refine_name=0`), the hot
acceptance receipt, the inner `0.7500` review, the pinned-rename resubmission,
and the final `0.625` exhaustion receipt. The review-only command exited 0;
the final refine/review command exited 1 only because its five-minute time
limit fired while the in-flight review was finishing, and the receipt and
graph write landed during the granted shutdown grace. The graph readback above
is the authoritative post-state.

The scoped worker also surfaced pre-existing, out-of-scope maintenance
findings: 22 live names without a recoverable producing source, 7 names still
fed by DD paths absent from the current DD, and the DD-gap evidence writer's
rejection of a `reference_evidence_repaired` disposition field. These were not
triaged or repaired here. The inline hot attempt also stalled in that global
maintenance surface before its receipt and was aborted; its graph mutation was
read back, and the subsequent staged edit completed the governed route.

## Follow-on steering of the exhausted X-ray width

The live baseline for `inner_hard_xray_peak_width` was re-read before any
attachment. It remains bound through `PRODUCED_NAME` to
`hard_x_rays/emissivity_profile_1d/half_width_internal`, with the following
state:

| Field | Value |
| --- | --- |
| `name_stage` | `exhausted` |
| `refine_attempts` | `3/3` |
| `docs_stage` | `pending` |
| `status` | `draft` |
| `validation_status` | `valid` |
| `validation_issues` | `[]` |
| `unit` | `1` |
| `reviewer_score_name` | `0.625` |
| Name-axis review rows | `6` |
| Prior aggregates | `0.750`, then `0.625` |
| `edit_status` | `exhausted` |

The runtime rename eligibility probe was run first with a grammar-valid
probe spelling and `--scope self`. It returned:

```text
DRY RUN sn edit: inner_hard_xray_peak_width  mode=rename axis=name
scope=only_self entry=review_name
  would carry 1 producing source(s) to
  'inner_hard_xray_peak_width_at_measurement_position'
  would rename 'inner_hard_xray_peak_width' →
  'inner_hard_xray_peak_width_at_measurement_position'
```

This confirms at runtime that an exhausted identity is admitted by the
sanctioned rename vehicle; `_RENAME_ELIGIBLE_STAGES` includes `exhausted`.
The probe was dry-run only and was not attached because that placeholder is not
the proposed physics correction.

The coordinator's proposed candidate was then tested through the same
sanctioned dry-run:

```text
imas-codex sn edit inner_hard_xray_peak_width \
  --rename inner_half_width_of_hard_xray_emissivity_peak \
  --scope self --dry-run
```

It was refused before any graph write:

```text
BLOCKED
new name fails ISN grammar round-trip: parse failed: ParseError: residue
'inner_half_width_of_hard_xray_emissivity_peak' does not match any
physical_base or geometry_carrier; nearest candidates: (none)
```

The fully specified predecessor,
`inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`,
already carried the inner qualifier, normalized toroidal-flux coordinate,
hard-X-ray emissivity peak, and half-width, but its recorded admission failure
was likewise:

```text
parse_error: grammar round-trip failed for
inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width
```

The two attempted complete semantic constructions therefore fail at the same
closed-vocabulary boundary. The proposed spelling improves the plain-language
semantic coverage relative to `inner_hard_xray_peak_width` by restoring
`half_width` and `emissivity_peak`, while leaving the normalized coordinate to
the authoritative description. The current ISN grammar has no registered
`half_width` physical base or matching geometry carrier for that construction,
so the candidate cannot be reviewed safely. No rename was attached, no review
row was created, and campaign spend remained `$102.872694` across 1,712
`LLMCost` rows.

The final graph state is unchanged from the baseline: the identity remains
`name_stage=exhausted`, `refine_attempts=3/3`, `docs_stage=pending`,
`validation_status=valid`, and `edit_status=exhausted`, with its score at
`0.625`. The WEST cut still does not carry it. This is an ISN vocabulary
finding requiring an upstream grammar decision or vocabulary addition, not a
reason to clear the recorded quarantine history or spend another no-op review.

## Latest steered successor attempt: validation blocks the admitted spelling

This later bounded attempt re-read the remaining identity before attaching a
new proposal. The starting state was:

| Field | Starting value |
| --- | --- |
| Identity | `inner_hard_xray_peak_width` |
| Producer | `hard_xrays/emissivity_profile_1d/half_width_internal` |
| `name_stage` | `exhausted` |
| `reviewer_score_name` | `0.625` |
| Prior recorded aggregates | `0.750`, then `0.625` |
| `refine_attempts` | `3/3` |
| `docs_stage` | `pending` |
| `status` | `draft` |
| `validation_status` | `valid` |
| `validation_issues` | `[]` |
| `unit` | `1` |
| `edit_status` | `exhausted` |

The DD source is one scalar, not a fold of two quantities. Its enriched source
description reads:

> Internal (towards magnetic axis) half-width of the hard X-ray emissivity peak
> in normalised toroidal flux coordinate (ρ_tor_norm). Characterizes the inward
> radial extent of the emissivity peak from fast electrons.

The physics reason supplied to the edit was that the current generic `width`
omits the half-width distinction, changing the implied magnitude by a factor of
two, and omits the emissivity-profile locus. The proposed successor restores
`half_width` and binds the quantity to the emissivity peak. The unit `1` and the
description carry the normalized-coordinate context, so that coordinate is not
duplicated as another identity segment.

### Runtime dry-run and attachment

The required dry-run was run before any write:

```text
DRY RUN sn edit: inner_hard_xray_peak_width  mode=rename axis=name
scope=only_self entry=review_name

Actions:
  -  would carry 1 producing source(s) to
'inner_hard_xray_half_width_of_emissivity_peak'
  -  would rename 'inner_hard_xray_peak_width' →
'inner_hard_xray_half_width_of_emissivity_peak'
```

The dry-run admitted the exhausted identity at `scope=only_self` and confirmed
one producing source. The real scoped attachment returned:

```text
APPLIED sn edit: inner_hard_xray_peak_width  mode=rename axis=name
scope=only_self entry=review_name
  - verified 1 producing source(s) on
'inner_hard_xray_half_width_of_emissivity_peak'
  - renamed 'inner_hard_xray_peak_width' →
'inner_hard_xray_half_width_of_emissivity_peak', entering name review
  (edit_status=open, run_id=sn-edit-20260914T161947Z)
```

No sibling was edited. The source readback for the new identity confirms one
`PRODUCED_NAME` edge and `dd_path=hard_xrays/emissivity_profile_1d/half_width_internal`,
with `dd_unit=1`, `dd_version=4.1.0`, and the enriched description quoted above.

### Grammar and review result

The active graph grammar is `ISNGrammarVersion.version=0.9.3`. A bounded query
for `GrammarToken.value='emissivity_peak'` returns the token under the
`geometry` and `position` segments (and also the `path` projection); querying
`t.token` would be the wrong instrument because the graph stores the spelling in
`value`. The attached candidate reads back with `grammar_parse_version=0.9.3`
and `physical_base=half_width`, `geometry=emissivity_peak`, establishing that
the ISN grammar round-trip itself succeeds.

The required scoped review command was:

```text
imas-codex sn run --name inner_hard_xray_half_width_of_emissivity_peak \
  --only review_name --skip-global-maintenance -c 15 -t 20
```

The run bypassed global maintenance, scoped the population to one name, and
exited with `review_name processed=0`, `in_flight=0`, `error_count=0`, and
`$0.0000` review spend. It did not claim the candidate because review
eligibility requires both `name_stage=drafted` and `validation_status=valid`;
the attached candidate had been quarantined by the full validator before the
review pool could claim it. No name-axis review rows were created for this
candidate, so there are no per-reviewer scores or aggregate to report for it.

The exact current validation issues are:

```text
[semantic] inner_hard_xray_half_width_of_emissivity_peak: WARNING - dimensionless unit '1' on physical quantity 'half_width' is unexpected. Quantities like 'half_width' normally carry SI units. Use '1' only for true dimensionless quantities (ratios, coefficients, counts).
[canonical] audit:canonical_locus_check: name 'inner_hard_xray_half_width_of_emissivity_peak' has field-evaluation structure but uses intrinsic-geometry relation '_of_'. Rewrite as 'inner_hard_xray_half_width_at_emissivity_peak'.
audit:canonical_locus_check: name 'inner_hard_xray_half_width_of_emissivity_peak' has field-evaluation structure but uses intrinsic-geometry relation '_of_'. Rewrite as 'inner_hard_xray_half_width_at_emissivity_peak'.
```

The semantic unit message is advisory; the blocking predicate is the two
`canonical_locus_check` entries requiring `_at_` rather than `_of_`. This is a
runtime contradiction to the assumption that grammar admission alone made the
candidate reviewable: the grammar accepts `emissivity_peak` and the candidate
round-trips, but the codex canonical-locus validator rejects the relation for a
field-evaluation name.

### Final readback and verdict

The new identity now reads:

| Field | Final value |
| --- | --- |
| Identity | `inner_hard_xray_half_width_of_emissivity_peak` |
| `name_stage` | `drafted` |
| `edit_status` | `open` |
| `edit_mode` | `rename` |
| `validation_status` | `quarantined` |
| `validation_issues` | 1 advisory unit warning plus 2 identical canonical-locus refusals |
| `reviewer_score_name` | `null` |
| Name-axis review edges | `0` |
| `refine_attempts` | `3/3` inherited from the exhausted predecessor |
| `docs_stage` | `pending` |
| Producer | one `PRODUCED_NAME` edge for `hard_xrays/emissivity_profile_1d/half_width_internal` |

The exact goal was not reached: the candidate is neither accepted nor a scored
rejection. It is blocked by the current validation predicate
`canonical_locus_check(... uses intrinsic-geometry relation '_of_' ...)`, which
requires the spelling `inner_hard_xray_half_width_at_emissivity_peak`. Clearing
this quarantine or deleting the recorded candidate is forbidden by the
identity's spend history, and this node does not attempt a second spelling.

The WEST cut would **not** carry the identity now. Its `name_stage` is only
`drafted`, its `validation_status` is `quarantined`, and its `docs_stage` is
`pending`; the cut requires an accepted, valid name and accepted documentation.
The original exhausted identity remains preserved in history and no other
formerly excluded identity was touched.

Campaign spend was `$103.60` before and `$103.60` after. The stage-only edit and
the zero-claim review made no LLM calls and spent `$0.00` against the `$15.00`
node ceiling and `$250.00` campaign ceiling.
