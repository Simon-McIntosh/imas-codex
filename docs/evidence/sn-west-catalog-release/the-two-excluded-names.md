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
