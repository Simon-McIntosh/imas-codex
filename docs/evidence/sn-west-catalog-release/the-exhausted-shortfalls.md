# Exhausted WEST shortfalls: measured recovery and barriers

## Outcome

The three exhausted identities were read from the live graph before any
action. One reached a publishable state through the sanctioned scoped edit,
name review, documentation generation, documentation review, and
documentation refinement path. Two remain excluded, for different exact
reasons:

| Identity | Result | WEST cut |
| --- | --- | --- |
| `neutral_pressure` -> `neutral_gas_pressure_of_gauge` | Successor accepted at name score `0.98125` and documentation score `0.875`; validation is `valid` and the DD producer is attached | Yes, by the current name/docs/validation gates |
| `distance_of_antenna_strap` | Still `name_stage=exhausted` at `0.7125`, `refine_attempts=3/3`; dry-run refused because `gap_of_antenna_strap` already exists | No |
| `accumulated_total_coolant_absorbed_energy_of_calorimetry_component` | Still `name_stage=exhausted`, `validation_status=quarantined`; `quarantine_reason` and `validation_issues` are both null, and `refine_stop_reason=grammar_invalid` | No |

No identity was deleted, no quarantine was cleared, no review row was
rewritten, and no flush or candidate cut was attempted. The compose seat was
not invoked. The graph was read on the login node through the project client,
which resolves the compute-node service directly at
`bolt://98dci4-gpu-0002:7687`.

## Starting census and instrument

The live graph census was bounded to the three named identities and their
direct producer and review relationships. The graph key-set probe returned a
`StandardName` property set containing `id`, `name_stage`, `docs_stage`,
`status`, `validation_status`, `reviewer_score_name`, `validation_issues`,
`edit_status`, `refine_attempts`, and review bookkeeping fields. There is no
`name` property: the identity key is `id`, and querying a nonexistent property
would silently return zero rows. Producer paths are properties of
`StandardNameSource`, reached by incoming `(:StandardNameSource)-[:PRODUCED_NAME]->(sn)`
edges.

The starting graph values were:

### `neutral_pressure`

* `name_stage=exhausted`, `reviewer_score_name=0.6875`, and
  `reviewer_scores_name={"grammar":20,"semantic":8,"convention":14,"completeness":13}`.
* `refine_attempts=3/3`, `docs_stage=accepted`, `status=draft`,
  `validation_status=valid`, and `validation_issues=[]`.
* Producer: `dd:barometry/gauge/pressure`, with the enriched source
  description “Neutral gas pressure measured by a barometry gauge within the
  vacuum vessel or surrounding vacuum system. Characterizes local gas
  inventory and vacuum conditions relevant to plasma fuelling, recycling, and
  neutral gas modelling.”
* The stored identity description was the wrong referent: “Neutral pressure
  is the scalar kinetic pressure of a neutral-particle population, defined as
  one third of the trace of its central second velocity moment. It is the
  general neutral-pressure quantity before any thermal, fast, or internal-state
  partition is specified.”
* The five current name-review rows scored `0.625`, `0.600`, `0.5625`,
  `0.725`, and `0.7875`; their dimension records range from grammar `16` or
  `20`, semantic `4`–`12`, convention `12`–`20`, and completeness `8`–`15`.
  The full reviewer commentary identifies the leaf as an instrument reading
  from a set of gauges, not a kinetic neutral-particle moment, and distinguishes
  it from `total_neutral_pressure`, `thermal_neutral_pressure`,
  `neutral_internal_state_pressure`, and
  `total_neutral_internal_state_pressure`.

### `distance_of_antenna_strap`

* `name_stage=exhausted`, `reviewer_score_name=0.7125`, and
  `reviewer_scores_name={"grammar":20,"semantic":9,"convention":18,"completeness":10}`.
* `refine_attempts=3/3`, `docs_stage=pending`, `status=draft`,
  `validation_status=valid`, and `validation_issues=[]`.
* Producer: `dd:ic_antennas/antenna/module/strap/distance_to_conductor`,
  whose enriched description says: “Distance from an ICRH antenna strap to
  the conducting wall or conductor located behind it. This geometric parameter
  influences the antenna loading resistance and RF coupling efficiency to the
  plasma.”
* The identity description is already specific: “Distance from the antenna
  strap rear surface to the conducting wall behind it.”
* The current review rows are `0.675` (`20/8/16/10`), `0.650`
  (`20/6/16/10`), `0.575` (`16/4/16/10`), `0.750` (`20/10/20/10`),
  `0.725` (`20/8/20/10`), `0.650` (`20/10/12/10`), and `0.5875`
  (`20/7/12/8`) for grammar/semantic/convention/completeness. The full
  reviewer commentary says the referent is clearance to the conducting wall
  behind the strap and recommends matching the specificity of
  `outer_radius_of_antenna_strap` and `inner_radius_of_antenna_strap`.

### `accumulated_total_coolant_absorbed_energy_of_calorimetry_component`

* `name_stage=exhausted`, `reviewer_score_name=0.75`, and
  `reviewer_scores_name={"grammar":20,"semantic":12,"convention":10,"completeness":18}`.
* `refine_attempts=2/3`, `docs_stage=pending`, `status=draft`,
  `validation_status=quarantined`, `validation_issues=null`, and
  `quarantine_reason=null`.
* Producer: `dd:calorimetry/group/component/energy_total/data`; the available
  source text is only “Data” and its enriched description is “Data array for
  energy total”. The identity description is “Discharge-accumulated total
  energy absorbed by coolant from a calorimetry component.”
* The three name-review rows are `0.6875` (`20/10/10/15`), `0.8125`
  (`20/15/10/20`), and `0.750` (`20/12/10/18`). The full commentary finds
  the spelling grammatically clean but a near-duplicate of the accepted
  `accumulated_coolant_absorbed_energy_of_calorimetry_component`, with the
  extra `total` redundant.
* The graph records `refine_stop_reason=grammar_invalid` in the review
  commentary: “refined candidate failed strict grammar validation”. Because
  the actual quarantine cause was not persisted, the exact current barrier
  is the quarantined validation state with a missing reason, not a permission
  to clear it. This is itself a write-path evidence defect. The accepted
  sibling has a different producer, `dd:calorimetry/group/component/energy_cumulated`,
  so it cannot be used as an unrecorded source substitution.

## Sanctioned recovery and candidate evidence

### Neutral-gauge successor

The initial scoped `sn edit --dry-run` for `neutral_pressure` returned exit 0:

```text
DRY RUN sn edit: neutral_pressure  mode=rename axis=name scope=only_self
entry=review_name

Actions:
  -  would carry 1 producing source(s) to 'neutral_gas_pressure_of_gauge'
  -  would rename 'neutral_pressure' -> 'neutral_gas_pressure_of_gauge'
```

The stage-only attachment carried the one producer and entered `review_name`:

```text
APPLIED sn edit: neutral_pressure  mode=rename axis=name scope=only_self
Actions:
  - verified 1 producing source(s) on 'neutral_gas_pressure_of_gauge'
  - renamed 'neutral_pressure' -> 'neutral_gas_pressure_of_gauge', entering name review (edit_status=open, run_id=sn-edit-20260915T042613Z)
```

The reason supplied to the reviewer was grounded in the DD leaf and physics:
the path is a barometry-gauge pressure reading in the vacuum vessel, so the
candidate names neutral gas and the gauge that indicates it, separating this
instrument quantity from kinetic neutral-pressure moments and their accepted
plasma-diagnostic siblings.

The attached candidate's complete name-review quorum was:

| Role/model | Aggregate | Dimensions (grammar/semantic/convention/completeness) |
| --- | ---: | --- |
| primary, `openrouter/x-ai/grok-4.5` | `0.975` | `20/19/19/20` |
| secondary, `openrouter/openai/gpt-5.6-luna` | `0.9875` | `20/19/20/20` |

The persisted aggregate is `0.98125`; the candidate is `name_stage=accepted`,
`edit_status=applied`, `validation_status=valid`, `validation_issues=[]`,
and `review_quorum_shortfall=null`. It has one incoming producer edge from
`dd:barometry/gauge/pressure`; the source's `produced_sn_id` is the accepted
candidate. The retired `neutral_pressure` remains `name_stage=superseded`
with its old score and no producer. The rename's lineage is recorded by
`neutral_gas_pressure_of_gauge-[:REFINED_FROM]->neutral_pressure`; the retired
node's scalar `superseded_by` remains null, so consumers must use the recorded
lineage edge rather than assume that scalar is populated.

The ordinary documentation route was then run with the same narrow scope and
`--skip-global-maintenance`. It generated one documentation candidate and
reviewed/refined that candidate until acceptance. The final graph description
is:

> Neutral gas pressure of a gage is the pressure indicated by a barometry gage
> for residual neutral gas in the vacuum vessel or surrounding vacuum system.
> It represents local vacuum conditions rather than a kinetic neutral-particle
> pressure moment.

The documentation candidate trajectory is recorded by the nine attached
documentation review rows, three per cycle:

| Cycle | Role/model | Scores for description quality/documentation quality/completeness/physics accuracy | Aggregate |
| ---: | --- | --- | ---: |
| 0 | primary, `openrouter/anthropic/claude-sonnet-5` | `15/10/20/20` | `0.8125` |
| 0 | primary, `openrouter/anthropic/claude-sonnet-5` | `15/15/20/20` | `0.875` |
| 0 | primary, `openrouter/anthropic/claude-sonnet-5` | `15/16/14/20` | `0.8125` |
| 1 | secondary, `openrouter/x-ai/grok-4.5` | `15/15/20/20` | `0.875` |
| 1 | secondary, `openrouter/x-ai/grok-4.5` | `15/15/15/20` | `0.8125` |
| 1 | secondary, `openrouter/x-ai/grok-4.5` | `15/15/20/20` | `0.875` |
| 2 | escalator, `openrouter/openai/gpt-5.5` | `15/15/20/20` | `0.875` |
| 2 | escalator, `openrouter/openai/gpt-5.5` | `15/15/15/20` | `0.8125` |
| 2 | escalator, `openrouter/openai/gpt-5.5` | `15/15/15/20` | `0.8125` |

The documented review concern was consistent across cycles: the generated
prose used “gage” while the accepted name and DD source use “gauge”. The
final persisted documentation review is `docs_stage=accepted` at `0.875`,
with `reviewer_scores_docs={"description_quality":15,"documentation_quality":15,"completeness":20,"physics_accuracy":20}`.
The review receipt was:

```text
persist_reviewed_docs: neutral_gas_pressure_of_gauge -> docs_stage=accepted
(score=0.875, chain=2/3, resolution=authoritative_escalation, shortfall=None)
review_docs: neutral_gas_pressure_of_gauge -> accepted
(score=0.875, cycles=3, method=authoritative_escalation)
```

The successor now satisfies name acceptance, documentation acceptance, and
valid validation with its DD producer intact. Its physics domain is
`mechanical_measurement_diagnostics`, its validation observation is
`2026-09-15T04:26:15.178Z`, and both review shortfall fields are null, so the
current WEST admission gates would carry it. No cut was performed by this
node.

### Antenna strap distance

The prior diagnosis was confirmed by a fresh dry run. It returned exit 2:

```text
sn edit distance_of_antenna_strap mode=rename axis=name scope=only_self
BLOCKED
a StandardName 'gap_of_antenna_strap' already exists

Actions considered:
  - a StandardName 'gap_of_antenna_strap' already exists
```

`gap_of_antenna_strap` is itself exhausted, so the collision is not a
publishable replacement. Candidate parser checks also show that
`distance_to_conductor_of_antenna_strap`,
`distance_of_antenna_strap_to_conductor`,
`distance_of_antenna_strap_to_conducting_wall`,
`distance_of_conductor_behind_antenna_strap`, `wall_distance_of_antenna_strap`,
and `antenna_strap_distance_to_conductor` are rejected by the active grammar;
`back_surface_distance_of_antenna_strap` parses but omits the wall/conductor
referent. No candidate was attached and no graph state changed. The exact
remaining barrier is therefore the existing `gap_of_antenna_strap` identity,
combined with no currently preferred grammar-valid spelling that preserves
the conducting-wall clearance semantics.

### Accumulated coolant energy

No edit dry run was attempted. The identity is quarantined and its graph state
does not contain the reason required to adjudicate or clear that quarantine:
`validation_status=quarantined`, `validation_issues=null`,
`quarantine_reason=null`, and `refine_stop_reason=grammar_invalid`. Directly
clearing it would falsify the validation record and was not permitted. No
candidate was attached and no graph state changed. The exact remaining
barrier is the quarantined validation predicate with an absent persisted
cause; the owning validation repair must revalidate through the sanctioned
route and record the actual grammar result before any edit can be considered.

## Final state, spend, and follow-ons

The final bounded readback is:

| Identity | Final name state | Final docs state | Validation | Attempts | Producer | WEST |
| --- | --- | --- | --- | --- | --- | --- |
| `neutral_gas_pressure_of_gauge` | `accepted`, score `0.98125` | `accepted`, score `0.875` | `valid` | `3/3` | `dd:barometry/gauge/pressure` | Yes |
| `distance_of_antenna_strap` | `exhausted`, score `0.7125` | `pending` | `valid` | `3/3` | `dd:ic_antennas/antenna/module/strap/distance_to_conductor` | No |
| `accumulated_total_coolant_absorbed_energy_of_calorimetry_component` | `exhausted`, score `0.75` | `pending` | `quarantined` | `2/3` | `dd:calorimetry/group/component/energy_total/data` | No |

The latest documented campaign total before this node was `$106.25`. This
node spent `$0.071967` on the name review and `$0.861414` on documentation
generation/review/refinement, for a measured node delta of `$0.933381` and a
campaign total after this node of `$107.183381` (about `$107.18`). This is
below the `$20.00` node cap and the authorised `$250.00` campaign ceiling.

The remaining follow-ons are deliberately separate: documentation was
completed for the recovered neutral-gauge successor, while the antenna
identity needs an authorised grammar-valid collision-free spelling and the
coolant identity needs an evidence-producing validation repair for its missing
quarantine reason. The two excluded identities remain out of the cut until
those exact predicates change.

## Receipts

The dry-run and pipeline receipts are in:

`/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log`

The name run was scoped to `neutral_gas_pressure_of_gauge` and exited 0. The
documentation run was scoped to that same accepted successor, bypassed global
maintenance, and exited 0; its final receipt is quoted above. All live graph
queries were bounded named-identity reads and completed within the ten-second
per-query limit.
