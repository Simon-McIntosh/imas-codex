# Attachment target ordinary-review preflight

## Outcome

The nine target-lifecycle refusals cannot be carried through ordinary review as
one legal cohort from the current live state. The fail-closed preflight recorded
all nine identities and ran the deterministic, LLM-free validation operator
against the sole active graph grammar, ISN **0.8.0rc66** with **956 tokens**.
Only four identities are review-admissible. Three re-quarantined with current
grammar or audit findings, and two have no description and therefore cannot be
claimed by the validation or review workers.

No quorum was started for a partial cohort. Consequently the requested measure
is **not met**: **0 of 9**, rather than 9 of 9, received one fresh quorum draw.
There were **0 retries**, **0 fresh review groups**, **0 provider calls**, and
**USD 0.000000** attributable spend against the USD 25 ceiling. The accepted
target count among the nine was **0 before and 0 after**. `LLMCost` remained at
**27,631 rows** and `StandardNameChange` remained at **7,754 rows**.

## Per-identity admission record

The acceptance threshold is **0.85**. A score shown in the `Pre-review score`
column is historical live state, not a result of this node. `No draw` is an
explicit fail-closed result and must not be read as a score.

| Attachment row | Identity | Pre-review stage | Preflight validation | Pre-review score | Fresh quorum draw | Result against 0.85 |
|---:|---|---|---|---:|---:|---|
| 02 | `cross_section_of_flux_surface` | pending | quarantined: strict grammar round-trip fails | — | 0 | no draw; not review-admissible |
| 05 | `line_integrated_electron_density` | drafted | quarantined: `cumulative_prefix_check` rejects `line_integrated` | — | 0 | no draw; not review-admissible |
| 07 | `minimum_of_safety_factor` | reviewed | valid | 0.72500 | 0 | no partial-cohort draw |
| 08 | `neutral_state_power_density` | reviewed | valid | 0.83125 | 0 | no partial-cohort draw |
| 13 | `poloidal_neutral_internal_state_momentum_convected_velocity` | reviewed | valid | 0.56875 | 0 | no partial-cohort draw |
| 15 | `poloidal_straight_field_line_angle` | drafted | quarantined: description contains storage-shape tag `2D`; `implicit_field_check` also rejects the spelling | — | 0 | no draw; not review-admissible |
| 20 | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | reviewed | valid | 0.83125 | 0 | no partial-cohort draw |
| 21 | `toroidal_line_integrated_impurity_ion_velocity` | drafted | quarantined and description absent; the validator cannot claim it | — | 0 | no draw; not review-admissible |
| 29 | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | drafted | quarantined and description absent; the validator cannot claim it | — | 0 | no draw; not review-admissible |

The count is exact: **1 pending + 4 drafted + 4 reviewed = 9 identities**;
**4 valid + 3 freshly re-quarantined + 2 description-less unclaimable = 9**.
All nine ended with null claim tokens.

## Why review stopped

Ordinary name review claims only a drafted identity whose
`validation_status='valid'` and whose description is present. Reviewed
identities can be returned to drafted through the sanctioned `sn rescore
--stage-only` route, but that does not make the five invalid or incomplete
identities admissible. Bypassing validation to force five quorum calls would
violate the ordinary-review contract and would turn a validation or grammar
failure into acceptance evidence, which the live plan explicitly forbids.

The current deterministic validation pass made no semantic or lifecycle
override. It cleared and re-ran validation stamps for the exact nine-name set:
four were confirmed valid, the three named findings were re-quarantined, and
the two description-less names remained unclaimable with their prior
quarantine findings visible. No name was staged, rewritten, refined, rescored,
accepted, attached, or retried.

## Required follow-on before this cohort can be reviewed

The five blocked identities require their existing defects to be resolved
through sanctioned identity or documentation work before a fresh quorum draw:

- decide the canonical replacement or grammar treatment for
  `cross_section_of_flux_surface`;
- correct or narrow the cumulative-prefix rule for the two legitimate
  `line_integrated` quantities, or stage reviewed replacement identities;
- resolve the straight-field-line spelling and remove the storage-shape prose
  from `poloidal_straight_field_line_angle`;
- provide governed descriptions for the magnetic-field and toroidal
  line-integrated identities, then re-run deterministic validation.

Only after all five validate should the exact nine be staged together and sent
through one ordinary review invocation. The four currently valid identities
must not be drawn separately and then drawn again as part of the repaired
cohort, because that would violate the one-draw-per-identity constraint.

## Runtime evidence

Durable runtime evidence is under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T145705632615-targetreview/`:

- `baseline.json` / `baseline.log`: pre-review lifecycle, validation, score,
  grammar, cost, and accepted-count census;
- `revalidate-targets.log`: exact nine-name deterministic validation result;
- `final.json` / `final.log`: postflight state, zero fresh cost, zero fresh
  review groups, null claims, and accepted count 0;
- `verification.json` / `verification.log`: quantitative blocker gate.
