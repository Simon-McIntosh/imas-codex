# Integrity and repair-operator closure verification

Date: 2026-08-25

Live graph measurement: 2026-08-25T04:45:06.584Z

Checkout: `495b697e0ef70da51cfe54b6848ae4d9323cf6db`

## Verdict

Neither plan is unqualifiedly complete.

- `sn-graph-wide-integrity` was genuinely closed on its 2026-08-22 snapshot,
  including explicit recorded refusals, but it is not closed against the live
  graph now. Two exact-zero invariants have regrown without plan writeback:
  **39 composed/attached sources have no live target** and **3 sources have a
  scalar mirror mismatch**. The graph-wide plan therefore has **4 outstanding
  declared-deliverable rows: 2 unrecorded regressions and 2 recorded residual
  classes**.
- `sn-repair-operator-consolidation` is explicitly incomplete backlog. It has
  **4 outstanding declared-deliverable rows**. Only **3 of the 10** targeted
  operators are migrated; **7 bespoke implementations remain**, and the
  disjoint-apply concurrency proof and corresponding serialization decision are
  still open.

The live queries were read-only. The first and repeated source-integrity
measurements were identical over a 3.5-minute interval, so the 39/3 result was
not a single transient read.

## Live results

Each result below is numbered so the deliverable tables can cite the exact live
measurement rather than a historical plan assertion.

### L1 — source-to-target integrity partition

The query uses the production invariant from
`imas_codex/standard_names/provenance_lifecycle.py:1002-1045`: composed or
attached sources require exactly one live target, a matching
`produced_sn_id`, and, for DD/signal sources, a matching upstream
`HAS_STANDARD_NAME` projection. Before trusting any zero, a schema sanity query
measured **4,656/4,656** `StandardName` nodes with `name_stage` and
**9,668/9,668** `StandardNameSource` nodes with `status`.

```text
measured_at             2026-08-25T04:45:06.584Z
no_live_target          39
multiple_live_targets    1
scalar_mismatch          3
projection_mismatch      0
```

The 39 no-live-target rows partition as **36 derived + 3 DD**. The DD rows are
`dd:ntms/time_slice/mode`, `dd:summary/pedestal_fits`, and
`dd:waves/coherent_wave`. The 3 scalar mismatches partition as **2 DD + 1
derived**. The sole dual-bound row is
`dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial`, with live
targets `radial_neutral_internal_state_momentum_flux` and
`radial_neutral_state_momentum_flux`.

### L2 — accepted-name strict grammar

Cypher selected every `StandardName {name_stage: 'accepted'}` and the installed
public ISN parser strictly parsed and recomposed each identity.

```text
accepted names                       2,295
strict parse or round-trip failures      0
```

### L3 — owner/geometry recorded refusals

The two exact sources named by the plan still select the old accepted target:

```text
dd:b_field_non_axisymmetric/time_slice/field_map/grid/phi
  status=composed  scalar=toroidal_angle_of_measurement_position
  live target=toroidal_angle_of_measurement_position

dd:spectrometer_x_ray_crystal/channel/reflector/centre/phi
  status=attached  scalar=toroidal_angle_of_measurement_position
  live target=toroidal_angle_of_measurement_position
```

These are recorded policy/identity refusals, not newly discovered rows; the
original exact semantics are documented at
`docs/evidence/sn-graph-wide-integrity/owner-geometry-residue-adjudication.md:24-35`.

### L4 — unsourced live-name partition

```text
live names with no producer and no live parent producer   4
with a live structural child                              0
without a live structural child                           4
```

The four identities are
`fast_ion_charge_state_power_at_inside_flux_surface`,
`neutron_flux_due_to_fusion`,
`tendency_of_total_thermal_plasma_internal_energy`, and
`toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions`.
All four already occur in the 2026-08-22 exclusive disposition record at
`docs/evidence/sn-graph-wide-integrity/sprint-closing-census.md:27-91`; they are
qualified recorded holds/dispositions, not the unrecorded regression identified
by L1.

### L5 — recorded residual lifecycle drift

The three historically below-threshold identities are now all `superseded`;
`voltage_of_diagnostic_antenna` is now `exhausted`. Of the four historically
DD-held identities, `fast_ion_charge_state_power_at_inside_flux_surface` and
`toroidal_ion_charge_state_torque_density` remain accepted, while the x/z
sensor-direction names are superseded. This means the plan's remaining-work
table is a valid 2026-08-22 record but is not a current census.

## `sn-graph-wide-integrity` declared deliverables

The declared closure table is at `docs/sn-graph-wide-integrity.html:52-65`; the
shared signed-envelope deliverable is at
`docs/sn-graph-wide-integrity.html:77-84`. A qualified landed verdict means the
plan explicitly chose a measured refusal or disposition instead of mutation.

| Declared deliverable | Evidence | Verdict |
|---|---|---|
| Every composed/attached semantic source has a live target | Live result **L1**: **39**, versus the recorded close value 0 at `docs/sn-graph-wide-integrity.html:58` | **OUTSTANDING — unrecorded regression** |
| `produced_sn_id` scalar mirrors the sole live target | Live result **L1**: **3**, versus the recorded close value 0 at `docs/sn-graph-wide-integrity.html:59` | **OUTSTANDING — unrecorded regression** |
| DD/signal upstream projection mirrors the sole live target | Live result **L1**: **0** | **LANDED** |
| Every accepted identity strictly parses and round-trips under the installed grammar | Live result **L2**: **0 failures of 2,295** | **LANDED** |
| Dual-bound sources are eliminated or carry a named last-producer refusal | Live result **L1**: **1** remaining, explicitly identified; the plan allows recorded refusals at `docs/sn-graph-wide-integrity.html:62` | **OUTSTANDING — recorded refusal** |
| Owner/geometry rows leave the old target or carry a named policy/identity refusal | Live result **L3**: **2** exact rows remain on the old target with prior adjudications | **OUTSTANDING — recorded refusals** |
| Live names without producing sources have an exclusive disposition | Live result **L4**: **4**, all four present in the exclusive 2026-08-22 disposition at `docs/evidence/sn-graph-wide-integrity/sprint-closing-census.md:27-91` | **LANDED — qualified disposition** |
| Structurally bare live names have no live child and a named reason | Live result **L4**: **4 childless, 0 childful**; all four are in the recorded disposition | **LANDED — qualified disposition** |
| Repairs use closed programs under the shared signed envelope, with schema-valid authority construction | `apply_signed_manifest` exists at `imas_codex/standard_names/signed_manifest.py:4374`; the validated builder exists at `imas_codex/standard_names/repair_authority.py:135`; focused contract verification was **21 passed, 10 graph cases deselected** | **LANDED** |

Outstanding row count for this plan: **4 of 9**. Of those, **2 are recorded
qualified residuals** and **2 are unrecorded live regressions**.

The green live ratchet suite does not contradict this verdict. It passed **4/4**,
but its declared ceilings at `tests/graph/test_sn_integrity_ratchets.py:12-16`
cover multiple targets, stale-source bindings, unsourced names, and one explicit
axis residue. It does not assert an exact zero for L1's no-live-target or scalar
mirror classes.

## `sn-repair-operator-consolidation` declared deliverables

The five implementation deliverables are declared at
`docs/plans/sn-repair-operator-consolidation.html:171-198`; the two verification
requirements are at `docs/plans/sn-repair-operator-consolidation.html:201-216`;
the pre-migration suite prerequisite is at
`docs/plans/sn-repair-operator-consolidation.html:255-281`.

| Declared deliverable | Evidence | Verdict |
|---|---|---|
| Characterize the target operators before consolidation | The landed record reports **10/10 targeted operators**, 9 genuine extension points, and 8 accidents at `docs/evidence/archive/sn-repair-operator-consolidation-landed.html:21-55` | **LANDED** |
| Define one authority schema that reads committed artifacts without re-signing | The landed record validates **4/4 artifacts and 412 rows** from original bytes at `docs/evidence/archive/sn-repair-operator-consolidation-landed.html:61-101` | **LANDED** |
| Implement the generic eight-step `apply_signed_manifest` envelope | Implementation entry point `imas_codex/standard_names/signed_manifest.py:4374`; the focused schema/builder/operator run was **21 passed** | **LANDED** |
| Migrate the ten targeted operators class by class | The plan's current execution record states **3 migrated and 7 unmigrated** at `docs/plans/sn-repair-operator-consolidation.html:360-361` | **OUTSTANDING — 7 of 10 migrations remain** |
| Delete each bespoke operator as its migration lands | Seven bespoke entry points remain at `imas_codex/standard_names/graph_ops.py:3200`, `:4234`, `:5018`, `:19147`, `:19830`, `:20288`, and `:21105`; the three migrated paths are partial adapters at `imas_codex/standard_names/graph_ops.py:13173` and `imas_codex/standard_names/provenance_lifecycle.py:874` plus the retired-provenance adapter described at `docs/plans/sn-repair-operator-consolidation.html:333-343` | **OUTSTANDING — 7 bespoke implementations remain** |
| Prove identical admitted/refused sets, reasons, receipts, and replay for each migration | The equivalence gate is declared at `docs/plans/sn-repair-operator-consolidation.html:201-210`; it is proven for the 3 migrated operators but necessarily remains to be run for the 7 pending migrations | **OUTSTANDING — 7 equivalence gates remain** |
| Prove disjoint generic invocations cannot corrupt each other's collateral proof, then settle serialization | The property is declared at `docs/plans/sn-repair-operator-consolidation.html:212-216`; the decision has no choice at `docs/plans/sn-repair-operator-consolidation.html:296-303`; the closing record explicitly says no evidence was gathered at `:360-361` | **OUTSTANDING — open decision and missing proof** |
| Build disposable-graph baselines for the three operators that lacked them before migration | The plan records all three baseline suites complete, **20 disposable-graph cases total**, at `docs/plans/sn-repair-operator-consolidation.html:351-352` | **LANDED** |

Outstanding row count for this plan: **4 of 8**. The four rows are not newly
discovered; the plan itself records them as unscheduled backlog.

## What the stored implementation fractions actually mean

### `sn-graph-wide-integrity`: `impl=0.965`

The fraction is stored at `docs/sn-graph-wide-integrity.html:19-22`. It is not a
formula over the live graph and it must not be interpreted as “only 3.5% of the
current integrity work remains.” Concretely, it represented a qualified
2026-08-22 closure in which the plan deliberately retained recorded external or
ordinary-pipeline residues: four dual-bound sources, two owner/geometry
refusals, three below-threshold identities, four DD-defect holds, 17 childless
structural names, and the standing voltage refusal
(`docs/sn-graph-wide-integrity.html:97-145`). Those sets overlap and were never
an arithmetic denominator. The live graph has since evolved to one dual-bound
source, the same two owner/geometry refusals, four childless unsourced names,
and changed lifecycle states for the other recorded rows — while also acquiring
the **39 no-live-target and 3 scalar-mismatch regressions** in L1. The `0.965`
value is therefore a stale stored progress marker for the historical qualified
close, not current completion evidence.

### `sn-repair-operator-consolidation`: `impl=0.98`

The fraction is stored at
`docs/plans/sn-repair-operator-consolidation.html:27-30`. It represents the
discharged sprint clause — characterization, generic envelope, three migrated
operators, pre-migration suites, and authority builder — not completion of the
whole plan. In concrete remaining work, **7 of the 10 targeted operator
migrations remain (70% of the migration cohort)**, their seven bespoke bodies
still exist, their per-class equivalence/deletion gates have not run, and the
serialization-retirement proof and decision are open. The plan says exactly
that at `docs/plans/sn-repair-operator-consolidation.html:360-361`. The stored
2% residual is therefore not proportional to remaining implementation volume.

## Verification commands and limitations

- Focused authority/operator contract:
  `pytest -m 'not graph' test_repair_authority_schema.py test_repair_authority_builder.py test_signed_manifest_operator.py`
  — **21 passed, 10 deselected, 0 failed**. Log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T043712518942-n-integrityoperatorclosure/logs/operator-contract-focused.log`.
- Live ratchets: `pytest tests/graph/test_sn_integrity_ratchets.py` — **4
  passed, 0 failed**. As noted above, these are ceilings and do not cover both
  newly regrown exact-zero classes.
- The Reckon `read_plan` endpoint failed before returning this plan because an
  unrelated nested research path violates the typed-resource path contract:
  `research/standard-names/01-current-state-standard-names.html`. The complete
  canonical HTML for both plans was therefore read directly from this checkout.
  No plan or index state was edited.
- No disposable-Neo4j graph cases were rerun in this node; those 10 cases are
  reported honestly as deselected. Their historical executed evidence remains
  in the operator plan, while this node's completion decision rests on current
  source presence and live read-only Cypher.

## Required writeback

1. Reopen graph-wide closure for the **39 no-live-target** and **3 scalar mirror
   mismatch** rows, derive exact cohorts from current state, and preserve the
   plan's fail-closed authority rules. The three DD no-target sources should be
   adjudicated separately from the 36 derived rows.
2. Refresh the graph-wide plan's current census and `impl` only after those
   exact-zero invariants are restored or each row has an explicit recorded
   refusal. Current lifecycle changes also require replacing the stale seven-row
   remaining-work table.
3. Keep operator consolidation active. The next trigger executes one of the
   seven remaining migrations with its unchanged disposable-graph baseline,
   deletes that bespoke implementation in the same landing, and records its
   exact equivalence result. Do not describe the plan as complete until all
   seven migrate or are explicitly removed from scope by a plan decision.
4. Keep repair applies serialized until the disjoint-collateral proof is run and
   the open serialization decision is resolved.
