# Nine accepted identities carry null catalog status — full-population export precondition

Node: disposition of the nine candidate identities whose null graph `status` blocks the
full-population export.

Measured at worktree base `b77c4b39e28cb2233d47cd4fb8479bb545a33d0a`, 2026-09-10 (UTC).
All live-graph reads were bounded indexed queries against the login-node-local tunnel;
no graph structure was modified by this node.

## Reproduction (confirmed at HEAD)

`imas-codex sn release --export-only --gate-only` (full population, no batch) exits 1 and
refuses exactly as recorded in the plan:

```
Export blocked: 9 candidate identity(ies) have null graph status: ['accumulated_coolant_absorbed_energy_of_calorimetry_component', 'alpha_angle_of_passive_loop_element', 'calibration_reference_magnetic_field_of_poloidal_magnetic_field_probe', 'neutron_flux_maximum_at_first_wall', 'neutron_response_function_of_neutron_detector', 'opacity_at_ece_channel_emission_position', 'strain_of_strain_gauge', 'total_heating_power_due_to_fusion_reactions', 'wave_curvature_of_electron_cyclotron_beam']
```

Render: total candidates 2474, `catalog_status: FAIL (9 issue(s))`, gate failure blocks.
The gate lives in `run_export` at `imas_codex/standard_names/export.py:2502-2529`
(`GATE_CATALOG_STATUS`); it refuses before any eligibility classification and returns an
`ExportReport` with `all_gates_passed=False`.

The cohort is exactly nine: a direct count of accepted/approved identities with
`status IS NULL` over the graph returns `n = 9` with precisely these ids. This is a
closed cohort, not a sample.

## Per-identity table

All nine share the same shape: `name_stage=accepted`, `docs_stage=accepted`,
`validation_status=valid`, no `edit_status`, no `deprecates`/`superseded_by`, one or two
`dd` producers, and a `refine` internal change written the second they were created —
all **minted 2026-09-10**, accepted on both axes, and every one carrying recorded LLM
spend (undeletable by standing ruling; the instrument is `LLMCost.standard_name_ids`).

| identity | name_stage | docs_stage | producers (source types) | DD bindings | LLM cost rows | LLM spend (USD) | created_at (UTC) | last touch `updated_at` (UTC) | last recorded change (`changed_at`) | null meaning |
|---|---|---|---|---|---|---|---|---|---|---|
| accumulated_coolant_absorbed_energy_of_calorimetry_component | accepted | accepted | 1 (dd) | 1 | 6 | 0.2178 | 2026-09-10 04:25:42 | 2026-09-10 09:52:56 | refine @ 04:25:42 | write path failed to set field |
| alpha_angle_of_passive_loop_element | accepted | accepted | 1 (dd) | 1 | 6 | 0.2333 | 2026-09-10 04:23:27 | 2026-09-10 04:48:24 | refine @ 04:23:27 | write path failed to set field |
| calibration_reference_magnetic_field_of_poloidal_magnetic_field_probe | accepted | accepted | 1 (dd) | 1 | 7 | 0.3428 | 2026-09-10 09:55:57 | 2026-09-10 10:16:12 | refine @ 09:55:57 | write path failed to set field |
| neutron_flux_maximum_at_first_wall | accepted | accepted | 1 (dd) | 1 | 9 | 0.6029 | 2026-09-10 04:23:41 | 2026-09-10 10:10:28 | refine @ 04:23:41 | write path failed to set field |
| neutron_response_function_of_neutron_detector | accepted | accepted | 1 (dd) | 1 | 7 | 0.3550 | 2026-09-10 04:21:31 | 2026-09-10 04:49:25 | refine @ 04:21:31 | write path failed to set field |
| opacity_at_ece_channel_emission_position | accepted | accepted | 1 (dd) | 1 | 6 | 0.2592 | 2026-09-10 04:24:00 | 2026-09-10 04:49:13 | refine @ 04:24:00 | write path failed to set field |
| strain_of_strain_gauge | accepted | accepted | 2 (dd) | 2 | 10 | 0.8713 | 2026-09-10 04:25:47 | 2026-09-10 05:03:07 | refine @ 04:25:47 | write path failed to set field |
| total_heating_power_due_to_fusion_reactions | accepted | accepted | 1 (dd) | 1 | 6 | 0.2538 | 2026-09-10 09:49:16 | 2026-09-10 09:55:57 | refine @ 09:49:16 | write path failed to set field |
| wave_curvature_of_electron_cyclotron_beam | accepted | accepted | 2 (dd) | 2 | 10 | 0.7962 | 2026-09-10 04:40:08 | 2026-09-10 10:12:35 | refine @ 04:40:08 | write path failed to set field |

Nine identities, **67 LLM cost rows, ≈ 3.93 USD** recorded spend in total. Aggregate
column values: producers min 1 / max 2; reviews min 4 / max 8 per identity (all accepted
on both axes); non-null on every row except `status`.

## What the null means — one verdict, not nine

The schema settles the question before any archaeology: `StandardNameStatus` is declared
with permissible values `draft | active | deprecated | superseded`, and its documentation
states the default for pipeline-minted names is `draft` ("Pipeline-composed, not yet
imported to the ISN catalog vocabulary"). A null `status` on an identity the pipeline
minted and accepted is therefore **not a legitimate unset state the precondition should
tolerate** — null is outside the enumerated vocabulary, and the export gate
(`catalog_status`) was added as a deliberate precondition. For **all nine** the null means:
**a write path that failed to set the field** (Hypothesis A). There is no member of the
cohort for which the gate should instead tolerate the null (Hypothesis B).

`status` is also a catalog-authoritative `PROTECTED_FIELD`
(`imas_codex/standard_names/protection.py:22-37`) after approval, so the repair must go
through the sanctioned writer (override-aware), never an in-place Cypher `SET`.

### The failed write path

Only one node-creation writer sets the field: `write_standard_names`
(`graph_ops.py:5424-5425`) does `ON CREATE SET sn.status = 'draft'` — and because
`status` is absent from its MERGE-update list, an identity first created *anywhere else*
with a null status is never repaired by any later governed write; `updated_at` advances
with every pass, `status` never does.

The two pipeline paths that spawn new StandardName identities today both omit the field:

1. **Refine-successor creation** — `persist_refined_name` (`graph_ops.py:18261-18276`):
   `MERGE (new:StandardName {id: $new_name}) ON CREATE SET` writes
   `name_stage='drafted'`, `docs_stage`, and inheritance fields, and nothing named
   `status`. This is the creating path for **all nine**: every one has a `REFINED_FROM`
   parent (stage superseded) and a `refine` internal change stamped at `created_at`
   (which equals `generated_at`), both written by this function in its one transaction.
2. **First-generation binding-reservation creation** —
   `_lock_claimed_name_bindings` (`graph_ops.py:10506-10513`) with `allow_missing=True`:
   `MERGE (target:StandardName {id: b.sn_id}) ON CREATE SET
   target.name_stage=$pending_stage ...` — again no `status`. `_finalize_generated_name_stage`
   (`graph_ops.py:7739-7798`) then transitions stage but also never writes `status`.

The same-shape evidence is already on disk: four of the nine superseded parents carry
`status='draft'` (minted by `write_standard_names`) while the other five carry
`status=null` (minted by one of the two paths above) — `opacity_at_ece_channel_emission_position`
comes from `line_integrated_opacity_at_ece_channel_emission_position` (null), and
`total_heating_power_due_to_fusion_reactions` from `total_thermal_power_due_to_fusion_reactions`
(null). Those parents are superseded, so they do not block the export; the next unguarded
refine cycle reproduces the null at every generation. The bug is systemic across the two
creation paths, not a one-off.

## When each status was last written

For all nine: **never**. No current or traced historical write path sets `status` on an
existing node, and these nodes were created by paths that omit it, so there is no write
to timestamp. The closest bounds are the node's own `created_at` (the write that *should*
have set `draft`) and the last internal change (`refine`, written at creation). The
`updated_at` column above is the *last touch of any kind* and is later than creation on
every row — evidence that later pipeline writes repeatedly visited these nodes (reviews,
acceptance) without ever writing the field, which is exactly the "check runs, produces
nothing, and nothing acts on it" shape this sprint has met before.

## Repair

**Nothing in the graph was modified by this node.** Deletion is forbidden on all nine
(standing ruling, instrument = `LLMCost.standard_name_ids`; 67 rows / ≈ 3.93 USD). An
in-place Cypher `SET` is forbidden by instruction. No sanctioned, in-scope CLI path
exists to write the catalog-status axis on these identities: `sn realign-segments`
touches grammar segments only; `write_standard_names` repairs `status` only on create;
the catalog round-trip that normally promotes `draft → active/...` targets catalog
entries, not graph-only identities; re-running the review persist does not backfill a
protected field.

### Residue left for authorisation (follow-on, not a signed-apply blocker)

None of the nine needs a lifecycle transition that only a signed manifest can make — all
nine are already `accepted` on both axes with `validation_status=valid`; `status` is an
orthogonal catalog-axis field, not an approval transition, so **no signed-apply blocker
is raised**. The repair is instead two pieces, both outside this node's write fence:

1. **Code fix** (new code node): add `status='draft'` to the `ON CREATE SET` in
   `persist_refined_name` (`graph_ops.py:18262`) and in the
   `_lock_claimed_name_bindings` target clause (`graph_ops.py:10506`); consider
   `SET sn.status = coalesce(sn.status, 'draft')` in `_finalize_generated_name_stage`
   (`graph_ops.py:7780`) as a belt-and-braces guard. Counterfactual that would prove the
   fix: generate and refine a fresh source into acceptance, then read back
   `status='draft'` on the minted and on the superseded-successor identities.
2. **One-time governed backfill** of `status='draft'` for the nine (and, for cohort
   consistency, the null-status superseded parents), through the sanctioned override-aware
   writer with a ledger entry per identity. Counterfactual: re-run
   `sn release --export-only --gate-only` and see `catalog_status: PASS (0 issues)` with
   the cohort re-measured at zero accepted nulls.

Until both land, every full-population export (and any batch that contains a member of
this cohort) refuses at `export.py:2522`. The gate itself is working as intended; the
defect is upstream of it in the identity creators.

## Evidence that could have failed

- The reproduction above quotes the live refusal rather than asserting it; it is
  reproducible at HEAD with any full-population export.
- The cohort count (`n = 9`, exact ids) was measured by direct query, not taken from the
  refusal message; the refusal and the count agree.
- The per-identity verdicts discriminate the two hypotheses on the schema's own
  enumeration, not on an absence: a zero (null cohort) would have been checked against a
  known-present control (the 2474-candidate population), which is what the reproduction
  did.
- The creating-site claim rests on receipts of that create path (`REFINED_FROM` parent +
  `refine` change at `created_at == generated_at`), not on the absence of other writes.

## Artifacts

- Reproduction log: `~/.local/share/imas-codex/logs/` (command `sn release --export-only
  --gate-only`, run 2026-09-10)
- Per-identity JSON: `/tmp/n-swcr-nine-analysis.json` (9 ids × status/stages/producers/
  bindings/reviews/spend/change-ledger)
- Cohort JSON: `/tmp/n-swcr-nine-cohort.json` (n=9 exact ids; REFINED_FROM parents with
  parent stage `superseded` and parent status)
- Analysis script: `/tmp/n-swcr-nine-analysis.py`, `/tmp/n-swcr-nine-cohort.py`
