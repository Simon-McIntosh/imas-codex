# The re-minted WEST cut, measured — candidate 220, published 218, exclusions 2, residue 0

Rehearsal measured 2026-09-11 from the west-task-2e batch. Cut nothing: no
roster frozen, no branch, no tag, no pull request, no release-candidate
counter advanced. This is the arrival state of the re-mint contract — the
dry-run of `sn release --batch` no longer emits the export accounting at its
one-step surface, so the accounting below is read from the identical exporter
leg the release drives (`catalog_release.py` resolves the focus file, then
`run_export(force=True, review_batch=terminal_ids, manifest_sources=...)`),
captured verbatim from that call.

## 1. The release tool names its own target — every identity field

`imas-codex sn release status`, run directly (no pipe, no redirect):

```
ISNC Release Status
  Path: /home/ITER/mcintos/Code/imas-standard-names-catalog
  State: rc
  Latest tag: v0.4.0rc6+west-task-2e
  Batch RC: +west-task-2e (cut against a review batch)
  Remote (origin): git@github.com:Simon-McIntosh/imas-standard-names-catalog.git
  Remote (upstream): git@github.com:iterorganization/imas-standard-names-catalog.git
  GitHub Pages: yes

Available commands:
  sn release -m 'Next RC'  (→ next RC of v0.4.0rc6+west-task-2e)
  sn release --final -m 'Finalize'  (→ stable)
```

All identity fields match the expected canonical resolution: the catalog path,
`State: rc`, the latest tag, the batch build metadata, and both remote values
(origin = the fork, upstream = the organisation repository).

**Series continuity (target canary).** The candidate continues the catalog's
own series: `State: rc` and `Latest tag: v0.4.0rc6+west-task-2e`, and the batch
rehearsal resolves to `v0.4.0rc7+west-task-2e` — the same `0.4.0` version line
and the same `+west-task-2e` batch metadata. No `--bump` is offered for an rc
state, and no unrelated version line appears. The ISNC checkout keeps exactly
`v0.4.0rc2 … v0.4.0rc6+west-task-2e` tags; no `rc7` tag was created.

## 2. Batch dry run over the committed token `west_production_dd_paths`

```
Standard-Name Review Batch
  ISNC: /home/ITER/mcintos/Code/imas-standard-names-catalog
  Batch: .../imas_codex/standard_names/manifests/west_production_dd_paths.yaml
  PR target: fork
  Mode: dry run

Review batch v0.4.0rc7+west-task-2e
  Batch label: +west-task-2e
  Batch size: 220 name(s)
  Unmatched sources: 25
  Artifact: .../manifests/reviews/v0.4.0rc7+west-task-2e.sn_names.yaml
```

The roster it would freeze is reported and not written; the working tree stayed
clean and no RC counter moved. **Candidate count = 220.**

## 3. Export rehearsal — the full accounting

The exporter leg over that batch (the call the release makes at step 3 of
`run_review_release`) returned:

| Field | Value |
|---|---:|
| `total_candidates` (handed-in denominator) | **220** |
| `exported` (published) | **218** |
| `accounted_exclusions` | **2** |
| `accounting_residue` | **0** |
| `emitted_identities` | 218 |

Every exclusion, by mechanism:

| Reason | Count | Identities |
|---|---:|---|
| `invalid_validation_status` | 1 | `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` |
| `name_not_accepted` | 1 | `hot_neutral_temperature` |

Arithmetic reconciliation — the repair that now refuses a cut whose exits are
not accounted:

```
total_candidates 220 == exported 218 + accounted 2  →  closes (residue 0)
```

Gates (all passed, none skipped): `catalog_status`, `graph_tests`,
`score_thresholds`, `cross_field_consistency`, `divergence_detection`,
`identity_token_collision`, **`exclusion_accounting`**, **`manifest_source_accounting`** —
the last two being the totals that define "no silent exit".

Source-level reconciliation also closes over the 355 manifest rows:
**accounted 355 = emitted 328 + excluded 2 + documented_non_nameable 25**,
with the 25 non-nameable rows matching the batch dry-run's "Unmatched sources:
25".

## 4. Comparison against the standing figures (212 emitted, 13 exclusions)

| Figure | Standing (measured under the domain drop) | This rehearsal | Delta |
|---|---:|---:|---:|
| Candidates (handed-in denominator) | 208 | 220 | +12 (fresh mint/denominator) |
| Published | 212 | 218 | +6 |
| Exclusions | 13 | 2 | −11 |

This is a **fresh measurement, not a delta**. The standing 212 was taken while
the exporter still dropped names for an unresolved physics domain — 170
identities became resolvable when the domain-derivation repair landed — and the
successor population the cut depended on drained as the batch review landed. The
−11 exclusions are the same movement: the previously-withheld names now carry an
accepted successor spelling and publish.

## 5. The three questions

**Q1 — are the eighteen names a cut was recorded as losing still lost?**
No silent loss remains; the mechanism is gone. The recorded loss was eighteen
identities whose migration successors sat at `drafted` with `edit_status`
open, so a cut would carry neither spelling. The batch now holds **exactly one**
identity in that blocked state — `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`
(drafted, edit open) — and it is now **excluded under the named mechanism
`invalid_validation_status`** (the graph stores it `quarantined`), so it is
accounted rather than lost. The other 219 handed-in candidates either publish
(218, all `name_stage=accepted`) or are excluded under the second named
mechanism (`hot_neutral_temperature`, `name_not_accepted`, at `reviewed`).
The success of the total accounting (residue 0, `exclusion_accounting` passed)
is what makes "still lost" impossible: any remaining blocked identity would
have to surface as an unaccounted exit and the gate would refuse.

**Q2 — is any exit now uncounted?**
No. `accounted_exclusions = 2`, `accounting_residue = 0`, and the
`exclusion_accounting` gate passed — the arithmetic 220 = 218 + 2 closed over
the full handed-in candidate set. Every exit is named with its mechanism.

**Q3 — does `hot_neutral_temperature` appear published or excluded?**
**Excluded**, under `name_not_accepted` (name_stage `reviewed`, docs `accepted`,
validation `valid`). Its stored `reviewer_score_name` is 0.30, but it is held
out by the acceptance gate rather than the score bar — it has not reached
`accepted` on the name axis.

## 6. No unresolved finding, nothing cut

All eight export gates passed with none skipped; source reconciliation closes;
no prose, byte-identity, validation, semantic, or accounting finding was
reported. No skip-gate flag was used. No pull request was opened, no branch
pushed, no tag cut, no release candidate advanced.
