# Attachment admission persistence

## Outcome

The admissibility repair is complete under the amended zero-fold contract.
Both governed descriptions are live, deterministically revalidated, receipted,
and replay-safe; all four selected quarantines were restamped from the current
deterministic audit. Folds are **0/0 required** and **0 applied**. The two folds
examined during preflight are redundant-identity cleanup, not attachment
admissibility work, and remain deferred for their own adjudication.

Attachment routing, not lineage rewriting, owns the three redirects. The later
attachment manifest must point rows 02, 05, and 15 directly at their canonical
identities. It must not derive any identity fold from those redirects. All
three predecessor lineages remain exactly as found.

## Separate admissibility measures

The measures deliberately remain separate. Stored lifecycle admissibility is
the attachment guard's binding input; deterministic revalidation is the current
audit result. The audit-precision repair makes those states disagree before
restamping.

| State | Measure A: stored lifecycle | Measure B: deterministic revalidation |
|---|---:|---:|
| Pre-node, whole nine identities | **4/9** | **5/9** |
| Post-node, whole nine identities | **6/9** | **7/9** |
| Pre-node, executable eight-row scope | **4/8** | **5/8** |
| Post-node, executable eight-row scope | **6/8** | **7/8** |

The before-to-after transitions are therefore **4/9 to 6/9 stored** and
**5/9 to 7/9 deterministic** for the whole cohort, and **4/8 to 6/8 stored**
and **5/8 to 7/8 deterministic** for the executable scope. The executable
scope excludes row 02's deliberately untouched `cross_section_of_flux_surface`
identity. Nine of nine is not reachable by design in this node because the
three redirects are later attachment-manifest routing decisions, not authority
to rewrite their recorded identities.

`line_integrated_electron_density` explains the original 4/9 versus 5/9 gap:
the current compound-aware audit accepts it deterministically, while its stored
lifecycle state retained an older quarantine. The restamp was intentionally
limited to the four canonical targets used by the admitted description and
direct attachment routes; it did not wash the deliberately excluded
predecessors to valid.

## Identities that remain inadmissible

The remaining divergence is intentional and reported rather than hidden:

| Attachment row | Recorded identity | Stored lifecycle | Current deterministic result | Reason and later route |
|---:|---|---|---|---|
| 02 | `cross_section_of_flux_surface` | Inadmissible | Inadmissible | Strict grammar round-trip failure. It remains quarantined and unsourced. Route directly to `poloidal_plane_cross_sectional_area_of_flux_surface`; do not collapse the derived `area_of_flux_surface` umbrella into either accepted child. |
| 05 | `line_integrated_electron_density` | Inadmissible | **Admissible** | Its stored quarantine predates the compound-aware audit repair. The permanent successor history is untouched. Route directly to `line_integrated_electron_number_density`, which already retains five attached DD producers including `interferometer/channel/n_e_line`. |
| 15 | `poloidal_straight_field_line_angle` | Inadmissible | Inadmissible | Its persisted description still contains a critical storage-shape claim (`2D`). Route directly to the valid `straight_field_line_angle`; do not infer a fold from the attachment redirect. |

Thus Measure A excludes rows 02, 05, and 15, while Measure B excludes rows 02
and 15. In the executable eight-row scope, Measure A excludes rows 05 and 15
and Measure B excludes only row 15.

## Governed descriptions: live before and after

The exact two-row signed authority was generated from the committed staging
artifact at SHA-256
`66b34ac9759bb40c0b07e9bc46229847f60b3b881ece019bb168d675a1245475`.
Authority file SHA-256 is
`3a2a95ee35601d50635e87ace1128f07c41dc638250a1516bc14e225e3cb92e0`,
signed payload SHA-256 is
`159f34777913221d0ebf3495e106b1fc68c0dc5909fd27b2546219f6f982bf76`,
and the live manifest is
`135ef28dae585cbefa9a602b031b2941d81e711dccca0dc846c6874f29b2d3c3`.
Preview admitted 2/2 rows with zero refusals.

| Identity | Before | Live after |
|---|---|---|
| `toroidal_line_integrated_impurity_ion_velocity` | `null` | Toroidal component of the impurity-ion velocity inferred from a charge-exchange diagnostic channel's line-of-sight-integrated signal, expressed in m.s^-1. Here line_integrated describes integration along the diagnostic observation path, not accumulation inside a flux surface. |
| `magnetic_field_at_pedestal_top_low_field_side_magnitude` | `null` | Magnitude of the total magnetic field, expressed in tesla, evaluated at the pressure-pedestal-top position determined by the fit on the low-field (outboard) side. |

The repaired generic operator applied both rows atomically: `changed=2`,
`mutations=2`, `receipt_rows=2`, and `persistent_writes=4`. Replay returned
`already_applied`, `changed=0`, `persistent_writes=0`, and `receipt_rows=2`.

| Identity | Receipt |
|---|---|
| `magnetic_field_at_pedestal_top_low_field_side_magnitude` | `sn-change:signed-manifest:135ef28dae585cbefa9a602b031b2941d81e711dccca0dc846c6874f29b2d3c3:fa75bb21e75ae83d00509a8c` |
| `toroidal_line_integrated_impurity_ion_velocity` | `sn-change:signed-manifest:135ef28dae585cbefa9a602b031b2941d81e711dccca0dc846c6874f29b2d3c3:cbb588efd7eaf900ec591b1c` |

## Exact deterministic restamp

The LLM-free shared audit restamped exactly four canonical targets, with
`cleared=4`, `valid_ids=4`, and `requarantined_ids=[]`:

- `line_integrated_electron_number_density`
- `straight_field_line_angle`
- `toroidal_line_integrated_impurity_ion_velocity`
- `magnetic_field_at_pedestal_top_low_field_side_magnitude`

Every target now has stored `validation_status='valid'`, an empty
`validation_issues` list, and a non-empty description. This was a deterministic
lifecycle synchronization, not name acceptance or review: it created no
`StandardNameReview`, made no provider call, and spent nothing.

## Source and last-producer closure

Description mutation and restamping changed no source scalar or relationship.
The aggregate producer closure for every involved identity remained exactly
`2d0f345b39d0a3eecfc0ea7b2216b3391394b9a4b28bcf2906615961b3983875`.
The exact nine DD-source projection remained exactly
`d003bfcb503186110c168046e114f1918f727df286d255f6ce56790f9ed15dc1`.

The two pre-existing attached sources stayed byte-equivalent and retained their
existing accepted targets:

| Exact source | Status | Scalar and relationship target before and after | Unit |
|---|---|---|---|
| `dd:charge_exchange/channel/ion/velocity_phi` | `attached` | `toroidal_ion_velocity` | `m.s^-1` |
| `dd:summary/pedestal_fits/mtanh/b_field_pedestal_top_lfs/value` | `attached` | `magnetic_field_at_pedestal_top_low_field_side` | `T` |

No target lost an attached source or its last producer. The two refused or
withheld predecessors themselves have no producing sources, while their
canonical targets retain their established closures: five attached DD sources
on `line_integrated_electron_number_density`, and two producers on
`straight_field_line_angle` (one attached DD source and one composed derived
source).

## Deferred identity hygiene and direct routing

No fold mutation or receipt was written because folds are **0/0 required** for
admissibility. The dry-run evidence is retained to prevent a later attachment
node from re-deriving lineage changes from routing decisions.

| Requested fold | Dry-run result | Disposition |
|---|---|---|
| `line_integrated_electron_density` → `line_integrated_electron_number_density` | **REFUSED**: `name 'line_integrated_electron_density' has another successor lineage; fold is ambiguous` | Correct refusal. Existing successor `line_integrated_electron_density_of_interferometer_beam` remains superseded and quarantined, but chain history is permanent. Defer any redundant-identity cleanup for separate adjudication. |
| `poloidal_straight_field_line_angle` → `straight_field_line_angle` | **WOULD APPLY**: zero sources carried, zero projections carried, zero rejected attachments, zero detached attachments, zero sources stranded | Deliberately unapplied. It buys no attachment admissibility and would leave the cohort inconsistently half-folded. Defer redundant-identity cleanup for separate adjudication. |

The attachment-manifest routing contract is explicit for all three redirects:

| Attachment row | Recorded identity | Later manifest target | Identity mutation |
|---:|---|---|---|
| 02 | `cross_section_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | **None.** Preserve the existing `cross_section_of_flux_surface` ← `area_of_flux_surface` lineage and both accepted area children. |
| 05 | `line_integrated_electron_density` | `line_integrated_electron_number_density` | **None.** Preserve the existing successor chain even though its surviving successor is itself superseded and quarantined. |
| 15 | `poloidal_straight_field_line_angle` | `straight_field_line_angle` | **None.** Route the attachment directly; do not turn the redirect into a lineage fold. |

## Review, cost, and counter proof

| Counter | Before | Current | Delta |
|---|---:|---:|---:|
| `StandardNameChange` | 7,754 | 7,756 | **+2**, exactly the two description receipts |
| `LLMCost` rows | 27,631 | 27,631 | **0** |
| `LLMCost.llm_cost` total | USD 1,366.843569 | USD 1,366.843569 | **USD 0.000000** |
| Global `StandardNameReview` | 20,754 | 20,754 | **0** |
| Historical review rows attached to the target scope | 20 | 20 | **0** |

Ordinary-review draws for the nine-target cohort: **0**. Provider calls: **0**.
Attributable spend: **USD 0.000000 of USD 25**.

## Completion boundary

The node is complete: descriptions are **2/2 applied, 2/2 receipted, and 2/2
replayed**; the exact restamp is **4/4 valid**; folds are **0/0 required** and
**0 applied**. The honest live result is whole-cohort stored 6/9 and
deterministic 7/9, with executable-scope stored 6/8 and deterministic 7/8.

The later attachment-manifest node must route rows 02, 05, and 15 directly to
the canonical targets recorded above while preserving the unchanged producer
and exact-source closures. The two dry-run folds remain redundant-identity
cleanup requiring separate adjudication. They are not blockers to this
admissibility result, and neither may be inferred from the attachment routes.
