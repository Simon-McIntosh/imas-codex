# Attachment admission persistence

## Outcome

The governed descriptions are live, deterministically revalidated, receipted,
and replay-safe. The identity lane remains **partially blocked**: the
`poloidal_straight_field_line_angle` fold is fully admissible but was not
applied because the companion `line_integrated_electron_density` fold refused
on an existing distinct successor lineage. No fold was allowed to begin after
the two-fold cohort failed its complete dry-run.

The cross-section disposition is final and must not be re-derived by a later
node. `cross_section_of_flux_surface` remains quarantined and unsourced, and its
existing lineage to the derived umbrella `area_of_flux_surface` remains
untouched. The later attachment manifest must change attachment row 02 directly
to `poloidal_plane_cross_sectional_area_of_flux_surface`. It must **not** fold
`cross_section_of_flux_surface`, fold `area_of_flux_surface`, or rewrite their
lineage. The plain umbrella cannot distinguish poloidal cross-sectional area
from swept flux-surface area; the accepted children
`poloidal_plane_cross_sectional_area_of_flux_surface` and
`surface_area_of_flux_surface` preserve that distinction.

## Separate admissibility measures

The measures deliberately remain separate. Stored lifecycle admissibility is
the attachment guard's binding input; deterministic revalidation is the current
audit result. The audit-precision repair makes those states disagree before
restamping.

| State | Measure A: stored lifecycle | Measure B: deterministic revalidation |
|---|---:|---:|
| Pre-node, whole nine identities | **4/9** | **5/9** |
| Pre-node, executable eight-row scope | **4/8** | **5/8** |
| Current partial state, whole nine identities | **6/9** | **7/9** |
| Current partial state, executable eight-row scope | **6/8** | **7/8** |
| Contracted result after both folds | **8/9** | **8/9** |
| Contracted result in executable scope | **8/8** | **8/8** |

The contracted 8/9 and 8/8 result is not yet live because the line-integrated
identity fold refused. Nine of nine is unreachable by design in this node:
`cross_section_of_flux_surface` is deliberately excluded and remains invalid.
The prospective attachment manifest can still select the accepted canonical
cross-sectional-area child directly; that future manifest selection is not a
license to report the excluded identity as repaired here.

`line_integrated_electron_density` explains the original 4/9 versus 5/9 gap:
the current compound-aware audit accepts it deterministically, while its stored
lifecycle state retained an older quarantine. The restamp was intentionally
limited to the four canonical targets used by the admitted description and fold
routes; it did not wash the deliberately excluded predecessors to valid.

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

## Identity-fold dry-run results

No fold mutation or receipt was written.

| Requested fold | Dry-run result | Disposition |
|---|---|---|
| `line_integrated_electron_density` → `line_integrated_electron_number_density` | **REFUSED**: `name 'line_integrated_electron_density' has another successor lineage; fold is ambiguous` | Stop. Existing successor `line_integrated_electron_density_of_interferometer_beam` is superseded and quarantined, but history is not disposable and the operator correctly refuses a second lineage without an explicit disposition. |
| `poloidal_straight_field_line_angle` → `straight_field_line_angle` | **WOULD APPLY**: zero sources carried, zero projections carried, zero rejected attachments, zero detached attachments, zero sources stranded | Withheld because the complete two-fold cohort did not pass preflight; no partial fold lane was started. |

The cross-section non-fold is separate and final:

| Attachment row | Recorded identity | Later manifest target | Identity mutation |
|---:|---|---|---|
| 02 | `cross_section_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | **None.** Preserve the existing `cross_section_of_flux_surface` ← `area_of_flux_surface` lineage and both accepted area children. |

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

## Remaining exact blocker

Completion now requires an explicit disposition for the existing historical
successor of `line_integrated_electron_density`. The fold operator must not be
weakened, the lineage must not be deleted by hand, and a transitive fold must
not be inferred. Until that disposition is separately authorized, the current
honest live result remains whole-cohort stored 6/9 and deterministic 7/9,
executable-scope stored 6/8 and deterministic 7/8. The two governed description
mutations are complete and replay-safe; the identity lane is not.
