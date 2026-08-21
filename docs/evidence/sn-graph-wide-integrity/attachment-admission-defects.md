# Attachment admission defect ownership

## Verdict

All **5 of 5** blocked attachment identities are **codex-repairable**; **0 of
5** requires a new `imas-standard-names` grammar-policy decision.  The active
public ISN parser is `0.8.0rc66`.  It strictly round-trips four of the five
current spellings.  The fifth, `cross_section_of_flux_surface`, is rejected by
an already-settled section-plane rule and has the strict-valid canonical
replacement `poloidal_plane_cross_sectional_area_of_flux_surface`.

The `cumulative_prefix_check`, `implicit_field_check`,
`repeated_token_check`, and `structural_dim_tag_check` findings are codex
post-generation audits, not ISN grammar findings.  Their false positives must
be corrected in codex; weakening or expanding the closed ISN grammar would
move the defect across the ownership boundary without fixing it.

## Exact five-row disposition

| Attachment row | Defect identity | Exact quoted validator finding | Owner classification | Concrete correction | Named sanctioned route | Why this is not an ISN-policy question |
|---:|---|---|---|---|---|---|
| 02 | `cross_section_of_flux_surface` | Preflight: <q>quarantined: strict grammar round-trip fails</q><br>Validator: `parse_error: grammar round-trip failed for cross_section_of_flux_surface` | **codex-repairable — identity target** | Replace the attachment target with `poloidal_plane_cross_sectional_area_of_flux_surface`.  This names the DD `grid/area` value as poloidal-plane cross-sectional area and keeps it distinct from swept `surface_area_of_flux_surface`. | Regenerate the exact signed `apply_signed_manifest` attachment cohort with the canonical accepted target.  If an identity transition is still needed, propose it through `sn edit` and ordinary review; never alter graph text directly. | ISN already answers the policy question: cross-sectional identities require a `section_plane`, and the replacement strictly round-trips.  No vocabulary or grammar change is needed. |
| 05 | `line_integrated_electron_density` | Preflight: <q>quarantined: `cumulative_prefix_check` rejects `line_integrated`</q><br>Validator: ``audit:cumulative_prefix_check: name 'line_integrated_electron_density' contains 'integrated_' — for DD `_inside`-style quantities use the suffix `_inside_flux_surface` placed after the quantity instead of prefixing with `integrated_`.`` | **codex-repairable — identity target plus audit** | Use the existing accepted `line_integrated_electron_number_density`, which states the counted quantity explicitly and already owns the interferometer/refractometer line-integral family.  Keep `line_integrated` legal: it means integration along the diagnostic beam, not accumulation inside a flux surface. | Correct codex `cumulative_prefix_check` to distinguish the `line_integrated` construction from DD `_inside` accumulation, re-run deterministic validation, and regenerate the signed attachment cohort against the accepted replacement target. | Both the rejected and replacement spellings strictly round-trip in ISN `0.8.0rc66`; the rejecting check is codex-owned and contradicts codex's own line-integrated diagnostic exemplar. |
| 15 | `poloidal_straight_field_line_angle` | Preflight: <q>quarantined: description contains storage-shape tag `2D`; `implicit_field_check` also rejects the spelling</q><br>Validator: `audit:structural_dim_tag_check: description contains storage-shape tag '2D' (remove or rephrase in terms of the physical quantity)`<br>`audit:implicit_field_check: name 'poloidal_straight_field_line_angle' contains bare '_field' after 'straight'; qualify as 'magnetic_field', 'electric_field', etc.` | **codex-repairable — identity target, documentation, and audit** | Reuse accepted `straight_field_line_angle` for `distributions/distribution/profiles_2d/grid/theta_straight`.  Its governed description should say that it is the poloidal angular coordinate of a straight-field-line magnetic coordinate system, in radians; it must not mention `2D` or another storage rank. | Correct codex `implicit_field_check` so the established compound `straight_field_line` is not treated as a bare field quantity; regenerate the description through the ordinary source-hint/compose route, validate deterministically, then regenerate the signed attachment cohort against the accepted identity. | Both `poloidal_straight_field_line_angle` and `straight_field_line_angle` strictly round-trip in the active ISN grammar.  The failures are codex audit and prose-layer findings. |
| 21 | `toroidal_line_integrated_impurity_ion_velocity` | Preflight: <q>quarantined and description absent; the validator cannot claim it</q><br>Validator: ``audit:cumulative_prefix_check: name 'toroidal_line_integrated_impurity_ion_velocity' contains 'integrated_' — for DD `_inside`-style quantities use the suffix `_inside_flux_surface` placed after the quantity instead of prefixing with `integrated_`.`` | **codex-repairable — documentation plus audit** | Retain `toroidal_line_integrated_impurity_ion_velocity`.  Supply a governed description of the toroidal impurity-ion velocity inferred from a charge-exchange channel's line-of-sight-integrated signal, in `m.s^-1`; make explicit that `line_integrated` describes the diagnostic observation and is not an inside-flux-surface cumulative operator. | Correct codex `cumulative_prefix_check`, then use the exact DD source with the sanctioned source-hint and focused ordinary compose/review route to mint the missing description; re-run deterministic validation before the attachment cohort. | The spelling strictly round-trips in ISN `0.8.0rc66`.  The only name finding is emitted by the codex audit, while the other admission defect is absent codex documentation. |
| 29 | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | Preflight: <q>quarantined and description absent; the validator cannot claim it</q><br>Validator: `audit:implicit_field_check: name 'magnetic_field_at_pedestal_top_low_field_side_magnitude' contains bare '_field' after 'low'; qualify as 'magnetic_field', 'electric_field', etc.`<br>`audit:repeated_token_check: name 'magnetic_field_at_pedestal_top_low_field_side_magnitude' contains duplicated content token 'field' — likely tautology` | **codex-repairable — documentation plus audit** | Retain the spelling.  Supply a governed description of the magnitude of the total magnetic field, in tesla, evaluated at the pressure-pedestal-top position determined by the fit on the low-field/outboard side.  Clarify that `low_field_side` is a spatial locus, so its `field` token is neither an unqualified field quantity nor a tautological repetition. | Correct codex `implicit_field_check` and `repeated_token_check` to recognize the registered `low_field_side` locus, then use the exact DD source with the sanctioned source-hint and focused ordinary compose/review route; re-run deterministic validation before attachment. | The full spelling strictly round-trips in ISN `0.8.0rc66`; both name findings are token-insensitive codex audit false positives, and the missing description is graph/pipeline state. |

## Boundary and sequencing

The replacement identities in rows 02, 05, and 15 already express the
authoritative DD meaning without changing ISN grammar.  Rows 21 and 29 need
governed composition because their drafted nodes have no description; direct
`sn edit --docs` is not a shortcut because docs edits require an accepted
identity.  In every row, deterministic validation precedes ordinary review,
and the repaired exact nine-name review cohort remains a single draw: the four
currently valid identities must not be reviewed separately first.

Evidence inputs were the live plan at version 229, the exact target preflight
and its `postvalidation.json`, the source/identity adjudication artifacts, and
the public ISN `0.8.0rc66` parser.  The quantitative artifact check and parser
matrix are recorded in
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T150823557532-admissionadj/validation.log`.
