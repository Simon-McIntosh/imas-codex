# Governed sign-convention document repair

Snapshot: 2026-08-25, live `codex` graph. Repair code commits: `ed4e27e0`,
`1795bdd2`. The operation made no model call.

## Outcome

The signed repair manifest covered all **280 deterministic identities** named by
the diagnosis. Its SHA-256 was
`0bd850b49c9f549c304d6716a66746816028ebdaca617b8260b78f8bb58ee0b7`.
Apply changed **279 documents** atomically and recorded
`sn-change:sign-convention-document-repair:0bd850b49c9f549c304d6716a66746816028ebdaca617b8260b78f8bb58ee0b7`.
`plasma_beta` was already clean when the live preview ran, so it was signed as a
covered no-op rather than rewritten or reported as a mutation.

| Disposition | Count | Applied change |
|---|---:|---|
| Documentation only | **226** | Removed the one unsupported final sign-convention paragraph; retained `one_like`, scalar COCOS 17 and the single `HAS_COCOS` edge |
| Documentation plus metadata | **53** | Removed the same paragraph; cleared `cocos_transformation_type`, scalar `cocos` and the single `HAS_COCOS` edge |
| Already clean | **1** | `plasma_beta`; no mutation |
| Regeneration required and excluded | **1** | `magnetic_field`; retained unchanged for `b0_like`-grounded regeneration and ordinary documentation quorum |

Representative before/after identities show the two admitted mechanisms. For
`accumulated_deuterated_methane_prefill_count`, the retained document ends with
the existing relationship to
`accumulated_total_prefill_gas_count`; only the trailing paragraph “Sign
convention: Positive when deuterated methane has been injected during the
prefill phase.” was removed. Its `one_like` and COCOS 17 authority remain. For
`area_of_flux_surface`, the trailing sign paragraph was removed and the stale
`one_like` property, scalar COCOS value and edge were cleared together because
structural authority recomputes no unique transformation class.

## Gate evidence

The diagnosis baseline over the deterministic cohort was **280 fail**. A fresh
preview immediately before apply was **1 pass / 279 fail / 0 not-evaluable**,
because `plasma_beta` had concurrently become clean. The post-apply conditional
sign gate over the same 280 exact identities is:

| Outcome | Before apply | After apply |
|---|---:|---:|
| Pass | 1 | **227** |
| Fail | 279 | **0** |
| Not evaluable | 0 | **53** |

The 53 abstentions are intentional: once unsupported transformation metadata is
cleared, the gate must not invent sign authority from absence. Candidate and
property coverage was checked before trusting the zero: **2,740 accepted
documents, 2,740 ids and 2,740 documentation values**; the scoped query returned
all **280 of 280** requested ids.

The live graph-wide rescore at the same instant was 325 pass / 4 fail / 2,411
not-evaluable. That is not a failure of the signed cohort. The remaining four
are the deliberately excluded `magnetic_field` plus three identities that
became accepted outside the frozen diagnosis while this node ran:
`length_of_poloidal_magnetic_field_probe`,
`radial_coordinate_at_inboard_midplane`, and
`ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density_and_neutral_density_of_isotope`.
They require a fresh authority diagnosis before any mutation; this receipt does
not silently expand to include them.

## Collateral bound and spend

Every manifest row signs the prior and replacement document hashes, lengths,
node and edge identities, exact removed suffix, and whether metadata may be
cleared. All **279 of 279** current document hashes match the signed after-state.
For every changed document:

`before == after + paragraph_separator + removed_sign_paragraph`

The equality holds on all rows. The largest per-document character delta is
**545**, the total is **43,542**, and no character before the final paragraph
separator changed. Metadata postconditions are **53/53** cleared and **226/226**
preserved. The durable receipt links to 279 changed identities; the one no-op
correctly carries no change edge.

The run-scoped `LLMCost` census is **0 calls / USD 0.00**. Preview, apply and
rescore are deterministic graph and local-Python operations only.

## Evidence files

- `live-preview.json`: signed pre-apply manifest, 280 covered, zero refusals.
- `live-apply.json`: committed receipt and exact per-row deltas.
- `live-verification.json`: independent post-apply gate, graph-state and cost
  checks.
- `focused-tests.log`: repair, gate and audit regression results.

These files are under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T011436076292-n-signstockrepair/`.
