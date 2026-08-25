# Conditional sign residual repair

Live repair and postflight: `2026-08-25`, default `codex` graph. Repair
operator commit: `7690d435`.

## Outcome

The graph-wide conditional sign gate now has **zero failures** over every
accepted document. Property coverage was checked before trusting that zero:
**2,952 candidates, 2,952 with `id`, and 2,952 with documentation**. The full
outcome vector moved as follows:

| Outcome | Before | After | Change |
|---|---:|---:|---:|
| Pass | 333 | **351** | +18 |
| Fail | 18 | **0** | -18 |
| Not evaluable | 2,601 | **2,601** | 0 |
| Total accepted documents | 2,952 | **2,952** | 0 |

The eighteen repairs followed their own diagnosed authority routes rather than
one blanket edit: **17 exact-hash documentation repairs plus one signed metadata
compare-and-set**. The run made **0 model calls and spent USD 0.00**.

## Invariant-document route: 17

The seventeen identities whose stored and recomputed classes both equal
`one_like` used the existing governed document-repair operator. Its preview
signed the graph element, exact before and after documentation hashes, retained
COCOS authority, and character delta for every identity. Apply re-derived the
same manifest under locks before writing.

- Signed manifest:
  `c6bb276e9e57f18dc43ebc5a0967f13b983b2a01b551be6fd36b7f550ca70763`.
- Durable change:
  `sn-change:sign-convention-document-repair:c6bb276e9e57f18dc43ebc5a0967f13b983b2a01b551be6fd36b7f550ca70763`.
- Admitted and changed: **17/17**; refusals: **0**; already clean: **0**.
- Removed text: **2,649 characters total**, at most **313** for one identity.
- Every delta obeyed
  `before == after + paragraph_separator + removed_final_paragraph`; no prefix
  byte changed.
- Postflight: **17/17** current document hashes equal the signed after-state;
  **17/17** retain `one_like`, scalar COCOS 17, and one `HAS_COCOS` edge to 17.

Representative repaired identities include `ion_average_temperature`,
`parallel_current_density_due_to_ohmic_current_drive`,
`toroidal_helium_3_velocity_at_plasma_boundary`, and
`z_minor_axis_unit_vector_of_shatter_cone`. Their names, descriptions, source
bindings, review history, and transformation metadata were not edited.

## Structural-metadata route: 1

`magnetic_field` took the inverse route. Its accepted documentation is correct
for `b0_like`; the stale stored `one_like` scalar was the contradiction. The
new governed operator signed the target graph identity, accepted-document hash,
COCOS edge, and complete direct-child closure before changing only that scalar.

- Signed manifest:
  `7dffdad96fc28b2d5d6fab2377726dd270c90914082fe04425dcfb9a51b37bd8`.
- Durable change:
  `sn-change:cocos-transformation-metadata-repair:7dffdad96fc28b2d5d6fab2377726dd270c90914082fe04425dcfb9a51b37bd8`.
- Structural closure: **13** direct children, **12** eligible after the shared
  derived-parent exclusions, and exactly one non-null eligible class:
  `b0_like` from `vacuum_magnetic_field`.
- Compare-and-set: `one_like` to `b0_like`; scalar COCOS remained 17 and the
  original `HAS_COCOS` edge remained targeted at COCOS 17.
- Documentation remained byte-identical at SHA-256
  `c89b0191211dc0d0e24350d18cc69d00cfcde794deac3903d063312404ff168b`
  over **1,869 characters**.

The postflight rescored that unchanged document against the corrected stored
class and it passes as a COCOS-sensitive quantity with canonical final sign
prose. No documentation review, regeneration, or acceptance transition was
performed.

## Reproducible records

The run-scoped transcripts are:

- `live-preview.log` — both exact signed previews;
- `live-apply.log` — both applied receipts and route counts;
- `graphwide-postflight.log` — property coverage, all 2,952 gate outcomes,
  signed postconditions, retained authority, and the zero-call cost census.

They are stored under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T080940064739-n-signresidualrepair/`.
